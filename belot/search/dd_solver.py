"""
Exact double-dummy solver for Belot card play.

WHY. Every strong program in this game family (Bridge, Skat, Hearts) is built on
PIMC with an EXACT perfect-information solve at each determinization. This project's
`pimc.py` instead rolls out with the greedy heuristic, which is a much weaker
evaluator -- and Phase 5 measured the consequence: PIMC(D=32) beats the model's card
play by only +0.845 +- 0.400, and the returns from more determinizations are already
saturating (+0.313 then +0.167 per doubling of D). A weak evaluator caps the whole
approach, including anything distilled from it.

Belot after the bidding is an 8-trick, 32-card perfect-information game -- far smaller
than Bridge's 13 tricks, which production solvers do in milliseconds. So an exact
solver should be affordable, and this file measures whether it is.

RULES MIRRORED EXACTLY from env.py (verified by `selftest`, which cross-checks legal
move sets, trick winners and final点 against the real env on random games):

  * trick power, non-trump  7<8<9<J<Q<K<10<A ; trump  7<8<Q<K<10<A<9<J
  * points, trump  J=20 9=14 A=11 10=10 K=4 Q=3 8=7=0
    points, non-trump  A=11 10=10 K=4 Q=3 J=2 9=8=7=0
  * must follow suit; if void, MUST ruff if holding trump -- even over a winning
    partner; and MUST OVER-ruff if able (env.py:163-170)
  * the over-ruff obligation also applies when FOLLOWING a led trump (env.py:154-160)
  * a non-declarer may not LEAD trump until the declarer has played one, unless the
    non-declarer holds nothing but trumps (env.py:124-127)
  * the winner of the 8th trick takes +10

VALUE. The solver returns TEAM 0's raw card points for the remainder of the hand,
including the last-trick bonus. Team 0 maximises, team 1 minimises. Raw points are what
the game-point conversion (bolt threshold, bile) is computed from, so this is the
correct quantity to search on.
"""
import sys

# rank index: 0=7 1=8 2=9 3=10 4=J 5=Q 6=K 7=A
PTS_T = (0, 0, 14, 10, 20, 3, 4, 11)
PTS_N = (0, 0, 0, 10, 2, 3, 4, 11)
POW_T = (0, 1, 6, 4, 7, 2, 3, 5)
POW_N = (0, 1, 2, 6, 3, 4, 5, 7)

SUIT = tuple(c >> 3 for c in range(32))
RANK = tuple(c & 7 for c in range(32))
SUIT_MASK = tuple(0xFF << (8 * s) for s in range(4))


def bits(m):
    out = []
    while m:
        b = m & -m
        out.append(b.bit_length() - 1)
        m ^= b
    return out


def card_pts(card, trump):
    return PTS_T[RANK[card]] if SUIT[card] == trump else PTS_N[RANK[card]]


def card_pow(card, trump):
    return POW_T[RANK[card]] if SUIT[card] == trump else POW_N[RANK[card]]


def trick_winner(trick, trump):
    """trick = ((player, card), ...) in play order. Mirrors env.py:313-333."""
    led_suit = SUIT[trick[0][1]]
    best_p, best_rank, best_is_trump = None, -1, False
    for p, c in trick:
        s = SUIT[c]
        is_t = (s == trump)
        rv = POW_T[RANK[c]] if is_t else POW_N[RANK[c]]
        if not is_t and not best_is_trump and s == led_suit:
            if rv > best_rank:
                best_rank, best_p = rv, p
        elif is_t and best_is_trump:
            if rv > best_rank:
                best_rank, best_p = rv, p
        elif is_t and not best_is_trump:
            best_is_trump = True
            best_rank, best_p = rv, p
    return best_p


def legal_moves(hand, trick, trump, cp, declarer, dhpt):
    """Exact mirror of env.get_legal_actions()'s PLAYING branch."""
    tm = SUIT_MASK[trump]
    if not trick:
        if not dhpt and cp != declarer and (hand & ~tm):
            return bits(hand & ~tm)          # may not lead trump yet
        return bits(hand)

    led = SUIT[trick[0][1]]
    led_mask = hand & SUIT_MASK[led]
    trump_in_hand = hand & tm

    high_t = -1
    for _, c in trick:
        if SUIT[c] == trump:
            v = POW_T[RANK[c]]
            if v > high_t:
                high_t = v
    over = [c for c in bits(trump_in_hand) if POW_T[RANK[c]] > high_t]

    if led_mask:
        if led == trump:
            return over if over else bits(led_mask)
        return bits(led_mask)
    if trump_in_hand:
        return over if over else bits(trump_in_hand)
    return bits(hand)


class DD:
    """Alpha-beta with a transposition table. One instance per solve tree."""

    __slots__ = ("trump", "declarer", "tt", "nodes")

    def __init__(self, trump, declarer):
        self.trump = trump
        self.declarer = declarer
        self.tt = {}
        self.nodes = 0

    def _rem_pts(self, hands, trick):
        t = 0
        for h in hands:
            for c in bits(h):
                t += card_pts(c, self.trump)
        for _, c in trick:
            t += card_pts(c, self.trump)
        return t + 10

    def search(self, hands, cp, trick, dhpt, alpha, beta):
        """Team 0's future raw points (incl. last-trick 10) under optimal play."""
        self.nodes += 1
        if not hands[0] and not hands[1] and not hands[2] and not hands[3] and not trick:
            return 0

        key = (hands, cp, trick, dhpt)
        hit = self.tt.get(key)
        if hit is not None:
            val, lo, hi = hit
            if lo >= beta:
                return lo
            if hi <= alpha:
                return hi
            if lo == hi:
                return val

        maxi = (cp % 2 == 0)
        # bound: team0 can gain at most every remaining point, at least none
        if maxi:
            if alpha >= self._rem_pts(hands, trick):
                return alpha
        else:
            if beta <= 0:
                return beta

        moves = legal_moves(hands[cp], trick, self.trump, cp, self.declarer, dhpt)
        # move ordering: highest trick power first -- wins tricks early, prunes hard
        moves.sort(key=lambda c: card_pow(c, self.trump), reverse=True)

        a0, b0 = alpha, beta
        best = -1 if maxi else 10 ** 9
        for c in moves:
            nh = list(hands)
            nh[cp] &= ~(1 << c)
            nt = trick + ((cp, c),)
            ndh = dhpt or (SUIT[c] == self.trump and cp == self.declarer)
            if len(nt) == 4:
                w = trick_winner(nt, self.trump)
                pts = sum(card_pts(cc, self.trump) for _, cc in nt)
                last = not (nh[0] or nh[1] or nh[2] or nh[3])
                if last:
                    pts += 10
                gain = pts if w % 2 == 0 else 0
                v = gain + self.search(tuple(nh), w, (), ndh,
                                       alpha - gain, beta - gain)
            else:
                v = self.search(tuple(nh), (cp + 1) % 4, nt, ndh, alpha, beta)

            if maxi:
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            else:
                if v < best:
                    best = v
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break

        if best <= a0:
            self.tt[key] = (best, -(10 ** 9), best)
        elif best >= b0:
            self.tt[key] = (best, best, 10 ** 9)
        else:
            self.tt[key] = (best, best, best)
        return best


def solve_root(hands, cp, trick, trump, declarer, dhpt, max_pts=162):
    """Exact team-0 value of EVERY legal move at the root.

    Returns (best_card, {card: team0_points}, nodes). One tree, reused TT."""
    dd = DD(trump, declarer)
    out = {}
    for c in legal_moves(hands[cp], trick, trump, cp, declarer, dhpt):
        nh = list(hands)
        nh[cp] &= ~(1 << c)
        nt = trick + ((cp, c),)
        ndh = dhpt or (SUIT[c] == trump and cp == declarer)
        if len(nt) == 4:
            w = trick_winner(nt, trump)
            pts = sum(card_pts(cc, trump) for _, cc in nt)
            if not (nh[0] or nh[1] or nh[2] or nh[3]):
                pts += 10
            gain = pts if w % 2 == 0 else 0
            out[c] = gain + dd.search(tuple(nh), w, (), ndh, -1, max_pts + 1)
        else:
            out[c] = dd.search(tuple(nh), (cp + 1) % 4, nt, ndh, -1, max_pts + 1)
    team = cp % 2
    best = (max if team == 0 else min)(out, key=lambda k: out[k])
    return best, out, dd.nodes


def hands_to_masks(hands):
    return tuple(sum(1 << c for c in h) for h in hands)
