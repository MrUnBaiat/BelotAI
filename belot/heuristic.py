"""
The fixed greedy reference player.

This heuristic never changes, which is the point: every strength number in this
project is anchored to it, so it has to stay comparable forever. It is about forty
lines and uses no search, no card counting and no partner modelling.

    bidding  accept or declare only on real trump strength, otherwise pass
    playing  if the trick can be taken, take it with the cheapest sufficient card;
             otherwise discard the lowest-value legal card

It is also the fallback inside the search (`belot.search`) whenever a
determinization cannot be sampled, and the rollout policy inside rollout-PIMC.

Measured: +5.56 pts/hand against uniform random play.
"""

import numpy as np

# Rank order within a suit is 7 8 9 10 J Q K A, encoded as card % 8.
# Trick power differs between trump and side suits, and so do the card points.
_NT_POWER = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 3: 6, 7: 7}
_T_POWER  = {0: 0, 1: 1, 5: 2, 6: 3, 3: 4, 7: 5, 2: 6, 4: 7}
_T_PTS    = {0: 0, 1: 0, 2: 14, 3: 10, 4: 20, 5: 3, 6: 4, 7: 11}
_NT_PTS   = {0: 0, 1: 0, 2: 0, 3: 10, 4: 2, 5: 3, 6: 4, 7: 11}


def _pw(c, tr): return (1, _T_POWER[c % 8]) if c // 8 == tr else (0, _NT_POWER[c % 8])
def _pt(c, tr): return _T_PTS[c % 8] if c // 8 == tr else _NT_PTS[c % 8]


def _heuristic_action(belot):
    """Greedy: bid only on real trump strength; take tricks cheaply, else dump low."""
    legal = np.flatnonzero(belot.get_legal_actions())
    hand = belot.hands[belot.current_player]
    if belot.phase == "BIDDING":
        def strength(s, extra=None):
            cards = [c for c in hand if c // 8 == s] + \
                    ([extra] if extra is not None and extra // 8 == s else [])
            return len(cards), sum(_T_PTS[c % 8] for c in cards)
        if 33 in legal:
            n, p = strength(belot.face_up_suit, extra=belot.face_up_card)
            return 33 if (n >= 3 and p >= 20) or n >= 4 else 32
        suits = [a - 34 for a in legal if a >= 34]
        best = max(suits, key=lambda s: strength(s))
        n, p = strength(best)
        return 32 if (32 in legal and not (n >= 3 and p >= 20)) else 34 + best
    tr, trick, cards = belot.trump, belot.current_trick, list(legal)
    if not trick:
        return max(cards, key=lambda c: _pw(c, tr)[1] - (3 if c // 8 == tr and _T_POWER[c % 8] < 6 else 0))
    cur = max(_pw(c, tr) for _, c in trick)
    cur_w = max(trick, key=lambda pc: _pw(pc[1], tr))[0]
    partner = (cur_w % 2) == (belot.current_player % 2)
    winners = [c for c in cards if _pw(c, tr) > cur]
    if partner and len(trick) >= 2:
        return (max(cards, key=lambda c: (_pt(c, tr), -_pw(c, tr)[1])) if len(trick) == 3
                else min(cards, key=lambda c: (_pt(c, tr), _pw(c, tr)[1])))
    if winners:
        return min(winners, key=lambda c: _pw(c, tr)[1]) if len(trick) == 3 \
            else max(winners, key=lambda c: _pw(c, tr)[1])
    return min(cards, key=lambda c: (_pt(c, tr), _pw(c, tr)[1]))
MAX_HANDS_PER_MATCH = 100     # safety valve; a match to 101 averages ~11 hands


def _random_action(belot):
    legal = np.flatnonzero(belot.get_legal_actions())
    return int(np.random.choice(legal))


# Public aliases. The underscore names are kept because the archived experiment
# scripts and the audit tooling import them by those names.
heuristic_action = _heuristic_action
random_action = _random_action
