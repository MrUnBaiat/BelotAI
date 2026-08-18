"""
Evaluation harness -- full matches to 101, not isolated hands.

Training losses tell you the optimizer is doing *something*; they do NOT tell you
the agent is getting stronger. In self-play with a shared policy you can drive
losses down while the policy quietly collapses or cycles, so strength is measured
directly against fixed reference opponents.

v2 changes, all motivated by measured defects in the old per-hand metric:
  * MATCHES TO 101, not single hands. Bolt counters persist within a match (so
    the 3rd-bolt -10 rule finally exists in eval) and the RUNNING match score is
    fed into build_observation -- the score-aware features were previously dead
    at eval time.
  * DEALER ROTATES across matches. Pinning dealer=0 handed the model team the
    dealing seat every time, worth ~3 points of win rate under random play.
  * TIES ARE COUNTED, not silently scored as losses (~6% of hands end 8-8).
  * The reference opponent can be a frozen model, so progress is tracked against
    a fixed yardstick rather than against a moving self.

The learned policy controls team 0 (seats 0 & 2); the opponent controls team 1
(seats 1 & 3). LSTM state resets each hand, matching training semantics.
"""

import numpy as np
import torch

from env import BelotEnv
from observation import build_observation

HIDDEN = 512

# --- fixed greedy reference player (never changes, so it is comparable forever) ---
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


def _zero_state(device):
    return (torch.zeros(1, 1, HIDDEN, device=device),
            torch.zeros(1, 1, HIDDEN, device=device))


@torch.no_grad()
def _net_action(net, belot, seat, hc, match_scores, device, greedy):
    local, glob, mask = build_observation(belot, seat, match_scores)
    lt = torch.from_numpy(local).unsqueeze(0).to(device)
    gt = torch.from_numpy(glob).unsqueeze(0).to(device)
    mt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
    dist, _, new_hc = net(lt, gt, hc, mt, is_sequence=False)
    action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
    return int(action.item()), new_hc


def _random_action(belot):
    legal = np.flatnonzero(belot.get_legal_actions())
    return int(np.random.choice(legal))


_PIMC_CACHE = {}


def _pimc_action(belot, D=16):
    """Lazily-built PIMC opponent -- a search player with no learning, used as a
    yardstick ABOVE the model. Built on demand so importing eval.py stays cheap
    and so nothing pays for it unless it is actually used."""
    if D not in _PIMC_CACHE:
        from pimc import make_pimc
        _PIMC_CACHE[D] = make_pimc(D=D, seed=0)
    return _PIMC_CACHE[D](belot)


@torch.no_grad()
def evaluate_matches(model, num_matches=100, device="cpu", opponent="random",
                     frozen=None, greedy=True, pimc_D=16):
    """
    Play `num_matches` complete matches to 101.

    opponent : "random"    uniform-random legal play (saturates fast; keep only
                           as a floor check)
               "heuristic" fixed greedy player -- an absolute yardstick that does
                           not saturate and never changes across runs
               "model"     a reference network, passed via `frozen`

    Returns match_win_rate / match_tie_rate, avg_hand_diff (mean per-hand game
    point difference -- the low-variance strength scalar), hand_win_rate,
    avg_hands_per_match, avg final scores, and declarer bolt counts per team.
    """
    model.eval()
    use_frozen = (opponent == "model" and frozen is not None)
    if use_frozen:
        frozen.eval()

    match_wins = match_ties = 0
    hand_diffs, hand_wins, hands_per_match, finals = [], 0, [], []
    n_hands_total = 0
    bolts = [0, 0]

    for m in range(num_matches):
        belot = BelotEnv()
        belot.dealer = m % 4          # rotate the first dealer across matches
        belot.reset()
        match_scores = [0, 0]
        hands = 0

        while match_scores[0] < 101 and match_scores[1] < 101 and hands < MAX_HANDS_PER_MATCH:
            hc = {s: _zero_state(device) for s in range(4)}   # per-seat LSTM state
            info = {}
            while not belot.done:
                seat = belot.current_player
                if seat % 2 == 0:                      # team 0 -> learned policy
                    a, hc[seat] = _net_action(model, belot, seat, hc[seat],
                                              match_scores, device, greedy)
                elif use_frozen:                       # team 1 -> frozen reference
                    a, hc[seat] = _net_action(frozen, belot, seat, hc[seat],
                                              match_scores, device, greedy)
                elif opponent == "heuristic":          # team 1 -> fixed greedy
                    a = _heuristic_action(belot)
                elif opponent == "pimc":               # team 1 -> search player
                    a = _pimc_action(belot, pimc_D)
                else:                                  # team 1 -> random legal
                    a = _random_action(belot)
                _, _, _, info = belot.step(a)

            gp = info.get("game_points", [0, 0, 0, 0])
            if belot.declaring_team is not None and \
                    belot.raw_points_by_team[belot.declaring_team] <= 80:
                bolts[belot.declaring_team] += 1
            match_scores[0] += gp[0]
            match_scores[1] += gp[1]
            hand_diffs.append(gp[0] - gp[1])
            hand_wins += int(gp[0] > gp[1])
            hands += 1
            n_hands_total += 1
            if match_scores[0] < 101 and match_scores[1] < 101:
                belot.reset()          # bolts persist within a match; step() rotated the dealer

        match_wins += int(match_scores[0] > match_scores[1])
        match_ties += int(match_scores[0] == match_scores[1])
        hands_per_match.append(hands)
        finals.append(match_scores)

    model.train()
    finals = np.array(finals, dtype=np.float64)
    p = match_wins / num_matches
    return {
        "match_win_rate": p,
        "match_tie_rate": match_ties / num_matches,
        "match_win_ci95": float(1.96 * np.sqrt(max(p * (1 - p), 1e-9) / num_matches)),
        "avg_hand_diff": float(np.mean(hand_diffs)),
        "hand_diff_ci95": float(1.96 * np.std(hand_diffs) / np.sqrt(len(hand_diffs))),
        "hand_win_rate": hand_wins / max(n_hands_total, 1),
        "avg_hands_per_match": float(np.mean(hands_per_match)),
        "avg_final_scores": [float(finals[:, 0].mean()), float(finals[:, 1].mean())],
        "declarer_bolts": bolts,
    }


def evaluate(model, num_games=200, device="cpu", frozen=None, greedy=True):
    """Backwards-compatible shim: `num_games` is now interpreted as matches."""
    res = evaluate_matches(model, num_matches=max(num_games // 10, 10), device=device,
                           opponent="model" if frozen is not None else "random",
                           frozen=frozen, greedy=greedy)
    return {"win_rate": res["match_win_rate"], "avg_point_diff": res["avg_hand_diff"]}