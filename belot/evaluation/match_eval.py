"""
Match-level evaluation -- full matches to 101, not isolated hands.

Training losses tell you the optimizer is doing something; they do not tell you the
agent is getting stronger. In self-play with a shared policy you can drive losses
down while the policy quietly collapses or cycles, so strength is measured directly
against fixed reference opponents.

Three details that were measured defects in an earlier per-hand metric:

  * MATCHES TO 101, not single hands. Bolt counters persist within a match, so the
    third-bolt penalty finally exists at eval time, and the running match score is
    fed into `build_observation` -- those score-aware features were previously dead.
  * THE DEALER ROTATES across matches. Pinning dealer=0 handed the model team the
    dealing seat every time, worth about three points of win rate under random play.
  * TIES ARE COUNTED rather than silently scored as losses; roughly 6% of hands
    end 8-8.

The learned policy controls team 0 (seats 0 and 2); the opponent controls team 1
(seats 1 and 3). LSTM state resets each hand, matching training semantics.

For comparisons finer than about 0.5 pts/hand use `swap_eval` instead -- this
harness is unbiased but roughly four times noisier.
"""

import numpy as np
import torch

from belot.env import BelotEnv
from belot.heuristic import _heuristic_action, _random_action
from belot.observation import build_observation

HIDDEN = 512
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


_PIMC_CACHE = {}


def _pimc_action(belot, D=16):
    """Lazily-built PIMC opponent -- a search player with no learning, used as a
    yardstick ABOVE the model. Built on demand so importing eval.py stays cheap
    and so nothing pays for it unless it is actually used."""
    if D not in _PIMC_CACHE:
        from belot.search.pimc import make_pimc
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
