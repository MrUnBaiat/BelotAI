"""
Evaluation harness.

Training losses (actor/critic/entropy) tell you the optimizer is doing
*something*; they do NOT tell you the agent is getting stronger. In self-play
with a shared policy you can drive losses down while the policy quietly collapses
or cycles. So we measure strength directly against fixed reference opponents.

The learned policy controls team 0 (seats 0 & 2); the opponent controls team 1
(seats 1 & 3). We report:
  - win_rate      : fraction of hands where team 0 outscores team 1
  - avg_point_diff: mean (team0_game_points - team1_game_points) per hand

Opponent is a uniform-random legal player by default, or a frozen past model
(pass `frozen=`) to track improvement against previous self.

Each hand is played in a neutral match context (match_scores=[0,0], fresh bolts).
That is a mild distribution shift from training, but it keeps the metric a clean,
comparable scalar across checkpoints.
"""

import numpy as np
import torch

from env import BelotEnv
from observation import build_observation

HIDDEN = 512


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


@torch.no_grad()
def evaluate(model, num_games=200, device="cpu", frozen=None, greedy=True):
    """Play `num_games` hands of team0=model vs team1=(frozen or random)."""
    model.eval()
    if frozen is not None:
        frozen.eval()

    point_diffs = []
    wins = 0

    for _ in range(num_games):
        belot = BelotEnv()
        match_scores = [0, 0]
        hc = {s: _zero_state(device) for s in range(4)}  # per-seat LSTM state
        info = {}

        while not belot.done:
            seat = belot.current_player
            if seat % 2 == 0:                      # team 0 -> learned policy
                a, hc[seat] = _net_action(model, belot, seat, hc[seat],
                                          match_scores, device, greedy)
            elif frozen is not None:               # team 1 -> frozen reference
                a, hc[seat] = _net_action(frozen, belot, seat, hc[seat],
                                          match_scores, device, greedy)
            else:                                  # team 1 -> random legal
                a = _random_action(belot)
            _, _, _, info = belot.step(a)

        gp = info.get("game_points", [0, 0, 0, 0])
        point_diffs.append(gp[0] - gp[1])
        wins += int(gp[0] > gp[1])

    model.train()
    return {
        "win_rate": wins / num_games,
        "avg_point_diff": float(np.mean(point_diffs)),
    }