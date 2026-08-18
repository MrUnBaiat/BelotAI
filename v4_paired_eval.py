"""
Paired isolated-hand evaluator -- the low-variance instrument.

AUDIT_HANDOFF section 8.3: the policy is deterministic under greedy action
selection and the greedy heuristic is deterministic, so re-seeding numpy
reproduces the identical deal. Comparing two policies on the SAME deals removes
the card-luck term, which EXP-6 measured at 73.8% of total outcome variance --
so a paired comparison has roughly 1/sqrt(1-0.738) ~ 2x smaller error than an
unpaired one, before counting the extra correlation from shared play.

evaluate_matches() is still the number to quote for ABSOLUTE strength (+2.37 vs
the heuristic), because it plays real matches to 101 with bolts and the running
score live in the observation. This module is for DIFFERENCES between conditions.

One hand per deal seed, dealer rotated across seeds, match score 0-0, bolts 0-0.
Model plays seats 0 & 2, the fixed greedy heuristic plays seats 1 & 3.
"""
import numpy as np
import torch

from env import BelotEnv
from eval import _heuristic_action
from observation import build_observation

HIDDEN = 512


@torch.no_grad()
def hand_diffs(model, n_deals, device, base_seed=770_000, greedy=True):
    """Per-deal (gp_team0 - gp_team1) for `n_deals` fixed deals."""
    model.eval()
    out = np.empty(n_deals, dtype=np.float64)
    for d in range(n_deals):
        env = BelotEnv()
        env.dealer = d % 4
        np.random.seed(base_seed + d)          # fixes the deal, and only the deal
        env.reset()
        env.bolts_by_team = [0, 0]
        scores = [0, 0]
        hc = {s: (torch.zeros(1, 1, HIDDEN, device=device),
                  torch.zeros(1, 1, HIDDEN, device=device)) for s in range(4)}
        info = {}
        while not env.done:
            seat = env.current_player
            if seat % 2 == 0:
                local, glob, mask = build_observation(env, seat, scores)
                dist, _, hc[seat] = model(
                    torch.from_numpy(local).unsqueeze(0).to(device),
                    torch.from_numpy(glob).unsqueeze(0).to(device),
                    hc[seat],
                    torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device),
                    is_sequence=False)
                a = int(dist.probs.argmax(-1).item() if greedy else dist.sample().item())
            else:
                a = _heuristic_action(env)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        out[d] = gp[0] - gp[1]
    model.train()
    return out


def summarise(name, d):
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return f"{name:<28s} {d.mean():+.3f} +- {ci:.3f} pts/hand   (n={len(d)})"


def paired(name, a, b):
    """a - b on identical deals, with a paired CI and a sign test."""
    d = a - b
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    nz = d[d != 0]
    wins = int((nz > 0).sum())
    verdict = "SIGNIFICANT" if abs(d.mean()) > ci else "not significant"
    return (f"{name:<34s} {d.mean():+.3f} +- {ci:.3f}   "
            f"[{verdict}]  deals changed {len(nz)}/{len(d)}, "
            f"better in {wins}/{max(len(nz), 1)}")
