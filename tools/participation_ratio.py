"""
PHASE 5 -- gradient participation of a checkpoint.

Reproduces `audit_v9/p3_04_gradient_participation.py`'s central statistic so a
distilled checkpoint can be compared to the starting one on the quantity that names
the plateau's mechanism.

For a softmax policy the score function of the sampled action is
`d log pi(a)/dz_b = [b==a] - pi(b)`, so its norm is exactly
`sqrt((1-pi_a)^2 + sum_{b!=a} pi_b^2)` -- a function of the probability vector alone,
needing no backward pass. A decision taken with probability 1.000 contributes EXACTLY
zero to the policy gradient.

The participation ratio `(sum x)^2 / sum x^2` is the Kish effective sample size: of
the ~9 decisions in an episode, how many carry signal.

IMPORTANT -- THIS IS THE SCORE-ONLY VARIANT, NOT the audit's headline 10.8%.
`p3_04` weights each decision by `|A| x score`, using advantages from a real rollout
with GAE. This script has no rollout and no advantages, so it weights by `score`
alone. The two are different statistics and the numbers are NOT comparable: measured
here, the epoch-3056 checkpoint scores 23.4% on the score-only version against the
audit's 10.8% on the advantage-weighted one -- advantage weighting concentrates
further. What this script IS valid for is a like-for-like COMPARISON between two
checkpoints on the same statistic, which is all it is used for.

WHY IT IS MEASURED HERE. If the distillation operator also RAISES participation, the
policy-gradient channel it was blocked on partially re-opens, and a distil-then-PPO
sequence becomes a live option. If participation is unchanged, distillation improves
strength without restoring the gradient channel, and PPO afterwards would be expected
to behave exactly as it did before. That is a fact about the mechanism, not about the
strength, and it costs one forward pass per decision.

USAGE:  python participation.py <ckpt> [<ckpt2> ...] [--games 512]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


from belot.env import BelotEnv                        # noqa: E402
from belot.model import RecurrentMAPPOModel           # noqa: E402
from belot.observation import build_observation        # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 512


def measure(path, games=512, seed0=960_000):
    net = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(DEV)
    net.load_state_dict(torch.load(path, map_location=DEV)["model_state_dict"])
    net.eval()
    st = np.random.get_state()
    rows = []
    for h in range(games):
        env = BelotEnv()
        env.dealer = h % 4
        np.random.seed(seed0 + h)
        env.reset()
        env.bolts_by_team = [0, 0]
        hc = {s: (torch.zeros(1, 1, HIDDEN, device=DEV),
                  torch.zeros(1, 1, HIDDEN, device=DEV)) for s in range(4)}
        while not env.done:
            s = env.current_player
            lo, gl, mk = build_observation(env, s, [0, 0])
            with torch.no_grad():
                dist, _, hc[s] = net(torch.from_numpy(lo).unsqueeze(0).to(DEV),
                                     torch.from_numpy(gl).unsqueeze(0).to(DEV), hc[s],
                                     torch.from_numpy(mk.astype(np.float32))
                                     .unsqueeze(0).to(DEV), is_sequence=False)
            p = dist.probs.squeeze(0).cpu().numpy()
            a = int(np.random.choice(len(p), p=p / p.sum()))   # SAMPLED: training
            k = int(mk.sum())
            score = float(np.sqrt((1 - p[a]) ** 2 + (p ** 2).sum() - p[a] ** 2))
            rows.append((score, p[a], k, 1 if env.phase == "BIDDING" else 0))
            env.step(a)
    np.random.set_state(st)
    sc = np.array([r[0] for r in rows])
    pa = np.array([r[1] for r in rows])
    kk = np.array([r[2] for r in rows])
    bid = np.array([r[3] for r in rows], dtype=bool)
    part = (sc.sum() ** 2) / max((sc ** 2).sum(), 1e-30)
    return dict(n=len(rows), games=games, part=part, part_frac=part / len(rows),
                per_ep=part / float(games * 4),
                med_pi=float(np.median(pa)),
                med_pi_bid=float(np.median(pa[bid])) if bid.any() else float("nan"),
                p25_bid=float(np.percentile(pa[bid], 25)) if bid.any() else float("nan"),
                forced=float((kk == 1).mean()),
                dead=float((sc < 1e-3).mean()),
                mean_score=float(sc.mean()), med_score=float(np.median(sc)))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    g = 512
    for a in sys.argv[1:]:
        if a.startswith("--games"):
            g = int(a.split("=")[1])
    print(f"{'checkpoint':<28s} {'decis':>7s} {'part(score)':>11s} {'/episode':>9s} "
          f"{'medpi':>7s} {'medpi_bid':>10s} {'p25_bid':>8s} {'forced':>7s} "
          f"{'dead':>7s} {'medscore':>9s}")
    for p in args:
        r = measure(p, games=g)
        print(f"{os.path.basename(p):<28s} {r['n']:>7d} {r['part_frac']*100:>8.1f}% "
              f"{r['per_ep']:>9.2f} {r['med_pi']:>7.4f} {r['med_pi_bid']:>10.4f} "
              f"{r['p25_bid']:>8.4f} {r['forced']*100:>6.1f}% {r['dead']*100:>6.1f}% "
              f"{r['med_score']:>9.5f}")
