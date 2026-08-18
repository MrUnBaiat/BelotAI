"""
AUDIT v4 / EXP-2 -- what the training log already knows.

Free evidence: runs/belot_ppo_v2 covers the plateau. EXP-1 showed the gradient is
pure noise at epoch 2400 and that gradient VARIANCE grew ~293x from init. The
mechanism I suspect is policy determinism: as the entropy bonus anneals
0.04 -> 0.005 the policy sharpens, and grad log pi(a) for a rarely-sampled action
scales like 1/pi(a), so variance explodes and SNR collapses.

If that is the mechanism, the vs-heuristic curve should go flat at roughly the
time the entropy curves bottom out -- not earlier, not later. This script just
dumps the recorded scalars and bootstraps slope CIs over windows, so the timing
claim is measured rather than eyeballed.
"""
import sys
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PATH = "runs/belot_ppo_v2"


def load():
    acc = EventAccumulator(PATH, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        ev = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def boot_slope(x, y, n=4000, seed=0):
    """Bootstrap CI on the OLS slope, per 100 epochs."""
    rng = np.random.default_rng(seed)
    if len(x) < 3:
        return float("nan"), (float("nan"), float("nan"))
    sl = []
    idx = np.arange(len(x))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(x[s])) < 2:
            continue
        sl.append(np.polyfit(x[s], y[s], 1)[0] * 100.0)
    base = np.polyfit(x, y, 1)[0] * 100.0
    return base, (float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5)))


def main():
    d = load()
    print("tags:", ", ".join(sorted(d)))

    for tag in ["Eval/HandDiffVsHeuristic", "Eval/HandDiffVsReference",
                "Eval/MatchWinVsHeuristic", "Entropy/Bidding", "Entropy/Playing",
                "Diag/ApproxKL", "Diag/ExplainedVariance", "Diag/ClipFraction",
                "Sched/EntropyCoef", "Diag/PPOItersCompleted", "Diag/GradSteps"]:
        if tag not in d:
            continue
        x, y = d[tag]
        print(f"\n--- {tag}  (n={len(x)}, epochs {x.min()}-{x.max()}) ---")
        # coarse trajectory
        for lo in range(0, int(x.max()) + 1, 400):
            m = (x >= lo) & (x < lo + 400)
            if m.sum():
                print(f"   ep {lo:5d}-{lo+399:5d}: mean {y[m].mean():+8.4f}  "
                      f"first {y[m][0]:+8.4f}  last {y[m][-1]:+8.4f}  n={m.sum()}")

    print("\n" + "=" * 72)
    print("BOOTSTRAP SLOPES (per 100 epochs) for the headline metric")
    print("=" * 72)
    for tag in ["Eval/HandDiffVsHeuristic", "Eval/HandDiffVsReference"]:
        if tag not in d:
            continue
        x, y = d[tag]
        for lo, hi in [(0, 800), (800, 1600), (1600, 2400), (1200, 2400), (0, 2400)]:
            m = (x >= lo) & (x <= hi)
            if m.sum() < 4:
                continue
            b, (l, u) = boot_slope(x[m], y[m])
            verdict = "RISING" if l > 0 else ("FALLING" if u < 0 else "FLAT")
            print(f"{tag:32s} ep {lo:5d}-{hi:5d} n={m.sum():3d}: "
                  f"{b:+7.3f} [{l:+7.3f},{u:+7.3f}] -> {verdict}")


if __name__ == "__main__":
    main()
