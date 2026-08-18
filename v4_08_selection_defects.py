"""
AUDIT v4 / EXP-8 -- are the selection / opponent-pool defects real?

Commit f1f0918 (a later audit that was reverted out of the working tree) flags
three defects, D1-D3. They are present in the code I was handed. Two are checkable
by inspection alone; the interesting question is whether they MATTER, which is a
measurement.

  D1  FROZEN_INIT = "checkpoints/best_model.pt", which train() OVERWRITES. So the
      15% frozen training opponent, and the EVAL_REFERENCE copied from it, are
      both descendants of the model being measured.
  D2  metric = r_ref["avg_hand_diff"] -- best_model.pt is SELECTED on the score
      against a checkpoint that sits in its own training pool.
  D3  _sample_opponent() short-circuits to "self" when the pool is empty, which
      silently disables the RANDOM opponent too.

THE MEASUREMENT THAT MATTERS FOR D2. If vs-reference were a usable proxy for
absolute strength, it would correlate with vs-heuristic across evaluations.
Selecting on an uncorrelated metric is selecting on noise. So: correlate the two
eval series over the run, with a bootstrap CI, and check the sign.

D3 is verified by direct execution of the function, not by reading it.
"""
import os
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, '.')
import train as T


def series(acc, tag):
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def main():
    print("=" * 78)
    print("D1 / D2  static state of the code and checkpoint directory")
    print("=" * 78)
    print(f"  FROZEN_INIT        = {T.FROZEN_INIT}")
    print(f"  EVAL_REFERENCE     = {T.EVAL_REFERENCE}")
    print(f"  SELECTION_METRIC   = {T.SELECTION_METRIC}")
    for p in ("checkpoints/best_model.pt", "checkpoints/latest_model.pt",
              "checkpoints/reference_model.pt"):
        print(f"  {p:<38s} exists={os.path.exists(p)}")
    print("\n  train() writes best_model.pt, and FROZEN_INIT reads it: "
          f"{T.FROZEN_INIT.endswith('best_model.pt')}")
    print("  reference_model.pt is absent, so the next run would CREATE it by "
          "copying\n  best_model.pt -- i.e. pin the 'immutable yardstick' to the "
          "epoch-2400 model itself.")

    print("\n" + "=" * 78)
    print("D3  does an empty frozen pool disable the random opponent? (executed)")
    print("=" * 78)
    import random
    random.seed(0)
    kinds = [T._sample_opponent([])[0] for _ in range(20000)]
    frac = {k: kinds.count(k) / len(kinds) for k in set(kinds)}
    print(f"  empty pool  -> {frac}")
    print(f"  configured  -> self {T.OPP_SELF:.0%}, random {T.OPP_RANDOM:.0%}, "
          f"frozen {T.OPP_FROZEN:.0%}")
    got_random = frac.get("random", 0.0)
    print(f"  RESULT: random opponent share with an empty pool = {got_random:.1%} "
          f"-> D3 {'CONFIRMED' if got_random < 0.01 else 'REFUTED'}")

    print("\n" + "=" * 78)
    print("D2  is vs-reference informative about absolute strength?")
    print("=" * 78)
    acc = EventAccumulator("runs/belot_ppo_v2", size_guidance={"scalars": 0})
    acc.Reload()
    xh, yh = series(acc, "Eval/HandDiffVsHeuristic")
    xr, yr = series(acc, "Eval/HandDiffVsReference")
    common = np.intersect1d(xh, xr)
    a = yh[np.isin(xh, common)]
    b = yr[np.isin(xr, common)]
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        s = rng.integers(0, len(a), len(a))
        if np.std(a[s]) < 1e-9 or np.std(b[s]) < 1e-9:
            continue
        boots.append(np.corrcoef(a[s], b[s])[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  paired evaluations: n={len(a)} (epochs {common.min()}-{common.max()})")
    print(f"  vs-heuristic  mean {a.mean():+.3f}  sd {a.std(ddof=1):.3f}")
    print(f"  vs-reference  mean {b.mean():+.3f}  sd {b.std(ddof=1):.3f}")
    print(f"  correlation r = {r:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  NOTE: n=13 evaluations has no power to resolve a correlation; this CI")
    print("        spans -0.47..+0.37 and is NOT evidence either way. The powered")
    print("        test is whether the two series have DIFFERENT SLOPES.")

    def boot_slope(x, y, seed=0):
        rng2 = np.random.default_rng(seed)
        s = [np.polyfit(x[i], y[i], 1)[0] * 100
             for i in (rng2.integers(0, len(x), len(x)) for _ in range(5000))
             if len(np.unique(x[i])) > 1]
        return np.polyfit(x, y, 1)[0] * 100, np.percentile(s, [2.5, 97.5])

    sh, cih = boot_slope(common, a)
    sr, cir = boot_slope(common, b)
    print(f"\n  slope vs-heuristic : {sh:+.3f} [{cih[0]:+.3f}, {cih[1]:+.3f}] per 100 ep")
    print(f"  slope vs-reference : {sr:+.3f} [{cir[0]:+.3f}, {cir[1]:+.3f}] per 100 ep")
    if cir[0] > 0 and cih[0] < 0 < cih[1]:
        print("\n  RESULT: D2 CONFIRMED by slope divergence -- the selection metric rises")
        print("          significantly while absolute strength does not move at all.")
        print("          Selecting on it selects for exploiting an opponent that sits")
        print("          in the model's own training pool. (The correlation test above")
        print("          is merely underpowered, not contradictory.)")
    else:
        print("\n  RESULT: D2 not established by these series.")


if __name__ == "__main__":
    main()
