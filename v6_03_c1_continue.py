"""
AUDIT v6 / EXP-3 -- continue C1, the only intervention with a measured gain.

C1 (whole-rollout batch, 24x LR) delivered +0.353 +- 0.218 pts/hand over 150
epochs from the epoch-2400 weights -- the first statistically significant training
gain in this project's history. This continues it from where v4_12 stopped
(epoch 2549) for another 300 epochs and reports the paired gain against the SAME
epoch-2400 baseline, so the two segments are directly comparable.

Linear extrapolation of +0.235 per 100 epochs closes the PIMC gap in ~320 epochs.
Gains normally decelerate, so that is a lower bound on effort, not a promise --
the point of this run is to report the ACTUAL curve, including whether it bends.

INSTRUMENT. The 250-match eval (CI +-0.40) cannot resolve a +0.35 effect; C1's own
250-match trajectory reads flat while the 4,000-deal paired instrument resolves
the gain cleanly. Judge this run on the paired numbers at the end, and treat the
per-eval series as a coarse trace only.

SAFETY: writes only to checkpoints/v6_exp/.
"""
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '.')
import train_v4 as T4
from eval import evaluate_matches
from model import RecurrentMAPPOModel
from v4_paired_eval import hand_diffs, paired, summarise
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TARGET_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
LR_MULT = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
START_EPOCH = 2549
EVAL_EVERY = 50
OUT = "checkpoints/v6_exp"
RESUME = "checkpoints/v4_exp/c1_latest.pt"      # end of the first C1 segment
BASELINE = "checkpoints/reference_model.pt"     # the untouched epoch-2400 weights


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    torch.set_float32_matmul_precision("high")

    ck = torch.load(RESUME, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T4.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    seg1 = copy.deepcopy(model)                   # C1 after 150 epochs
    opt = optim.Adam(model.parameters(), lr=T4.LR_START)
    opt.load_state_dict(ck["optimizer_state_dict"])

    base = RecurrentMAPPOModel(hidden_dim=T4.HIDDEN).to(DEVICE)
    base.load_state_dict(torch.load(BASELINE, map_location=DEVICE,
                                    weights_only=False)["model_state_dict"])

    pool = [T4.snapshot(model, DEVICE)]
    vec = VectorizedBelot(T4.NUM_ENVS)
    writer = SummaryWriter("runs/belot_ppo_v6_c1")
    print(f"C1 continuation: epochs {START_EPOCH}..{START_EPOCH + EPOCHS - 1} "
          f"x {TARGET_GAMES} games, LR x{LR_MULT}\n", flush=True)

    t_all = time.time()
    for i in range(EPOCHS):
        ep = START_EPOCH + i
        lr = T4.anneal(T4.LR_START, T4.LR_END, ep, T4.LR_ANNEAL_EPOCHS) * LR_MULT
        for g in opt.param_groups:
            g["lr"] = lr
        ent = T4.anneal(T4.ENTROPY_START, T4.ENTROPY_END, ep, T4.ENTROPY_ANNEAL_EPOCHS)
        t0 = time.time()
        eps, _ = T4.collect_rollout(model, vec, TARGET_GAMES, DEVICE, pool)
        m = T4.update(model, opt, eps, DEVICE, entropy_coef=ent)
        if not m:
            continue
        writer.add_scalar("Diag/ApproxKL", m["approx_kl"], ep)
        writer.add_scalar("Diag/ExplainedVariance", m["explained_variance"], ep)
        if i % 10 == 0 or i == EPOCHS - 1:
            print(f"  ep {ep} ({i+1}/{EPOCHS}) eps/step {m['episodes_per_step']} "
                  f"KL {m['approx_kl']:.4f} EV {m['explained_variance']:.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            np.random.seed(12345); torch.manual_seed(12345)
            r = evaluate_matches(model, num_matches=250, device=DEVICE,
                                 opponent="heuristic")
            writer.add_scalar("Eval/HandDiffVsHeuristic", r["avg_hand_diff"], ep)
            print(f"  EVAL ep {ep}: hand_diff {r['avg_hand_diff']:+.3f} "
                  f"+- {r['hand_diff_ci95']:.3f} (coarse; see paired result)",
                  flush=True)
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "selection_metric": "v6_c1"},
                       os.path.join(OUT, "c1_continued.pt"))
    print(f"\ntraining took {(time.time()-t_all)/60:.1f} min compute", flush=True)

    N = 4000
    print(f"\npaired comparison on {N} identical deals", flush=True)
    d0 = hand_diffs(base, N, DEVICE)
    d1 = hand_diffs(seg1, N, DEVICE)
    d2 = hand_diffs(model, N, DEVICE)
    print("\n" + "=" * 78)
    print("  " + summarise("epoch 2400 (baseline)", d0))
    print("  " + summarise("C1 +150 epochs", d1))
    print("  " + summarise(f"C1 +{150 + EPOCHS} epochs", d2))
    print("\n" + "=" * 78)
    print("  " + paired("C1 seg1 (+150ep) - baseline", d1, d0))
    print("  " + paired(f"C1 seg1+2 (+{150+EPOCHS}ep) - baseline", d2, d0))
    print("  " + paired("seg2 marginal gain (seg1+2 - seg1)", d2, d1))
    g1, g2 = (d1 - d0).mean(), (d2 - d0).mean()
    print(f"\n  rate seg1  : {g1 / 150 * 100:+.3f} per 100 epochs")
    print(f"  rate seg2  : {(g2 - g1) / EPOCHS * 100:+.3f} per 100 epochs")
    print(f"  -> {'DECELERATING' if (g2-g1)/EPOCHS < g1/150 else 'sustained or accelerating'}")
    writer.close()


if __name__ == "__main__":
    main()
