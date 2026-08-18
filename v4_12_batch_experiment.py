"""
AUDIT v4 / EXP-12 -- does the primary fix (C1) actually move avg_hand_diff?

This is the experiment the whole audit points at, and it turned out to be
affordable: v4_09 measured a training epoch at 9.4s, so the 40-minute wait there
was the EVALUATION, not the training. It is therefore not necessary to hand this
back as a guess.

HYPOTHESIS. The plateau is a signal-to-noise failure. EXP-1/4 measured the
128-episode optimizer step to be pure noise (disjoint-chunk cosine
-0.0005 +- 0.0099) with a gradient noise scale of 1e4-2e5 episodes. Raising the
episodes per optimizer step ~55x should let the accumulated signal exceed the
noise and break the plateau.

WHAT WOULD FALSIFY IT. avg_hand_diff vs the greedy heuristic still inside the CI
of the +2.365 +- 0.400 baseline after ~800 baseline-epochs' worth of data.

CONTROL. The baseline arm is already measured, twice: 400+ epochs of the shipped
config moved nothing (TensorBoard slope +0.021 [-0.082, +0.117] per 100 epochs),
and v4_09's own 30-epoch resume gave +0.123 +- 0.206 pts/hand on paired deals.

SAFETY. Writes only to checkpoints/v4_exp/ and runs/belot_ppo_v4_exp. It never
touches checkpoints/latest_model.pt or checkpoints/best_model.pt.

    python v4_12_batch_experiment.py [epochs] [target_games]
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
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TARGET_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
# LINEAR SCALING RULE -- this is NOT optional tuning, it is required for the
# experiment to be interpretable. C1 replaces 56 noisy optimizer steps per epoch
# with 4 clean ones. At unchanged LR the first pilot measured approx_kl = 0.0000
# per step, i.e. the policy stops moving, so a flat result would be flat for a
# TRIVIAL reason (no parameter motion) rather than because the SNR hypothesis is
# wrong. Optimal LR scales as B/(B+B_simple): going 128 -> ~7000 episodes/step
# with B_simple ~ 2e4 justifies ~40x. 8x is the conservative choice; PPO clipping
# and the TARGET_KL=0.02 early stop are the guard rails, and the printed KL tells
# you whether the trust region is actually being used (want ~0.005-0.02).
LR_MULT = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
EVAL_EVERY = 25
EVAL_MATCHES = 250
START_EPOCH = 2400
OUT = "checkpoints/v4_exp"


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    torch.set_float32_matmul_precision("high")

    ck = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T4.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    start = copy.deepcopy(model)
    opt = optim.Adam(model.parameters(), lr=T4.LR_START)
    opt.load_state_dict(ck["optimizer_state_dict"])

    # FROZEN_INIT == best_model.pt == this checkpoint, so the pool member is a copy
    # of the model itself -- faithful to what the real run does on resume.
    pool = [T4.snapshot(model, DEVICE)]
    vec = VectorizedBelot(T4.NUM_ENVS)
    writer = SummaryWriter("runs/belot_ppo_v4_exp")

    eq = TARGET_GAMES / 512.0
    print(f"C1 arm: {EPOCHS} epochs x {TARGET_GAMES} games = "
          f"{EPOCHS * TARGET_GAMES:,} games "
          f"(~{EPOCHS * eq:.0f} baseline-epochs of data)", flush=True)
    print(f"episodes per optimizer step: whole rollout (~{TARGET_GAMES * 3.4:.0f}), "
          f"was 128", flush=True)
    print(f"LR multiplier {LR_MULT}x (linear-scaling rule; watch KL ~0.005-0.02)\n",
          flush=True)

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
        writer.add_scalar("Diag/EpisodesPerOptimStep", m["episodes_per_step"], ep)
        writer.add_scalar("Diag/GradSteps", m["grad_steps"], ep)
        if i % 5 == 0 or i == EPOCHS - 1:
            print(f"  ep {ep} ({i+1}/{EPOCHS}) eps/step {m['episodes_per_step']} "
                  f"steps {m['grad_steps']} KL {m['approx_kl']:.4f} "
                  f"EV {m['explained_variance']:.3f} clip {m['clip_frac']:.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)

        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            np.random.seed(12345); torch.manual_seed(12345)
            r = evaluate_matches(model, num_matches=EVAL_MATCHES, device=DEVICE,
                                 opponent="heuristic")
            writer.add_scalar("Eval/HandDiffVsHeuristic", r["avg_hand_diff"], ep)
            print(f"  EVAL ep {ep}: hand_diff {r['avg_hand_diff']:+.3f} "
                  f"+- {r['hand_diff_ci95']:.3f}  match% {r['match_win_rate']:.3f} "
                  f"(baseline +2.365 +- 0.400)", flush=True)
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "selection_metric": "v4_exp"},
                       os.path.join(OUT, "c1_latest.pt"))
    print(f"\ntraining took {(time.time()-t_all)/60:.1f} min", flush=True)

    print("\nfinal paired comparison, 4000 identical deals", flush=True)
    d0 = hand_diffs(start, 4000, DEVICE)
    d1 = hand_diffs(model, 4000, DEVICE)
    print("\n" + "=" * 78)
    print("  " + summarise("start (epoch 2400)", d0))
    print("  " + summarise(f"C1 (+{EPOCHS} ep @ {TARGET_GAMES})", d1))
    print("  " + paired("C1 - start (same deals)", d1, d0))

    print("\n" + "=" * 78)
    print("MATCHES TO 101 vs greedy heuristic")
    for name, net in (("start", start), ("C1", model)):
        np.random.seed(12345); torch.manual_seed(12345)
        r = evaluate_matches(net, num_matches=400, device=DEVICE, opponent="heuristic")
        print(f"  {name:<8s} hand_diff {r['avg_hand_diff']:+.3f} "
              f"+- {r['hand_diff_ci95']:.3f}   match% {r['match_win_rate']:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
