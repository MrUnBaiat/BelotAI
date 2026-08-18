"""
AUDIT v5 / EXP-4 -- does league + re-heat beat plain C1?

CONTROLLED THREE-WAY COMPARISON. All arms start from the SAME weights (epoch 2400,
preserved in checkpoints/reference_model.pt), get the SAME data budget
(150 epochs x 2048 games), and use the SAME optimizer settings (whole-rollout
batch, LR_BATCH_SCALE-equivalent 24x). The only differences are the two changes
under test:

    start      epoch 2400
    C1         big batch only                    -> already run (v4_12), +0.353 +- 0.218
    C1+L+R     big batch + league + re-heat      <- this script

HYPOTHESES.
  C7 LEAGUE. The old "frozen 15%" opponent was a byte-identical clone of the live
     model (0.00% decision disagreement, measured in v5_03), so the real mix was
     85% self-play + 15% saturated uniform random. Self-play cycling was measured
     directly: vs-reference rose +0.073 [+0.042, +0.112] per 100 epochs while
     absolute strength stayed flat. A pool of 8 genuinely-spaced snapshots at 50%
     of games should convert some of that cycling into absolute strength.
  C6 RE-HEAT. Entropy hit its 0.005 floor at ~epoch 2000 and pi(sampled action)
     has median 1.000 in bidding. The plateau may be partly a plateau OF THE
     SCHEDULE. Entropy coefficient is held at 0.025.

WHAT WOULD FALSIFY THEM. C1+L+R within CI of the C1 arm on identical deals. The
paired instrument resolves ~+-0.22 at 4,000 deals, and C1 itself bought +0.353,
so an effect of similar size is detectable.

CONFOUND, STATED UP FRONT: this arm changes TWO things at once. If it wins, a
follow-up must separate them; if it loses or ties, both are dead together. That
trade was taken deliberately -- one 90-minute run instead of two, given the C1
result showed the effect sizes here are small.

SAFETY. Writes only to checkpoints/v5_exp/ and runs/belot_ppo_v5_exp.
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
import train_v5 as T5
from eval import evaluate_matches
from model import RecurrentMAPPOModel
from v4_paired_eval import hand_diffs, paired, summarise
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
TARGET_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
LR_MULT = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
EVAL_EVERY = 25
EVAL_MATCHES = 250
START_EPOCH = 2400
OUT = "checkpoints/v5_exp"
START_CKPT = "checkpoints/reference_model.pt"     # the untouched epoch-2400 weights
C1_CKPT = "checkpoints/v4_exp/c1_latest.pt"       # the C1-only arm's result


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    torch.set_float32_matmul_precision("high")

    ck = torch.load(START_CKPT, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T5.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    start = copy.deepcopy(model)
    opt = optim.Adam(model.parameters(), lr=T5.LR_START)
    if "optimizer_state_dict" in ck:
        opt.load_state_dict(ck["optimizer_state_dict"])

    # C7: seed the pool from the immutable reference, then let SNAPSHOT_EVERY=33
    # fill it with genuinely spaced members.
    pool = [T5.load_frozen(START_CKPT, DEVICE)]
    vec = VectorizedBelot(T5.NUM_ENVS)
    writer = SummaryWriter("runs/belot_ppo_v5_exp")

    print(f"C1+LEAGUE+REHEAT: {EPOCHS} epochs x {TARGET_GAMES} games "
          f"(~{EPOCHS * TARGET_GAMES / 512:.0f} baseline-epochs of data)", flush=True)
    print(f"  mix self/random/frozen = {T5.OPP_SELF:.0%}/{T5.OPP_RANDOM:.0%}/"
          f"{T5.OPP_FROZEN:.0%}, pool max {T5.FROZEN_POOL_MAX}, "
          f"snapshot every {T5.SNAPSHOT_EVERY}", flush=True)
    print(f"  entropy coef held at {T5.ENTROPY_START}  (was annealed to 0.005)",
          flush=True)
    print(f"  LR x{LR_MULT}\n", flush=True)

    t_all = time.time()
    for i in range(EPOCHS):
        ep = START_EPOCH + i
        lr = T5.anneal(T5.LR_START, T5.LR_END, ep, T5.LR_ANNEAL_EPOCHS) * LR_MULT
        for g in opt.param_groups:
            g["lr"] = lr
        ent = T5.anneal(T5.ENTROPY_START, T5.ENTROPY_END, ep, T5.ENTROPY_ANNEAL_EPOCHS)

        t0 = time.time()
        eps, roll = T5.collect_rollout(model, vec, TARGET_GAMES, DEVICE, pool)
        m = T5.update(model, opt, eps, DEVICE, entropy_coef=ent)
        if not m:
            continue

        if T5.SNAPSHOT_EVERY and i > 0 and i % T5.SNAPSHOT_EVERY == 0:
            pool.append(T5.snapshot(model, DEVICE))
            if len(pool) > T5.FROZEN_POOL_MAX:
                pool.pop(1)                       # keep the immutable reference at 0
            print(f"  ep {ep}: pool size {len(pool)}", flush=True)

        writer.add_scalar("Diag/ApproxKL", m["approx_kl"], ep)
        writer.add_scalar("Diag/ExplainedVariance", m["explained_variance"], ep)
        writer.add_scalar("Entropy/Playing", m["entropy_playing"], ep)
        writer.add_scalar("Entropy/Bidding", m["entropy_bidding"], ep)
        if i % 5 == 0 or i == EPOCHS - 1:
            print(f"  ep {ep} ({i+1}/{EPOCHS}) eps/step {m['episodes_per_step']} "
                  f"KL {m['approx_kl']:.4f} EV {m['explained_variance']:.3f} "
                  f"entB {m['entropy_bidding']:.3f} entP {m['entropy_playing']:.3f} "
                  f"pool {len(pool)} {time.time()-t0:.0f}s", flush=True)

        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            np.random.seed(12345); torch.manual_seed(12345)
            r = evaluate_matches(model, num_matches=EVAL_MATCHES, device=DEVICE,
                                 opponent="heuristic")
            writer.add_scalar("Eval/HandDiffVsHeuristic", r["avg_hand_diff"], ep)
            print(f"  EVAL ep {ep}: hand_diff {r['avg_hand_diff']:+.3f} "
                  f"+- {r['hand_diff_ci95']:.3f} match% {r['match_win_rate']:.3f}",
                  flush=True)
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "selection_metric": "v5_exp"},
                       os.path.join(OUT, "league_reheat_latest.pt"))
    print(f"\ntraining took {(time.time()-t_all)/60:.1f} min", flush=True)

    # ---------------- three-way paired comparison ----------------
    N = 4000
    c1 = RecurrentMAPPOModel(hidden_dim=T5.HIDDEN).to(DEVICE)
    c1.load_state_dict(torch.load(C1_CKPT, map_location=DEVICE,
                                  weights_only=False)["model_state_dict"])
    print(f"\nthree-way paired comparison on {N} identical deals", flush=True)
    d0 = hand_diffs(start, N, DEVICE)
    d1 = hand_diffs(c1, N, DEVICE)
    d2 = hand_diffs(model, N, DEVICE)

    print("\n" + "=" * 78)
    print("ABSOLUTE (paired isolated hands vs the greedy heuristic)")
    print("=" * 78)
    print("  " + summarise("start (epoch 2400)", d0))
    print("  " + summarise("C1 (big batch only)", d1))
    print("  " + summarise("C1 + league + re-heat", d2))
    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCES on identical deals")
    print("=" * 78)
    print("  " + paired("C1 - start", d1, d0))
    print("  " + paired("C1+L+R - start", d2, d0))
    print("  " + paired("C1+L+R - C1  (the test)", d2, d1))

    print("\n" + "=" * 78)
    print("MATCHES TO 101 vs greedy heuristic (400 matches)")
    print("=" * 78)
    for name, net in (("start", start), ("C1", c1), ("C1+L+R", model)):
        np.random.seed(12345); torch.manual_seed(12345)
        r = evaluate_matches(net, num_matches=400, device=DEVICE, opponent="heuristic")
        print(f"  {name:<10s} hand_diff {r['avg_hand_diff']:+.3f} "
              f"+- {r['hand_diff_ci95']:.3f}  match% {r['match_win_rate']:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
