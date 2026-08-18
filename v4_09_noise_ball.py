"""
AUDIT v4 / EXP-9 -- the noise-ball test. Is the plateau a stochastic equilibrium?

WHY THIS AND NOT "the batch is too small".
EXP-1/4 proved each 128-episode optimizer step is pure noise, but that alone does
NOT explain a 400-epoch flat line: over K steps a random walk accumulates like
sqrt(K) while the drift accumulates like K, so 21,600 steps should have surfaced
even a 0.6%-signal gradient. The mechanism that DOES fit the evidence is a noise
ball. Near a local optimum, performance is locally concave, so random parameter
motion is not performance-neutral -- it costs. The policy equilibrates at the
radius where drift inward balances noise outward, and sits there forever. That is
exactly "KL moves every update, strength never does".

THE PREDICTION THAT SEPARATES IT FROM EVERYTHING ELSE.
If the iterates are orbiting a better centre, the TIME-AVERAGE of the weights is
closer to that centre than any single iterate. So an EMA/Polyak average of the
weights must be measurably STRONGER than the final iterate -- and stronger than
the starting checkpoint -- even though it costs no extra data and changes no
hyper-parameter.

WHAT WOULD FALSIFY IT. EMA within noise of the iterates. That would mean the
iterates are not orbiting anything: either the walk is not centred on a better
point, or the plateau has a different cause (representation limit, or a genuine
optimum of the self-play objective).

DESIGN. Resume the REAL training loop from the epoch-2400 checkpoint, unchanged.
Track an EMA of the weights alongside. Evaluate {start, final iterate, EMA} on
identical deals with the paired instrument, plus evaluate_matches() for an
absolute number comparable to the +2.37 baseline.
"""
import copy
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, '.')
import train as T
from eval import evaluate_matches
from model import RecurrentMAPPOModel
from v4_paired_eval import hand_diffs, paired, summarise
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_DEALS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
EMA_DECAY = 0.98            # ~50-epoch window; the run is 30 epochs so it averages most of it
START_EPOCH = 2400


def main():
    torch.manual_seed(0); np.random.seed(0)
    ck = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)

    model = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    start = copy.deepcopy(model)

    opt = optim.Adam(model.parameters(), lr=T.LR_START)
    opt.load_state_dict(ck["optimizer_state_dict"])

    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)

    pool = [T.snapshot(model, DEVICE)]          # FROZEN_INIT == best_model.pt == this
    vec = VectorizedBelot(T.NUM_ENVS)

    print(f"resuming real training config from epoch {START_EPOCH} for {EPOCHS} epochs "
          f"on {DEVICE}")
    t_all = time.time()
    for i in range(EPOCHS):
        ep = START_EPOCH + i
        lr = T.anneal(T.LR_START, T.LR_END, ep, T.LR_ANNEAL_EPOCHS)
        for g in opt.param_groups:
            g["lr"] = lr
        ent = T.anneal(T.ENTROPY_START, T.ENTROPY_END, ep, T.ENTROPY_ANNEAL_EPOCHS)

        t0 = time.time()
        eps, _ = T.collect_rollout(model, vec, T.TARGET_GAMES, DEVICE, pool)
        m = T.update(model, opt, eps, DEVICE, entropy_coef=ent)

        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(EMA_DECAY).add_(pm, alpha=1 - EMA_DECAY)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        print(f"  epoch {ep} ({i+1}/{EPOCHS}) KL {m['approx_kl']:.4f} "
              f"EV {m['explained_variance']:.3f} clip {m['clip_frac']:.3f} "
              f"ent_play {m['entropy_playing']:.3f} {time.time()-t0:.0f}s", flush=True)
    print(f"training took {(time.time()-t_all)/60:.1f} min")

    # ---------------- paired isolated-hand comparison ----------------
    print(f"\nevaluating on {N_DEALS} identical deals (greedy, vs greedy heuristic)")
    d_start = hand_diffs(start, N_DEALS, DEVICE)
    d_final = hand_diffs(model, N_DEALS, DEVICE)
    d_ema = hand_diffs(ema, N_DEALS, DEVICE)

    print("\n" + "=" * 78)
    print("ABSOLUTE (paired isolated hands -- not directly comparable to matches)")
    print("=" * 78)
    for n, d in (("start (epoch 2400)", d_start), (f"final iterate (+{EPOCHS})", d_final),
                 (f"EMA of weights (+{EPOCHS})", d_ema)):
        print("  " + summarise(n, d))

    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCES on identical deals")
    print("=" * 78)
    print("  " + paired("final iterate - start", d_final, d_start))
    print("  " + paired("EMA - start", d_ema, d_start))
    print("  " + paired("EMA - final iterate", d_ema, d_final))

    # ---------------- absolute match-based number ----------------
    print("\n" + "=" * 78)
    print("MATCHES TO 101 vs the greedy heuristic (comparable to the +2.37 baseline)")
    print("=" * 78)
    for name, net in (("start", start), ("final iterate", model), ("EMA", ema)):
        np.random.seed(12345); torch.manual_seed(12345)
        r = evaluate_matches(net, num_matches=250, device=DEVICE, opponent="heuristic")
        print(f"  {name:<14s} hand_diff {r['avg_hand_diff']:+.3f} "
              f"+- {r['hand_diff_ci95']:.3f}   match% {r['match_win_rate']:.3f}")

    ci = 1.96 * (d_ema - d_final).std(ddof=1) / np.sqrt(N_DEALS)
    print("\n" + "=" * 78)
    if (d_ema - d_final).mean() > ci:
        print("NOISE BALL CONFIRMED: averaging the weights over the walk beats the")
        print("walk itself. The iterates orbit a better centre, so the plateau is a")
        print("stochastic equilibrium set by LR x gradient noise -- not a capability")
        print("limit. Weight averaging and/or a smaller LR-to-noise ratio recover it.")
    else:
        print("NOISE BALL NOT SUPPORTED: EMA is within noise of the final iterate.")
        print("The iterates are not orbiting a better centre; look elsewhere.")
    torch.save({"model_state_dict": ema.state_dict(), "epoch": START_EPOCH + EPOCHS,
                "selection_metric": "v4_ema_probe"}, "checkpoints/v4_ema_probe.pt")


if __name__ == "__main__":
    main()
