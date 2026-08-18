"""
AUDIT v7 / EXP-C -- does a non-descendant opponent break the plateau?

THE HYPOTHESIS, and it is now measured rather than assumed. v4_13 on the
epoch-2848 weights, 128 chunks per regime, episode-equalised:

    regime      deviation from pure noise   disjoint cosine        B_simple
    self                          3.7%      +0.00029+-0.00107       437,395
    random                       27.6%      +0.00265+-0.00091        48,248
    heuristic                    39.3%      +0.00430+-0.00111        29,660

The policy is stationary under the SELF-PLAY objective specifically, not globally.
A real improvement direction exists against non-descendant opponents; a training
mix that is ~85% self-play cannot see it. v6 section 6.1's "one-off shift then
flat" across five interventions follows from exactly this: each perturbed the
fixed point and the self-play objective pulled it back.

THE ARM. self 40 / frozen 30 / perturbed-heuristic 25 / random 5, episode-
equalised, large batch retained (B_simple is 29,660 even in the good regime, so
~7,000 episodes/step is still ~4x short -- do not revert to 128).

CONTROL. The v6 C1-continuation arm, which is the same optimizer settings with a
~85% self-play mix, measured at +0.334 +- 0.228 over baseline with a MARGINAL rate
of -0.020 +- 0.223. If EXP-C is merely another one-off shift it will land there.

YARDSTICKS, and why the exact heuristic is no longer one of them.
  PIMC (D=16)            SOLE selection yardstick, never in the pool.
  exact greedy heuristic TRAINING-ADJACENT now -- the pool contains a perturbed
                         variant whose bidding still agrees with it 84.5% of the
                         time, so gains against it are partly exploitation.
                         Logged as a cheap dense trace, never selected on.

CONTAMINATION DIAGNOSTIC. pimc.py delegates bidding to _heuristic_action, so the
perturbed opponent only partially decontaminates the yardstick. Reported directly:
if the gain vs PIMC is far smaller than the gain vs the exact heuristic, the
heuristic-side gain is largely opponent-specific exploitation and the PIMC number
is the one to believe.

INSTRUMENT. Effects below ~0.5 pts/hand are invisible to match-based evals
(40 matches = +-0.99, 250 matches = +-0.40). Resolving +0.3 needs ~3,300 PAIRED
deals. Final call is paired isolated hands vs PIMC.

SAFETY: writes only to checkpoints/v7_exp/.
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
import train_v7 as T7
from eval import evaluate_matches
from model import RecurrentMAPPOModel
from v4_paired_eval import hand_diffs, paired, summarise
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
TARGET_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 7000
LR_MULT = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0

# BASELINE is the weights every result is measured AGAINST; RESUME is where training
# picks up. They are deliberately separate: an interrupted run must resume from its
# own last checkpoint while still being scored against the ORIGINAL baseline,
# otherwise the final paired comparison silently measures only the epochs since the
# interruption instead of the whole arm.
BASELINE = "checkpoints/v6_exp/c1_continued.pt"    # epoch 2848
RESUME = sys.argv[4] if len(sys.argv) > 4 else BASELINE
EVAL_EVERY = 25
OUT = "checkpoints/v7_exp"


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    torch.set_float32_matmul_precision("high")

    # training weights + optimizer come from RESUME; the comparison model comes from
    # BASELINE, so a resumed run is still scored over the WHOLE arm.
    ck = torch.load(RESUME, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T7.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    opt = optim.Adam(model.parameters(), lr=T7.LR_START)
    opt.load_state_dict(ck["optimizer_state_dict"])
    START_EPOCH = int(ck.get("epoch", 2848))

    base_ck = torch.load(BASELINE, map_location=DEVICE, weights_only=False)
    start = RecurrentMAPPOModel(hidden_dim=T7.HIDDEN).to(DEVICE)
    start.load_state_dict(base_ck["model_state_dict"])
    if RESUME != BASELINE:
        print(f"RESUMING from {RESUME} (epoch {START_EPOCH}); still scored against "
              f"{BASELINE} (epoch {base_ck.get('epoch')})", flush=True)

    pool = [T7.snapshot(model, DEVICE)]
    vec = VectorizedBelot(T7.NUM_ENVS)
    writer = SummaryWriter("runs/belot_ppo_v7_expc")
    snap_every = max(1, EPOCHS // (T7.FROZEN_POOL_MAX - 1))   # fill the pool

    print(f"EXP-C: {EPOCHS} epochs x {TARGET_EPISODES:,} EPISODES per step "
          f"(episode-equalised), LR x{LR_MULT}")
    print(f"  mix self {T7.OPP_SELF:.0%} / frozen {T7.OPP_FROZEN:.0%} / "
          f"script {T7.OPP_SCRIPT:.0%} / random {T7.OPP_RANDOM:.0%}")
    print(f"  pool max {T7.FROZEN_POOL_MAX}, snapshot every {snap_every} epochs\n",
          flush=True)

    for i in range(EPOCHS):
        ep = START_EPOCH + i
        lr = T7.anneal(T7.LR_START, T7.LR_END, ep, T7.LR_ANNEAL_EPOCHS) * LR_MULT
        for g in opt.param_groups:
            g["lr"] = lr
        ent = T7.anneal(T7.ENTROPY_START, T7.ENTROPY_END, ep, T7.ENTROPY_ANNEAL_EPOCHS)
        t0 = time.time()
        eps, roll = T7.collect_rollout(model, vec, 10 ** 9, DEVICE, pool,
                                       target_episodes=TARGET_EPISODES)
        m = T7.update(model, opt, eps, DEVICE, entropy_coef=ent)
        if not m:
            continue
        if i > 0 and i % snap_every == 0:
            pool.append(T7.snapshot(model, DEVICE))
            if len(pool) > T7.FROZEN_POOL_MAX:
                pool.pop(1)
        writer.add_scalar("Diag/ApproxKL", m["approx_kl"], ep)
        writer.add_scalar("Diag/ExplainedVariance", m["explained_variance"], ep)
        writer.add_scalar("Diag/EpisodesPerOptimStep", m["episodes_per_step"], ep)
        if i % 10 == 0 or i == EPOCHS - 1:
            print(f"  ep {ep} ({i+1}/{EPOCHS}) eps/step {m['episodes_per_step']} "
                  f"KL {m['approx_kl']:.4f} EV {m['explained_variance']:.3f} "
                  f"pool {len(pool)} mix {roll['games_by_opponent']} "
                  f"{time.time()-t0:.0f}s", flush=True)
        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            np.random.seed(12345); torch.manual_seed(12345)
            r = evaluate_matches(model, num_matches=250, device=DEVICE,
                                 opponent="heuristic")
            writer.add_scalar("Eval/HandDiffVsHeuristic_TRAINING_ADJACENT",
                              r["avg_hand_diff"], ep)
            print(f"  EVAL ep {ep}: vs heuristic {r['avg_hand_diff']:+.3f} "
                  f"+- {r['hand_diff_ci95']:.3f}  [TRAINING-ADJACENT, coarse]",
                  flush=True)
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "selection_metric": "v7_expc"},
                       os.path.join(OUT, "expc_latest.pt"))

    # ------------- final: paired vs PIMC (the yardstick) and vs heuristic -------------
    from pimc import make_pimc
    from env import BelotEnv
    from eval import _heuristic_action
    from observation import build_observation
    N_PIMC = 3500
    D = 16

    def duel_vs(actor_fn, opp_fn, n, tag):
        out = np.empty(n); t0 = time.time()
        for d in range(n):
            e = BelotEnv(); e.dealer = d % 4
            np.random.seed(770_000 + d); e.reset(); e.bolts_by_team = [0, 0]
            hc = {s: (torch.zeros(1, 1, T7.HIDDEN, device=DEVICE),
                      torch.zeros(1, 1, T7.HIDDEN, device=DEVICE)) for s in range(4)}
            info = {}
            while not e.done:
                if e.current_player % 2 == 0:
                    a = actor_fn(e, hc)
                else:
                    a = opp_fn(e)
                _, _, _, info = e.step(a)
            gp = info["game_points"]; out[d] = gp[0] - gp[1]
        print(f"    {tag}: {time.time()-t0:.0f}s", flush=True)
        return out

    def net_fn(net):
        @torch.no_grad()
        def f(e, hc):
            s = e.current_player
            l, g, msk = build_observation(e, s, [0, 0])
            dist, _, hc[s] = net(
                torch.from_numpy(l).unsqueeze(0).to(DEVICE),
                torch.from_numpy(g).unsqueeze(0).to(DEVICE), hc[s],
                torch.from_numpy(msk.astype(np.float32)).unsqueeze(0).to(DEVICE),
                is_sequence=False)
            return int(dist.probs.argmax(-1).item())
        return f

    print(f"\nfinal paired evaluation, {N_PIMC} identical deals", flush=True)
    pimc = make_pimc(D=D, seed=0)
    res = {}
    for name, net in (("start", start), ("expc", model)):
        res[(name, "pimc")] = duel_vs(net_fn(net), pimc, N_PIMC, f"{name} vs PIMC")
        res[(name, "heur")] = duel_vs(net_fn(net), _heuristic_action, N_PIMC,
                                      f"{name} vs heuristic")

    print("\n" + "=" * 80)
    print("VS PIMC  (the yardstick -- never in the training pool)")
    print("=" * 80)
    print("  " + summarise("start (epoch 2848)", res[("start", "pimc")]))
    print("  " + summarise("EXP-C", res[("expc", "pimc")]))
    print("  " + paired("EXP-C - start  [THE TEST]",
                        res[("expc", "pimc")], res[("start", "pimc")]))

    print("\n" + "=" * 80)
    print("VS EXACT HEURISTIC  (TRAINING-ADJACENT -- report, never select on)")
    print("=" * 80)
    print("  " + paired("EXP-C - start",
                        res[("expc", "heur")], res[("start", "heur")]))

    gp = (res[("expc", "pimc")] - res[("start", "pimc")]).mean()
    gh = (res[("expc", "heur")] - res[("start", "heur")]).mean()
    print("\n" + "=" * 80)
    print("CONTAMINATION DIAGNOSTIC")
    print("=" * 80)
    print(f"  gain vs PIMC {gp:+.3f}   gain vs exact heuristic {gh:+.3f}")
    if gh > 0.2 and gp < 0.5 * gh:
        print("  -> the heuristic-side gain is largely OPPONENT-SPECIFIC exploitation;")
        print("     believe the PIMC number. The perturbed opponent only partially")
        print("     decontaminates, as expected (84.5% bidding agreement).")
    else:
        print("  -> the two track each other; no strong evidence of exploitation-only")
        print("     gain. Still select on PIMC.")
    writer.close()


if __name__ == "__main__":
    main()
