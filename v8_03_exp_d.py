"""
AUDIT v8 / EXP-D -- regime-filtered actor loss at a step efficiency that can decide.

WHY THIS ARM EXISTS. EXP-A2 measured, on epoch-2848 weights over 3 independent
draws, that `self` has a NEGATIVE unbiased |mu|^2 -- its gradient is
indistinguishable from zero and has no definable direction -- while `perturbed`
carries real signal (B_simple 34.6k / 53.7k / 62.2k). Training the actor on
self-play episodes therefore adds variance and no mean, and the dilution penalty is
QUADRATIC (B_eff = B_simple / f^2, verified analytically and by simulation).

EXP-C failed at f = 17.9% of episodes -> 0.8% step efficiency, which is why its null
bounded nothing. This arm trains the actor on `perturbed` ONLY (f = 1) and sizes the
batch for ~25% step efficiency.

DEVIATION FROM THE PRE-REGISTERED 3x INCLUSION RULE, signed off by the user: that
rule selects ['random','perturbed'], but mixing two only-partially-aligned regimes
(cos ~ 0.1-0.3) shrinks the mean while variance is unchanged, so B_eff RISES
62,164 -> 88,929. The rule assumed inclusion adds signal; measurement shows it
dilutes the direction.

SIZING (conservative -- worst of three draws, B_eff = 62,164):
    25% efficiency  ->  B_actor = 62,164 * 0.25/0.75 = 20,721 actor episodes/step
    mix script 90 / self 5 / random 5  ->  180 actor of 210 episodes per 100 games
    ->  ~11,500 games per rollout

INSTRUMENT. Final call on 3,500 SWAP-PAIRED deals vs PIMC. The swap-paired control
(identical policies) returns exactly 0.000 +- 0.000 per deal, and measured 4.17x
variance reduction vs PIMC gives CI ~+-0.141 and 98.7% power at the +0.3 threshold,
against 53.5% for the unpaired instrument EXP-C used.

DECISION RULE (fixed in advance):
    >= +0.3 significant vs PIMC   -> CONFIRMED
    null WITH efficiency >= 20%   -> a real refutation
    efficiency < 20%              -> INCONCLUSIVE, not a refutation

SAFETY: writes only to checkpoints/v8_exp/.
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
import train_v8 as T8
from eval import _heuristic_action
from model import RecurrentMAPPOModel
from v8_swap_eval import net_policy, paired, scripted_policy, summarise, swap_edges
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TARGET_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 24_200   # ~20.7k actor
LR_MULT = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
BASELINE = "checkpoints/v7_exp/expc_latest.pt"     # epoch 2997, the decision-rule start
RESUME = sys.argv[4] if len(sys.argv) > 4 else BASELINE
OUT = "checkpoints/v8_exp"
B_EFF = 62_164          # conservative (worst) draw for `perturbed`


def efficiency(b_actor):
    return b_actor / (b_actor + B_EFF)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    torch.set_float32_matmul_precision("high")

    ck = torch.load(RESUME, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T8.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    opt = optim.Adam(model.parameters(), lr=T8.LR_START)
    opt.load_state_dict(ck["optimizer_state_dict"])
    start_epoch = int(ck.get("epoch", 2997))

    base_ck = torch.load(BASELINE, map_location=DEVICE, weights_only=False)
    start = RecurrentMAPPOModel(hidden_dim=T8.HIDDEN).to(DEVICE)
    start.load_state_dict(base_ck["model_state_dict"])

    vec = VectorizedBelot(T8.NUM_ENVS)
    writer = SummaryWriter("runs/belot_ppo_v8_expd")
    print(f"EXP-D: actor regimes {T8.ACTOR_REGIMES}, mix script {T8.OPP_SCRIPT:.0%} / "
          f"self {T8.OPP_SELF:.0%} / random {T8.OPP_RANDOM:.0%}")
    print(f"  target {TARGET_EPISODES:,} episodes/step, B_eff {B_EFF:,} (conservative)")
    print(f"  baseline {BASELINE} (epoch {base_ck.get('epoch')}), "
          f"resume epoch {start_epoch}, {EPOCHS} epochs, LR x{LR_MULT}\n", flush=True)

    eff_hist = []
    for i in range(EPOCHS):
        ep = start_epoch + i
        lr = T8.anneal(T8.LR_START, T8.LR_END, ep, T8.LR_ANNEAL_EPOCHS) * LR_MULT
        for g in opt.param_groups:
            g["lr"] = lr
        ent = T8.anneal(T8.ENTROPY_START, T8.ENTROPY_END, ep, T8.ENTROPY_ANNEAL_EPOCHS)
        t0 = time.time()
        eps, roll = T8.collect_rollout(model, vec, 10 ** 9, DEVICE, None,
                                       target_episodes=TARGET_EPISODES)
        n_actor = sum(len(e) for e in eps
                      if getattr(e, "regime", "script") in T8.ACTOR_REGIMES)
        n_actor_eps = sum(1 for e in eps
                          if getattr(e, "regime", "script") in T8.ACTOR_REGIMES)
        m = T8.update(model, opt, eps, DEVICE, entropy_coef=ent)
        if not m:
            print(f"  ep {ep}: update returned empty (no actor episodes)", flush=True)
            continue
        eff = efficiency(n_actor_eps)
        eff_hist.append(eff)
        writer.add_scalar("Diag/StepEfficiency", eff, ep)
        writer.add_scalar("Diag/ActorEpisodes", n_actor_eps, ep)
        writer.add_scalar("Diag/ApproxKL", m["approx_kl"], ep)
        writer.add_scalar("Diag/ExplainedVariance", m["explained_variance"], ep)
        if i % 5 == 0 or i == EPOCHS - 1:
            print(f"  ep {ep} ({i+1}/{EPOCHS}) actor_eps {n_actor_eps:,} "
                  f"(steps {n_actor:,}) of {len(eps):,} | STEP EFFICIENCY {eff:.1%} | "
                  f"KL {m['approx_kl']:.4f} EV {m['explained_variance']:.3f} "
                  f"mix {roll['games_by_opponent']} {time.time()-t0:.0f}s", flush=True)
        if i % 20 == 0 or i == EPOCHS - 1:
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "selection_metric": "v8_expd"},
                       os.path.join(OUT, "expd_latest.pt"))

    mean_eff = float(np.mean(eff_hist)) if eff_hist else 0.0
    print(f"\nACHIEVED STEP EFFICIENCY: mean {mean_eff:.1%} "
          f"(min {min(eff_hist):.1%}, max {max(eff_hist):.1%})", flush=True)

    # ---------------- swap-paired evaluation vs PIMC ----------------
    from pimc import make_pimc
    N = 3500
    D = 16
    pimc = scripted_policy(make_pimc(D=D, seed=0))
    print(f"\nswap-paired evaluation, {N} deals vs PIMC (D={D})", flush=True)
    t0 = time.time()
    e_start = swap_edges(net_policy(start, DEVICE), pimc, N)
    print(f"  start done ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    e_expd = swap_edges(net_policy(model, DEVICE), pimc, N)
    print(f"  EXP-D done ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 80)
    print("VS PIMC -- swap-paired (control for identical policies is exactly 0)")
    print("=" * 80)
    print("  " + summarise(f"start (epoch {base_ck.get('epoch')})", e_start))
    print("  " + summarise("EXP-D", e_expd))
    print("  " + paired("EXP-D - start  [THE TEST]", e_expd, e_start))

    heur = scripted_policy(_heuristic_action)
    h_start = swap_edges(net_policy(start, DEVICE), heur, N)
    h_expd = swap_edges(net_policy(model, DEVICE), heur, N)
    print("\n  vs exact heuristic (TRAINING-ADJACENT, never selected on)")
    print("  " + paired("EXP-D - start", h_expd, h_start))

    d = e_expd - e_start
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)
    print(f"  EXP-D - start vs PIMC = {d.mean():+.3f} +- {ci:.3f}")
    print(f"  achieved step efficiency = {mean_eff:.1%}")
    if d.mean() >= 0.3 and d.mean() > ci:
        print("  -> CONFIRMED. The plateau was a training-distribution problem.")
    elif mean_eff >= 0.20:
        print("  -> REFUTED at adequate power. A regime-filtered actor loss on a")
        print("     non-descendant opponent does not break the plateau, and the")
        print("     effect is bounded to +-{:.2f}.".format(ci))
    else:
        print(f"  -> INCONCLUSIVE: efficiency {mean_eff:.1%} < 20%. Per the standing")
        print("     guard this is NOT a refutation and must not be reported as one.")
    writer.close()


if __name__ == "__main__":
    main()
