"""
AUDIT v8 / EXP-D evaluation -- re-run standalone after a session teardown killed it.

The 60-epoch EXP-D arm COMPLETED (achieved step efficiency mean 25.0%, min 24.8%,
max 25.2%; checkpoint saved at epoch 3056). Only the evaluation was interrupted, so
this re-runs that part alone rather than repeating 2.3h of training.

Everything here uses the swap-paired instrument (v8_swap_eval.py), whose control for
identical policies is exactly 0.000 per deal and which measured a 4.17x variance
reduction against PIMC -- CI ~+-0.14 at 3,500 deals versus +-0.287 unpaired.

Decision rule, fixed before the run:
  >= +0.3 significant vs PIMC   -> CONFIRMED
  null with efficiency >= 20%   -> REFUTED at adequate power
  efficiency < 20%              -> INCONCLUSIVE (achieved 25.0%, so this cannot fire)
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from eval import _heuristic_action
from model import RecurrentMAPPOModel
from pimc import make_pimc
from v8_swap_eval import net_policy, paired, scripted_policy, summarise, swap_edges

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3500
D = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BASELINE = "checkpoints/v7_exp/expc_latest.pt"     # epoch 2997, the arm's start
EXPD = "checkpoints/v8_exp/expd_latest.pt"         # epoch 3056, 60 epochs later
MEAN_EFF = 0.250                                   # measured during the run


def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    net = RecurrentMAPPOModel(hidden_dim=512).to(DEVICE)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net, ck


def main():
    start, bck = load(BASELINE)
    model, mck = load(EXPD)
    print(f"baseline {BASELINE} (epoch {bck.get('epoch')})")
    print(f"EXP-D    {EXPD} (epoch {mck.get('epoch')})")
    print(f"achieved step efficiency during the run: {MEAN_EFF:.1%}")
    print(f"{N} swap-paired deals, PIMC D={D}\n", flush=True)

    pimc = scripted_policy(make_pimc(D=D, seed=0))
    t0 = time.time()
    e_start = swap_edges(net_policy(start, DEVICE), pimc, N)
    print(f"  start vs PIMC done ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    e_expd = swap_edges(net_policy(model, DEVICE), pimc, N)
    print(f"  EXP-D vs PIMC done ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 80)
    print("VS PIMC -- the held-out yardstick, swap-paired")
    print("=" * 80)
    print("  " + summarise(f"start (epoch {bck.get('epoch')})", e_start))
    print("  " + summarise(f"EXP-D (epoch {mck.get('epoch')})", e_expd))
    print("  " + paired("EXP-D - start  [THE TEST]", e_expd, e_start))

    heur = scripted_policy(_heuristic_action)
    h_start = swap_edges(net_policy(start, DEVICE), heur, N)
    h_expd = swap_edges(net_policy(model, DEVICE), heur, N)
    print("\n  vs exact heuristic (TRAINING-ADJACENT -- report, never select on)")
    print("  " + paired("EXP-D - start", h_expd, h_start))

    d = e_expd - e_start
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    gp, gh = d.mean(), (h_expd - h_start).mean()

    print("\n" + "=" * 80)
    print("CONTAMINATION DIAGNOSTIC")
    print("=" * 80)
    print(f"  gain vs PIMC {gp:+.3f}   gain vs exact heuristic {gh:+.3f}")
    if gh > 0.2 and gp < 0.5 * gh:
        print("  -> heuristic-side gain is largely OPPONENT-SPECIFIC exploitation;")
        print("     believe the PIMC number. The perturbed opponent only partially")
        print("     decontaminates (84.5% bidding agreement with the exact heuristic).")
    else:
        print("  -> no strong evidence of exploitation-only gain. Still select on PIMC.")

    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)
    print(f"  EXP-D - start vs PIMC = {gp:+.3f} +- {ci:.3f}")
    print(f"  achieved step efficiency = {MEAN_EFF:.1%}")
    if gp >= 0.3 and gp > ci:
        print("  -> CONFIRMED. The plateau was a training-distribution problem.")
    elif MEAN_EFF >= 0.20:
        print(f"  -> REFUTED at adequate power. A regime-filtered actor loss on a")
        print(f"     non-descendant opponent does not break the plateau; the effect")
        print(f"     is bounded to +-{ci:.2f}, versus EXP-C's +-0.287 which bounded nothing.")
    else:
        print(f"  -> INCONCLUSIVE: efficiency {MEAN_EFF:.1%} < 20%.")


if __name__ == "__main__":
    main()
