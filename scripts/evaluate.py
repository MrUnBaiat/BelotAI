"""
Measure the composite player's strength on the swap-paired instrument.

    python scripts/evaluate.py --ckpt checkpoints/v8_exp/expd_latest.pt --n 1500

WHAT THIS MEASURES. `model bidding + model card play at tricks 0-2 + exact-solve
PIMC (D=128) from trick 3` against the bare network, on identical deals. Reference
figures, all with the control at exactly zero:

    +1.270 +- 0.197  vs the bare agent          (n=1000, D=128)
    +0.835 +- 0.219  vs the bare agent          (n=1000, D=8)
    +0.936 +- 0.243  vs a held-out frozen net   (n=500,  D=8, --opponent FROZEN.pt)

WHY SWAP-PAIRED. Belot's per-hand outcome swings by tens of game points on card
luck alone, so an unpaired comparison of two decent policies is mostly noise. Each
deal is therefore played TWICE with the seat pairs exchanged and the edge taken as
`(dA - dB)/2`. Identical policies then cancel deal by deal rather than on average,
which is why the control below must print exactly `+0.000` with `max|edge|` exactly
zero -- a merely small control means something is leaking and no number below it
counts. Measured variance reduction on real pairs: 1.9x to 2.3x in standard
deviation.

The search's determinization draw is NOT cancelled by pairing, so its generator is
reseeded per deal. Skipping that once left 67% of a headline interval as
uncancelled search noise.

COST. Measured on a laptop GPU: about 2.4 s per deal at D=8 and 30.6 s at D=128,
so n=500 is ~20 min at D=8 and ~4.3 h at D=128. `tools/dd_depth_sweep.py` plays
both arms in one interleaved loop instead, which pairs them on deals as well as
seats and is far cheaper per unit of resolution. The control always runs first; if it fails the
script exits without spending the rest.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import belot.evaluation.swap_eval as SW
from belot.evaluation.swap_eval import ci
from belot.search.composite import (DEFAULT_D, load_model, make_dd_pimc,
                                    model_backed)

DEFAULT_CKPT = os.path.join("checkpoints", "v8_exp", "expd_latest.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="the base policy, also the default opponent")
    ap.add_argument("--opponent", default=None,
                    help="checkpoint for the opponent; defaults to --ckpt "
                         "(i.e. the composite against its own bare network)")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-control", type=int, default=30)
    ap.add_argument("--D", type=int, default=DEFAULT_D)
    ap.add_argument("--min-trick", type=int, default=3)
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        sys.exit(f"checkpoint not found: {a.ckpt}\n"
                 f"Weights are not distributed with the repo -- see the README.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = np.random.get_state()          # this script must not disturb it
    base = load_model(a.ckpt, dev)
    opp = load_model(a.opponent, dev) if a.opponent else base

    print(f"device {dev}")
    print(f"composite = base + exact-solve PIMC D={a.D} from trick {a.min_trick}")
    print(f"opponent  = {'bare network ' + os.path.basename(a.opponent) if a.opponent else 'its own bare network'}\n")

    c0, r0 = make_dd_pimc(a.D, seed=0, min_trick=a.min_trick)
    c1, r1 = make_dd_pimc(a.D, seed=0, min_trick=a.min_trick)

    print(f"--- CONTROL: the composite against itself, {a.n_control} deals ---")
    print("    (must be exactly zero, or nothing below counts)")
    ctl = SW.swap_edges(model_backed(base, c0, a.min_trick, dev),
                        model_backed(base, c0, a.min_trick, dev), a.n_control,
                        pimc_seed_fn=lambda d: r0(11_000 + d))
    ok = np.abs(ctl).max() == 0.0
    print(f"    {ctl.mean():+.3f} +- {ci(ctl):.3f}   max|edge| {np.abs(ctl).max():.2e}"
          f"   -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        np.random.set_state(state)
        sys.exit("CONTROL FAILED -- identical policies did not cancel.")

    print(f"\n--- COMPOSITE vs BARE NETWORK, {a.n} deals ---")
    t0 = time.time()
    e = SW.swap_edges(model_backed(base, c1, a.min_trick, dev),
                      SW.net_policy(opp, dev), a.n,
                      pimc_seed_fn=lambda d: (r0(12_000 + d), r1(12_000 + d))[0])
    dt = time.time() - t0
    sig = "SIGNIFICANT" if abs(e.mean()) > ci(e) else "not significant"
    print(f"    {e.mean():+.3f} +- {ci(e):.3f} pts/hand   [{sig}]")
    print(f"    deals changed {int((e != 0).sum())}/{a.n}"
          f"   {1e3 * dt / a.n:.0f} ms/deal   {dt / 60:.1f} min total")
    print("\n    reference: +0.974 +- 0.175 at n=1500 against the bare agent.")
    print("    A per-hand edge compounds over a match to 101 (~11.5 hands):")
    print("    +0.5 -> 56% match win, +1.0 -> 62%, +1.9 -> 72%.")
    np.random.set_state(state)


if __name__ == "__main__":
    main()
