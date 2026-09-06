"""
Does a deeper determinization search actually play better?

    python tools/dd_depth_sweep.py --n 500 --arms 8,128

WHY THIS AND NOT A REPLAY. Replaying recorded hands at a different D gives the
timing exactly and the decision changes exactly, and says NOTHING about strength:
the moment we play a different card the opponents' replies are a counterfactual
that was never observed. Strength needs deals played out, which is what this does.

WHY BOTH ARMS IN ONE LOOP. `swap_eval._play` seeds the deal from
`BASE_SEED + deal`, so deal `d` is the SAME deal in every run this project has
ever done. Running the arms together therefore pairs them on deals as well as on
seats: the per-deal DIFFERENCE cancels card luck a second time, and its interval
is far tighter than combining two separately-run means. Interleaving also means a
run stopped early still has both arms over the same prefix.

This replicates `swap_edges`' loop rather than calling it, only so per-deal values
can be written out as they are produced -- four hours of compute should not be
lost to a crash at the end. The deal indexing, the seat swap and the per-deal
PIMC reseed are identical; anything else would make the numbers incomparable.

CONTROL FIRST. Each arm is played against itself before anything else. Identical
policies must cancel deal by deal, giving exactly +0.000 with max|edge| exactly 0.
A merely small control means something is leaking and no number below it counts.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import belot.evaluation.swap_eval as SW
from belot.evaluation.swap_eval import ci
from belot.search.composite import load_model, make_dd_pimc, model_backed

DEFAULT_CKPT = os.path.join("checkpoints", "v8_exp", "expd_latest.pt")
OUT = os.path.join("sessions", "dd_depth_sweep.json")


def _arm(base, D, min_trick, dev):
    """(policy, reseed) for the composite at this search setting."""
    act, reseed = make_dd_pimc(D, seed=0, min_trick=min_trick)
    return model_backed(base, act, min_trick, dev), reseed


def control(base, D, min_trick, dev, n):
    """The composite against itself. Must be exactly zero."""
    (xf, xr), reseed = _arm(base, D, min_trick, dev)
    (yf, yr), reseed2 = _arm(base, D, min_trick, dev)
    out = np.empty(n)
    for d in range(n):
        reseed(11_000 + d); reseed2(11_000 + d)
        xr(); yr()
        a = SW._play(d, xf, yf)
        reseed(11_000 + d); reseed2(11_000 + d)
        xr(); yr()
        b = SW._play(d, yf, xf)
        out[d] = (a - b) / 2.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-control", type=int, default=16)
    ap.add_argument("--arms", default="8,128", help="comma-separated D values")
    ap.add_argument("--min-trick", type=int, default=3)
    ap.add_argument("--deal-offset", type=int, default=0,
                    help="start from deal N instead of 0, for a replication on "
                         "FRESH deals. Keep it a multiple of 4: the dealer is "
                         "`deal %% 4`, and seat-rotation symmetry is what makes "
                         "the paired control cancel exactly.")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.deal_offset % 4:
        sys.exit("--deal-offset must be a multiple of 4 (dealer rotation)")

    Ds = [int(x) for x in a.arms.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gstate = np.random.get_state()          # never disturb the global stream
    base = load_model(a.ckpt, dev)

    print(f"device {dev}   deals {a.n}   arms D={Ds} from trick {a.min_trick}",
          flush=True)

    for D in Ds:
        c = control(base, D, a.min_trick, dev, a.n_control)
        ok = np.abs(c).max() == 0.0
        print(f"CONTROL D={D:<4} {c.mean():+.3f} +- {ci(c):.3f}  "
              f"max|edge| {np.abs(c).max():.2e}  -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            np.random.set_state(gstate)
            sys.exit(f"CONTROL FAILED at D={D} -- identical policies did not cancel.")

    arms = {}
    for D in Ds:
        (xf, xr), reseed = _arm(base, D, a.min_trick, dev)
        arms[D] = (xf, xr, reseed)
    net = SW.net_policy(base, dev)

    edges = {D: [] for D in Ds}
    t0 = time.time()
    for i in range(a.n):
        d = i + a.deal_offset
        for D in Ds:
            xf, xr, reseed = arms[D]
            yf, yr = net
            reseed(12_000 + d); xr(); yr()
            dA = SW._play(d, xf, yf)
            reseed(12_000 + d); xr(); yr()
            dB = SW._play(d, yf, xf)
            edges[D].append((dA - dB) / 2.0)

        done = i + 1
        if done % 10 == 0 or done == a.n:
            rate = (time.time() - t0) / done
            row = {"deals": done, "min_trick": a.min_trick,
                   "deal_offset": a.deal_offset,
                   "edges": {str(k): v for k, v in edges.items()},
                   "seconds_per_deal": rate}
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(row, fh)
            msg = f"[{done}/{a.n}] {rate:.1f}s/deal  eta {rate*(a.n-done)/3600:.2f}h"
            for D in Ds:
                e = np.array(edges[D])
                msg += f"   D={D}: {e.mean():+.3f}+-{ci(e):.3f}"
            if len(Ds) == 2:
                diff = np.array(edges[Ds[1]]) - np.array(edges[Ds[0]])
                msg += (f"   PAIRED D{Ds[1]}-D{Ds[0]}: "
                        f"{diff.mean():+.3f}+-{ci(diff):.3f}")
            print(msg, flush=True)

    np.random.set_state(gstate)
    print("done", flush=True)


if __name__ == "__main__":
    main()
