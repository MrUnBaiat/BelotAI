"""
AUDIT v8 / instrument verification -- the swap-pairing control must be EXACTLY zero.

The old harness's control (greedy heuristic vs greedy heuristic, which must be zero by
symmetry) measured -0.245 +- 1.063. Under duplicate-bridge pairing identical policies
cancel by construction, deal by deal, so the control must be 0.000 +- 0.000 with
max|edge| == 0 -- not merely small.

Also measures the variance reduction on genuinely different policy pairs, including the
one that matters (a network vs the greedy heuristic), because the pair I first tested
(perturbed vs exact heuristic) shares identical card play and therefore cancels more
than a realistic comparison will.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from eval import _heuristic_action
from model import RecurrentMAPPOModel
from perturbed_heuristic import perturbed_heuristic_action
from v8_swap_eval import net_policy, scripted_policy, swap_edges, _play, summarise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
CKPT = "checkpoints/v6_exp/c1_continued.pt"


def unpaired(even_fn, odd_fn, reset, n):
    out = np.empty(n)
    for d in range(n):
        reset()
        out[d] = _play(d, even_fn, odd_fn)
    return out


def main():
    heur = scripted_policy(_heuristic_action)
    pert = scripted_policy(perturbed_heuristic_action)
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=512).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    net = net_policy(model, DEVICE)

    print(f"{N} deals, checkpoint {CKPT} (epoch {ck.get('epoch')})\n")
    print("=" * 78)
    print("CONTROL -- identical policies must cancel EXACTLY, deal by deal")
    print("=" * 78)
    c_un = unpaired(_heuristic_action, _heuristic_action, lambda: None, N)
    c_sw = swap_edges(heur, heur, N)
    print("  " + summarise("unpaired  heuristic vs heuristic", c_un))
    print("  " + summarise("SWAPPED   heuristic vs heuristic", c_sw))
    print(f"  max|edge| = {np.abs(c_sw).max():.3e}   nonzero deals = {(c_sw != 0).sum()}")
    ok_ctrl = np.abs(c_sw).max() == 0.0
    print(f"  RESULT: {'PASS -- exactly zero' if ok_ctrl else 'FAIL'}")

    # a network against itself: also must cancel exactly (LSTM state is reset per hand)
    n_sw = swap_edges(net, net_policy(model, DEVICE), N // 3)
    print(f"\n  network vs itself (n={N//3}): max|edge| = {np.abs(n_sw).max():.3e}"
          f"  -> {'PASS' if np.abs(n_sw).max() == 0.0 else 'FAIL'}")

    print("\n" + "=" * 78)
    print("VARIANCE REDUCTION on genuinely different policies")
    print("=" * 78)
    for tag, X, unf in (("perturbed vs exact heuristic", pert, perturbed_heuristic_action),
                        ("network vs exact heuristic", net, None)):
        t0 = time.time()
        if unf is not None:
            u = unpaired(unf, _heuristic_action, lambda: None, N)
        else:
            u = unpaired(net[0], _heuristic_action, net[1], N)
        s = swap_edges(X, heur, N)
        r = u.std(ddof=1) / s.std(ddof=1)
        print(f"\n  {tag}")
        print(f"    unpaired sd {u.std(ddof=1):>7.3f}   CI +-{1.96*u.std(ddof=1)/np.sqrt(N):.3f}")
        print(f"    SWAPPED  sd {s.std(ddof=1):>7.3f}   CI +-{1.96*s.std(ddof=1)/np.sqrt(N):.3f}")
        print(f"    sd ratio {r:.2f}x -> variance {r**2:.2f}x for 2x hands "
              f"=> net {r**2/2:.2f}x deals per unit CI   ({time.time()-t0:.0f}s)")

    # ---- the ACTUAL EXP-D instrument: model vs PIMC ----
    # PIMC's determinization sampling is not cancelled by pairing, so this must be
    # measured rather than extrapolated from the heuristic pair.
    from pimc import make_pimc
    D = 16
    pimc_fn = make_pimc(D=D, seed=0)
    # NOTE: per-deal re-seeding of PIMC's private RNG is a further refinement that
    # would remove more of its residual noise, but make_pimc does not expose the
    # Generator. Measured here WITHOUT it, so this figure is a lower bound on the
    # achievable variance reduction.
    npimc = max(150, N // 3)
    t0 = time.time()
    u = unpaired(net[0], pimc_fn, net[1], npimc)
    s = swap_edges(net, scripted_policy(pimc_fn), npimc)
    r = u.std(ddof=1) / s.std(ddof=1)
    print(f"\n  network vs PIMC (D={D}, n={npimc})  <-- the actual EXP-D instrument")
    print(f"    unpaired sd {u.std(ddof=1):>7.3f}")
    print(f"    SWAPPED  sd {s.std(ddof=1):>7.3f}")
    print(f"    sd ratio {r:.2f}x -> variance {r**2:.2f}x  ({time.time()-t0:.0f}s)")
    vr_pimc = r ** 2

    print("\n" + "=" * 78)
    print("PROJECTED CI on the 3,500-deal EXP-D decision  (MEASURED, not planned)")
    print("=" * 78)
    from statistics import NormalDist
    nd = NormalDist()
    for tag, vr in (("EXP-C as run (unpaired)", 1.0),
                    ("swap-paired, plan assumed", 5.7),
                    ("swap-paired, MEASURED vs PIMC", vr_pimc)):
        ci = 0.287 / np.sqrt(vr)
        se = ci / 1.96
        pw = 1 - nd.cdf(1.96 - 0.3 / se) + nd.cdf(-1.96 - 0.3 / se)
        print(f"  {tag:<32} CI +-{ci:.3f}   power@+0.3 {pw:.1%}")


if __name__ == "__main__":
    main()
