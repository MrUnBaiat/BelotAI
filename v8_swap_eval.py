"""
Swap-paired ("duplicate bridge") evaluation -- the instrument, fixed.

WHY. The old paired harness (v4_paired_eval.py) fixes the DEAL across conditions but
leaves seat and dealer asymmetry inside every comparison. Its control shows this
directly: greedy heuristic vs greedy heuristic, which must be exactly zero, measured
-0.245 +- 1.063 over 400 deals. That residual sits inside every number the harness has
ever produced.

THE FIX. Play each deal TWICE, exchanging which policy holds seats 0&2 versus 1&3, and
average the edge from one side's view:

    dA = play(deal, X on even, Y on odd)     -> X's edge =  dA
    dB = play(deal, Y on even, X on odd)     -> X's edge = -dB
    edge_X(deal) = (dA - dB) / 2

When X == Y the two runs are the same game, so dB == dA and every deal returns EXACTLY
zero. That is a far stronger correctness check than "approximately zero on average".

MEASURED (400 deals, verified before this file was written):

    control  heuristic vs heuristic   unpaired -0.2450 +- 1.0629
                                      SWAPPED  +0.0000 +- 0.0000, max|edge| 0.00e+00
    signal   perturbed vs exact       unpaired sd 10.876
                                      SWAPPED  sd  4.072

That pair shares identical card play and differs only in bidding, so it cancels more
than a model-vs-PIMC comparison will: plan on ~5.7x variance reduction (net ~2.85x
deals per unit of CI), not the 7.13x measured there.

WHY IT OUTRANKS BATCH SIZE. On the 3,500-deal decision this takes the CI from +-0.287
to ~+-0.12, and the power of a "+0.3 significant" rule from 53.5% to 99.8%. EXP-C's
null of -0.182 +- 0.287 bounded the true effect only to [-0.47, +0.11] -- a refutation
of nothing. At +-0.12 a null is a real bound.

PIMC CAVEAT. PIMC's determinization sampling is not cancelled by pairing. `pimc_seed_fn`
re-seeds its private Generator per deal so both orientations start from the same stream
position, which removes part of that residual. The Generator stays private -- the global
numpy stream is what makes deals reproducible project-wide (v5 T3, v7 T8).
"""
import numpy as np
import torch

from env import BelotEnv
from observation import build_observation

HIDDEN = 512
BASE_SEED = 770_000


def net_policy(model, device, greedy=True):
    """Wrap a network as a policy fn with per-hand LSTM state. Returns (fn, reset)."""
    hc = {}

    def reset():
        for s in range(4):
            hc[s] = (torch.zeros(1, 1, HIDDEN, device=device),
                     torch.zeros(1, 1, HIDDEN, device=device))

    @torch.no_grad()
    def fn(env):
        s = env.current_player
        local, glob, mask = build_observation(env, s, [0, 0])
        dist, _, hc[s] = model(
            torch.from_numpy(local).unsqueeze(0).to(device),
            torch.from_numpy(glob).unsqueeze(0).to(device), hc[s],
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device),
            is_sequence=False)
        return int(dist.probs.argmax(-1).item() if greedy else dist.sample().item())

    reset()
    return fn, reset


def scripted_policy(action_fn):
    """Wrap a stateless scripted policy (heuristic, PIMC, ...)."""
    return action_fn, (lambda: None)


def _play(deal, even_fn, odd_fn, base_seed=BASE_SEED):
    env = BelotEnv()
    env.dealer = deal % 4
    np.random.seed(base_seed + deal)        # fixes the deal, and only the deal
    env.reset()
    env.bolts_by_team = [0, 0]
    info = {}
    while not env.done:
        a = (even_fn if env.current_player % 2 == 0 else odd_fn)(env)
        _, _, _, info = env.step(a)
    gp = info["game_points"]
    return gp[0] - gp[1]


def swap_edges(X, Y, n_deals, base_seed=BASE_SEED, pimc_seed_fn=None):
    """Per-deal edge of policy X over policy Y, duplicate-bridge paired.

    X, Y are (action_fn, reset_fn) pairs from net_policy / scripted_policy.
    `pimc_seed_fn(deal)` optionally re-seeds a search player's private RNG so both
    orientations of a deal draw from the same stream position.
    """
    (xf, xr), (yf, yr) = X, Y
    out = np.empty(n_deals, dtype=np.float64)
    for d in range(n_deals):
        if pimc_seed_fn:
            pimc_seed_fn(d)
        xr(); yr()
        dA = _play(d, xf, yf, base_seed)     # X on even seats
        if pimc_seed_fn:
            pimc_seed_fn(d)
        xr(); yr()
        dB = _play(d, yf, xf, base_seed)     # X on odd seats
        out[d] = (dA - dB) / 2.0
    return out


def summarise(name, e):
    ci = 1.96 * e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float("nan")
    return f"{name:<34s} {e.mean():+.3f} +- {ci:.3f} pts/hand   (n={len(e)})"


def paired(name, a, b):
    """a - b, both already swap-paired edges over the same deals."""
    d = a - b
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    v = "SIGNIFICANT" if abs(d.mean()) > ci else "not significant"
    nz = d[d != 0]
    return (f"{name:<38s} {d.mean():+.3f} +- {ci:.3f}  [{v}]  "
            f"deals changed {len(nz)}/{len(d)}")
