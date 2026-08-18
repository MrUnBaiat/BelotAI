"""
AUDIT v4 / EXP-7 -- is the critic underfit, and does the privileged input hurt?

EXP-6 showed the critic explains 0.441 of the variance at the hand's first
decision where 0.738 is available (card luck is 73.8% of outcome variance and a
perfect privileged critic removes all of it). That gap is the largest single
lever on tr(Sigma) short of raising the batch. Three competing explanations:

  V7a  UNDERFIT. VALUE_COEF=0.5 with 4 PPO iterations over ~15k timesteps simply
       does not fit the critic far enough each epoch, and it never catches up
       because the target moves. FALSIFIED IF training the SAME architecture to
       convergence on one rollout leaves held-out EV at ~0.65.

  V7b  H1, the handoff's top prior: the PRIVILEGED input hurts. The critic sees
       all four hands, the actor sees one. FALSIFIED IF a critic trained on the
       actor's LOCAL observation reaches HIGHER held-out EV than the global one.
       Note the theory actually points the other way -- since a ~ pi(.|o) makes
       the hidden state independent of the action given o, a privileged baseline
       stays unbiased AND has strictly lower conditional variance -- so I expect
       this to be refuted. Measuring it anyway because it is the handoff's H1.

  V7c  AT CAPACITY. Neither input reaches the ceiling; the network or the
       features are the limit.

METHOD. One honest rollout under the real opponent mix. Per-timestep DISCOUNTED
RETURN-TO-GO is the regression target (not the episode total -- that error is
called out in v4_06). Fit three critics from scratch on a train split and score
held-out EV: global (privileged, the current design), local (actor's view), and
both concatenated. Report EV by timestep index so the bidding decision -- where
EXP-3 measured advantages 2.2x larger than in card play -- is visible separately.
"""
import sys
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
import train as T
from model import RecurrentMAPPOModel
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROLLOUTS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EPOCHS_FIT = 60


def gather(model, pool, n_rollouts):
    """Returns per-timestep local obs, global obs, discounted return-to-go,
    the live critic's own prediction, timestep index and is-bidding flag."""
    L, G, R, V, TI, BID = [], [], [], [], [], []
    for k in range(n_rollouts):
        random.seed(500 + k); np.random.seed(500 + k); torch.manual_seed(500 + k)
        vec = VectorizedBelot(T.NUM_ENVS)
        eps, _ = T.collect_rollout(model, vec, 512, DEVICE, pool)
        for ep in eps:
            g, acc = [], 0.0
            for r in reversed(ep.rewards):          # discounted return-to-go
                acc = r + T.GAMMA * acc
                g.append(acc)
            g.reverse()
            L.append(np.array(ep.obs, dtype=np.float32))
            G.append(np.array(ep.global_obs, dtype=np.float32))
            R.append(np.array(g, dtype=np.float32))
            V.append(np.array(ep.values, dtype=np.float32))
            TI.append(np.arange(len(ep), dtype=np.int64))
            BID.append((np.array(ep.masks)[:, 32:].sum(1) > 0).astype(np.int64))
        print(f"  rollout {k + 1}/{n_rollouts}: {len(eps)} episodes", flush=True)
    cat = lambda xs: np.concatenate(xs, 0)
    return cat(L), cat(G), cat(R), cat(V), cat(TI), cat(BID)


def ev(pred, target):
    return float(1.0 - np.var(target - pred) / (np.var(target) + 1e-12))


def fit_critic(X, y, Xte, yte, tag, hidden=512, epochs=EPOCHS_FIT, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1)).to(DEVICE)          # same arch as the real critic
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X).to(DEVICE); yt = torch.from_numpy(y).to(DEVICE)
    Xv = torch.from_numpy(Xte).to(DEVICE)
    n, bs = len(Xt), 4096
    best = -9.9
    for e in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = nn.functional.mse_loss(net(Xt[idx]).squeeze(-1), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if e % 10 == 9 or e == epochs - 1:
            with torch.no_grad():
                p = net(Xv).squeeze(-1).cpu().numpy()
            best = max(best, ev(p, yte))
    with torch.no_grad():
        p = net(Xv).squeeze(-1).cpu().numpy()
    print(f"    {tag:<28s} held-out EV {ev(p, yte):+.3f}   (best during fit {best:+.3f})")
    return p


def main():
    ck = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    pool = [T.snapshot(model, DEVICE)]

    L, G, R, V, TI, BID = gather(model, pool, ROLLOUTS)
    n = len(R)
    print(f"\ntimesteps {n:,}   target sd {R.std():.4f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = perm[:cut], perm[cut:]

    print("\n" + "=" * 78)
    print("LIVE CRITIC (the one training actually used), scored on held-out steps")
    print("=" * 78)
    print(f"  EV overall                : {ev(V[te], R[te]):+.3f}")
    for name, m in (("bidding steps", BID[te] == 1), ("playing steps", BID[te] == 0)):
        print(f"  EV on {name:<20s}: {ev(V[te][m], R[te][m]):+.3f}   (n={m.sum():,})")
    print("  EV by timestep index within the episode:")
    for t in range(0, 10):
        m = TI[te] == t
        if m.sum() > 200:
            print(f"    t={t}: EV {ev(V[te][m], R[te][m]):+.3f}   "
                  f"target sd {R[te][m].std():.3f}   n={m.sum():,}")

    print("\n" + "=" * 78)
    print("REFIT FROM SCRATCH on the same data (same architecture, 60 epochs Adam)")
    print("=" * 78)
    pg = fit_critic(G[tr], R[tr], G[te], R[te], "global obs (privileged)")
    pl = fit_critic(L[tr], R[tr], L[te], R[te], "local obs (actor's view)")
    both = np.concatenate([L, G], 1)
    pb = fit_critic(both[tr], R[tr], both[te], R[te], "local + global")

    print("\n  by phase:")
    for name, m in (("bidding", BID[te] == 1), ("playing", BID[te] == 0)):
        print(f"    {name:<8s} live {ev(V[te][m], R[te][m]):+.3f} | "
              f"global {ev(pg[m], R[te][m]):+.3f} | "
              f"local {ev(pl[m], R[te][m]):+.3f} | "
              f"both {ev(pb[m], R[te][m]):+.3f}")

    print("\n" + "=" * 78)
    live, gl, lo = ev(V[te], R[te]), ev(pg, R[te]), ev(pl, R[te])
    print(f"V7a UNDERFIT : refit global {gl:+.3f} vs live {live:+.3f} -> "
          f"{'CONFIRMED' if gl > live + 0.02 else 'REFUTED'}")
    print(f"V7b H1 (privileged input hurts): local {lo:+.3f} vs global {gl:+.3f} -> "
          f"{'CONFIRMED' if lo > gl + 0.02 else 'REFUTED'}")


if __name__ == "__main__":
    main()
