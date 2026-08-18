"""
Is |G| ~ 0 because the SELF-PLAY OBJECTIVE is at a stationary point?

v4 §2 D-A establishes that at batch 128 the gradient is indistinguishable from
noise, and leaves open (p = 0.142) whether |G| > 0 at all — "noise-limited" vs
"converged". That open question decides whether C1 (bigger batch) is sufficient
or futile, and v4 says so.

This script attacks it from an angle the gradient statistics alone cannot reach.

THE ARGUMENT. v4 §4a also found `best_model.pt` is byte-identical to
`latest_model.pt`, so the "frozen" 15% is a copy of the live policy: the real mix
is ~85% self-play. In symmetric self-play with one shared parameter set, the
objective is zero-sum and symmetric, so at a symmetric equilibrium the TRUE
gradient is genuinely ~0 — not drowned, but absent. Absolute strength can be
mediocre there and the policy will still not move. Under that reading, batching
only measures a near-zero gradient more precisely, and C1's null result at
epoch 2700 would be a false negative for the wrong reason.

THE TEST. Estimate gradient SNR separately per opponent regime, on the same
model, with everything else held fixed:

    self       all four seats = current policy   (the ~85% case)
    random     seats 1,3 = uniform-random legal
    heuristic  seats 1,3 = the fixed greedy player (a real, non-descendant policy)

Only learning-seat episodes are stored, exactly as train.py does, so the three
gradients are directly comparable.

READING THE RESULT
  self ~ 0 AND random/heuristic clearly > 0
      -> the self-play objective is stationary; the batch is NOT the binding
         constraint. Fix the opponent distribution first (H4), then re-measure
         B_simple. Do this BEFORE spending a GPU-day on C1 alone.
  all three ~ 0
      -> supports v4's reading: genuinely noise-limited across objectives, C1 is
         the right lever.
  self clearly > 0
      -> refutes this hypothesis outright; go with v4 §7 as written.

Usage:  python v4_13_snr_by_opponent.py checkpoints/latest_model.pt [games_per_regime]
"""
import os, sys, math, random
import numpy as np
import torch

sys.path.insert(0, '.')
from env import BelotEnv
from observation import build_observation
from model import RecurrentMAPPOModel
from memory import Episode, make_minibatch
from eval import _heuristic_action
from perturbed_heuristic import perturbed_heuristic_action

HIDDEN      = 512
GAMMA, LAM  = 0.999, 0.95
CHUNK       = 128          # episodes per gradient chunk == MINIBATCH_EPISODES
N_ENVS      = 64


def zero_state(device, n=1):
    return (torch.zeros(1, n, HIDDEN, device=device),
            torch.zeros(1, n, HIDDEN, device=device))


@torch.no_grad()
def collect(model, regime, n_games, device, seed=0, opp_net=None):
    """Lockstep rollout over N_ENVS envs; store only learning-seat episodes.

    regime: "self" | "random" | "heuristic" | "perturbed" | "frozen"
    `opp_net` supplies the opponent policy for regime == "frozen".
    """
    # REPRODUCIBILITY FIX. env.reset() draws deals from numpy's GLOBAL stream, and
    # this script never seeded it -- so every invocation sampled different deals and
    # the per-run CI (computed over chunk pairs WITHIN one draw) did not capture
    # between-draw variance. That is why `heuristic` came back at B_simple 29,660 in
    # EXP-A and 120,008 here on the same checkpoint, with non-overlapping cosine CIs.
    # Seed it, and seed Python's `random` too (used by the chunk shuffle).
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    envs = [BelotEnv() for _ in range(N_ENVS)]
    for i, e in enumerate(envs):
        e.dealer = i % 4; e.reset()
    opp_hidden = {(e, a): zero_state(device) for e in range(N_ENVS) for a in range(4)}
    hidden  = {(e, a): zero_state(device) for e in range(N_ENVS) for a in range(4)}
    epis    = {(e, a): Episode() for e in range(N_ENVS) for a in range(4)}
    racc    = {(e, a): 0.0 for e in range(N_ENVS) for a in range(4)}
    learning = (lambda a: True) if regime == "self" else (lambda a: a % 2 == 0)

    done_games, out = 0, []
    while done_games < n_games:
        agents = [e.current_player for e in envs]
        local = np.zeros((N_ENVS, 513), np.float32)
        glob  = np.zeros((N_ENVS, 332), np.float32)
        masks = np.zeros((N_ENVS, 38),  np.float32)
        for i, a in enumerate(agents):
            l, g, m = build_observation(envs[i], a, [0, 0])
            local[i], glob[i], masks[i] = l, g, m.astype(np.float32)

        rows = [i for i in range(N_ENVS) if learning(agents[i])]
        acts = np.zeros(N_ENVS, np.int64)
        lps  = np.zeros(N_ENVS, np.float32)
        vals = np.zeros(N_ENVS, np.float32)
        if rows:
            idx = torch.as_tensor(rows, device=device)
            h = torch.cat([hidden[(i, agents[i])][0] for i in rows], dim=1)
            c = torch.cat([hidden[(i, agents[i])][1] for i in rows], dim=1)
            dist, v, (nh, nc) = model(
                torch.from_numpy(local).to(device).index_select(0, idx),
                torch.from_numpy(glob).to(device).index_select(0, idx),
                (h, c),
                torch.from_numpy(masks).to(device).index_select(0, idx),
                is_sequence=False)
            sa = dist.sample(); sl = dist.log_prob(sa)
            sa_, sl_, v_ = sa.cpu().numpy(), sl.cpu().numpy(), v.squeeze(-1).cpu().numpy()
            for j, i in enumerate(rows):
                acts[i], lps[i], vals[i] = sa_[j], sl_[j], v_[j]
                hidden[(i, agents[i])] = (nh[:, j:j+1].contiguous(), nc[:, j:j+1].contiguous())
        opp_rows = [i for i in range(N_ENVS) if not learning(agents[i])]
        if regime == "frozen" and opp_rows and opp_net is not None:
            idx = torch.as_tensor(opp_rows, device=device)
            h = torch.cat([opp_hidden[(i, agents[i])][0] for i in opp_rows], dim=1)
            c = torch.cat([opp_hidden[(i, agents[i])][1] for i in opp_rows], dim=1)
            dist, _, (nh, nc) = opp_net(
                torch.from_numpy(local).to(device).index_select(0, idx),
                torch.from_numpy(glob).to(device).index_select(0, idx),
                (h, c),
                torch.from_numpy(masks).to(device).index_select(0, idx),
                is_sequence=False)
            oa = dist.sample().cpu().numpy()
            for j, i in enumerate(opp_rows):
                acts[i] = oa[j]
                opp_hidden[(i, agents[i])] = (nh[:, j:j+1].contiguous(),
                                              nc[:, j:j+1].contiguous())
        else:
            for i in opp_rows:
                if regime == "random":
                    acts[i] = int(rng.choice(np.flatnonzero(masks[i])))
                elif regime == "perturbed":
                    acts[i] = perturbed_heuristic_action(envs[i])
                else:                                    # "heuristic"
                    acts[i] = _heuristic_action(envs[i])

        for i in range(N_ENVS):
            a = agents[i]; k = (i, a)
            if learning(a):
                epis[k].credit_pending_reward(racc[k]); racc[k] = 0.0
                epis[k].add(local[i].copy(), glob[i].copy(), masks[i].copy(),
                            int(acts[i]), float(lps[i]), float(vals[i]))
            _, sr, dn, info = envs[i].step(int(acts[i]))
            for q in range(4):
                racc[(i, q)] += sr[q]
            if dn:
                for q in range(4):
                    kk = (i, q)
                    epis[kk].credit_pending_reward(racc[kk])
                    if learning(q) and len(epis[kk]) > 0:
                        out.append(epis[kk])
                done_games += 1
                envs[i].reset()
                for q in range(4):
                    epis[(i, q)] = Episode(); racc[(i, q)] = 0.0
                    hidden[(i, q)] = zero_state(device)
                    opp_hidden[(i, q)] = zero_state(device)
                if done_games >= n_games:
                    break
    return out


def flat_actor_grad(model, mb, device):
    """Plain policy gradient (first PPO iteration: ratio == 1 identically)."""
    b_obs, b_g, b_m, b_a, b_lp, b_adv, b_ret, pad = make_minibatch(mb)
    b_obs, b_g, b_m = b_obs.to(device), b_g.to(device), b_m.to(device)
    b_a, b_adv, pad = b_a.to(device), b_adv.to(device), pad.to(device)
    B = b_obs.size(0)
    dist, _, _ = model(b_obs, b_g, (torch.zeros(1, B, HIDDEN, device=device),
                                    torch.zeros(1, B, HIDDEN, device=device)),
                       b_m, is_sequence=True)
    loss = -((dist.log_prob(b_a) * b_adv) * pad).sum() / pad.sum()
    actor_params = [p for n, p in model.named_parameters() if not n.startswith("critic")]
    g = torch.autograd.grad(loss, actor_params, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).flatten()
                      for x, p in zip(g, actor_params)]).detach()


# Keeping 6 regimes x 128 chunks x 2.6M params on GPU needs 8.1 GB and OOMs a 4 GB
# card. Project onto a FIXED random coordinate subset and hold it on CPU: both the
# ratio/cosine statistics and the alignment estimator are ratios in which the
# |S|/d scaling cancels, so the subset is unbiased for all of them (the same trick
# v4_04 relied on).
SUBSET = 262_144
_IDX = None


def _subset_idx(n_params, device):
    global _IDX
    if _IDX is None:
        g = torch.Generator().manual_seed(0)
        _IDX = torch.randperm(n_params, generator=g)[:SUBSET].to(device)
    return _IDX


def chunk_grads(model, episodes, device, max_chunks=None):
    """Per-chunk actor gradients, advantage-normalised within the regime."""
    for ep in episodes:
        ep.returns, ep.advantages = ep.compute_gae(GAMMA, LAM)
    adv = torch.cat([ep.advantages for ep in episodes])
    mu, sd = adv.mean(), adv.std()
    for ep in episodes:
        ep.advantages = (ep.advantages - mu) / (sd + 1e-8)
    random.shuffle(episodes)
    chunks = [episodes[i:i+CHUNK] for i in range(0, len(episodes) - CHUNK + 1, CHUNK)]
    if max_chunks:
        chunks = chunks[:max_chunks]
    out = []
    for c in chunks:
        g = flat_actor_grad(model, c, device)
        out.append(g[_subset_idx(g.numel(), device)].detach().cpu().clone())
    return torch.stack(out)


def unbiased_musq(G):
    """E[g_i . g_j] over i != j is an UNBIASED estimator of |mu|^2: the noise is
    independent across chunks, so the cross terms vanish in expectation. No
    debias subtraction and no division by a near-zero reliability -- both of the
    alternatives I tested (naive cosine of the means; cosine disattenuated by
    split-half reliability) fail badly at these SNRs."""
    M = G @ G.T
    n = G.shape[0]
    return float((M.sum() - M.diagonal().sum()) / (n * (n - 1)))


def alignment(Ga, Gb, boot=2000, seed=0):
    """cos(mu_a, mu_b) from cross-chunk inner products, with a bootstrap CI."""
    mab = float((Ga @ Gb.T).mean())
    maa, mbb = unbiased_musq(Ga), unbiased_musq(Gb)
    cos = mab / np.sqrt(maa * mbb) if (maa > 0 and mbb > 0) else float("nan")
    rng = np.random.default_rng(seed)
    na, nb = Ga.shape[0], Gb.shape[0]
    vals = []
    for _ in range(boot):
        ia = torch.as_tensor(rng.integers(0, na, na), device=Ga.device)
        ib = torch.as_tensor(rng.integers(0, nb, nb), device=Gb.device)
        A, B = Ga[ia], Gb[ib]
        m2 = float((A @ B.T).mean())
        a2, b2 = unbiased_musq(A), unbiased_musq(B)
        if a2 > 0 and b2 > 0:
            vals.append(m2 / np.sqrt(a2 * b2))
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"),) * 2)
    return cos, lo, hi, mab


def snr_from_grads(G):
    """Same statistics as snr(), computed from precomputed chunk gradients so the
    alignment measurement can reuse them instead of re-running the rollouts."""
    n = G.shape[0]
    cos = [float(torch.nn.functional.cosine_similarity(G[i], G[j], dim=0))
           for i in range(n) for j in range(i + 1, n)]
    Gm = G.mean(0)
    ratio = float(G.pow(2).sum(1).mean() / Gm.pow(2).sum())
    cos_mean = float(((G @ Gm) / (G.norm(dim=1) * Gm.norm() + 1e-30)).mean())
    c = np.array(cos)
    ci = 1.96 * c.std(ddof=1) / math.sqrt(len(c)) if len(c) > 1 else float('nan')
    b_simple = CHUNK * (1 - c.mean()) / c.mean() if c.mean() > 1e-9 else float('inf')
    return dict(n_chunks=n, cos=c.mean(), ci=ci, ratio=ratio, cos_to_mean=cos_mean,
                pure_noise_ratio=n, pure_noise_cos=1 / math.sqrt(n), b_simple=b_simple)


def snr(model, episodes, device, max_chunks=None):
    for ep in episodes:
        ep.returns, ep.advantages = ep.compute_gae(GAMMA, LAM)
    adv = torch.cat([ep.advantages for ep in episodes])
    mu, sd = adv.mean(), adv.std()
    for ep in episodes:
        ep.advantages = (ep.advantages - mu) / (sd + 1e-8)
    random.shuffle(episodes)
    chunks = [episodes[i:i+CHUNK] for i in range(0, len(episodes) - CHUNK + 1, CHUNK)]
    # FIX 2: truncate every regime to the SAME chunk count. `self` stores 4
    # episodes/game and random/heuristic store 2, so equal games would hand `self`
    # twice the chunks and a tighter CI -- on precisely the regime we must not
    # over-resolve relative to the others.
    if max_chunks:
        chunks = chunks[:max_chunks]
    grads = [flat_actor_grad(model, c, device) for c in chunks]
    n = len(grads)
    cos = [float(torch.nn.functional.cosine_similarity(grads[i], grads[j], dim=0))
           for i in range(n) for j in range(i+1, n)]
    G = torch.stack(grads).mean(0)
    ratio = float(torch.stack([g.pow(2).sum() for g in grads]).mean() / G.pow(2).sum())
    cos_mean = float(np.mean([float(torch.nn.functional.cosine_similarity(g, G, dim=0))
                              for g in grads]))
    c = np.array(cos)
    ci = 1.96 * c.std(ddof=1) / math.sqrt(len(c)) if len(c) > 1 else float('nan')
    b_simple = CHUNK * (1 - c.mean()) / c.mean() if c.mean() > 1e-9 else float('inf')
    return dict(n_chunks=n, cos=c.mean(), ci=ci, ratio=ratio, cos_to_mean=cos_mean,
                pure_noise_ratio=n, pure_noise_cos=1/math.sqrt(n), b_simple=b_simple)


# FIX 1: hash() on str is salted per process unless PYTHONHASHSEED is set, so the
# original seed=hash(regime) made this script irreproducible across runs.
SEEDS = {"self": 11, "random": 22, "heuristic": 33,
         "perturbed": 44, "frozen_ref": 55, "frozen_new": 66}
EPS_PER_GAME = {"self": 4, "random": 2, "heuristic": 2,
                "perturbed": 2, "frozen_ref": 2, "frozen_new": 2}
# regime -> the collect() opponent kind
KIND = {"self": "self", "random": "random", "heuristic": "heuristic",
        "perturbed": "perturbed", "frozen_ref": "frozen", "frozen_new": "frozen"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/v6_exp/c1_continued.pt"
    target_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    # Seed offset: run the same regimes on an INDEPENDENT data draw. The per-run CI
    # is computed over chunk pairs within one draw and does not capture between-draw
    # variance; repeating at 2-3 offsets is the only way to see it.
    seed_off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    # NOT eval(): cuDNN refuses to backward through an RNN in eval mode
    # ("cudnn RNN backward can only be called in training mode"), and this script
    # needs gradients. The network has no dropout and no batchnorm, so train() is
    # behaviourally identical -- which is why v4's scripts, which built models
    # fresh (train mode by default), never hit this.
    model.train()

    target_eps = target_chunks * CHUNK
    print(f"checkpoint {path}")
    print(f"target {target_chunks} chunks x {CHUNK} episodes = {target_eps:,} episodes "
          f"per regime (episode-equalised, not game-equalised)\n")
    print(f"{'regime':10} {'games':>7} {'chunks':>7} {'|g|^2/|G|^2':>12} {'pure-noise':>11} "
          f"{'deviation':>10} {'disjoint cos':>20} {'B_simple':>12}")
    # frozen opponents: the immutable reference and the newest snapshot
    frozen_nets = {}
    for tag, path in (("frozen_ref", "checkpoints/reference_model.pt"),
                      ("frozen_new", "checkpoints/v6_exp/c1_continued.pt")):
        if os.path.exists(path):
            net = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
            net.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
            net.train()
            for p in net.parameters():
                p.requires_grad_(False)
            frozen_nets[tag] = net
        else:
            print(f"  NOTE: {path} missing -- skipping {tag}")

    regimes = [r for r in ("self", "random", "heuristic", "perturbed",
                           "frozen_ref", "frozen_new")
               if (not r.startswith("frozen") or r in frozen_nets)
               and (only is None or r in only)]
    if seed_off:
        print(f"  seed offset {seed_off} -- INDEPENDENT data draw\n")
    out, grads = {}, {}
    for regime in regimes:
        games = int(np.ceil(target_eps / EPS_PER_GAME[regime]))
        eps = collect(model, KIND[regime], games, device,
                      seed=SEEDS[regime] + 1000 * seed_off,
                      opp_net=frozen_nets.get(regime))
        grads[regime] = chunk_grads(model, eps, device, max_chunks=target_chunks)
        s = snr_from_grads(grads[regime])
        out[regime] = s
        dev = 100.0 * (s['pure_noise_ratio'] - s['ratio']) / s['pure_noise_ratio']
        print(f"{regime:12} {games:>7,} {s['n_chunks']:>7} {s['ratio']:>12.2f} "
              f"{s['pure_noise_ratio']:>11} {dev:>9.1f}% "
              f"{s['cos']:>+11.5f}+-{s['ci']:.5f} {s['b_simple']:>12,.0f}", flush=True)

    # ---------------- cross-regime alignment ----------------
    print("\n" + "=" * 78)
    print("CROSS-REGIME ALIGNMENT  cos(mu_a, mu_b), unbiased cross-chunk estimator")
    print("=" * 78)
    print("The f^2 dilution model assumes diluting regimes are ZERO-MEAN. They are not:")
    print("|mu| ~ B_simple^-1/2 puts `self` at ~0.26x `heuristic`, contributing a")
    print("comparable term to the mixed gradient. If self ANTI-aligns it actively")
    print("cancels, and exclusion is worth more than the f^2 arithmetic implies.\n")
    ref = "perturbed" if "perturbed" in grads else "heuristic"
    # |mu|^2 must be POSITIVE for the normalised cosine to mean anything. The
    # unbiased cross-chunk estimator can go negative when |mu|^2 is small relative
    # to the noise, which is exactly the low-n failure mode -- surface it rather
    # than silently emitting nan or an out-of-range cosine.
    musq = {r: unbiased_musq(grads[r]) for r in regimes}
    print("unbiased |mu|^2 per regime (must be > 0 for the cosine to be meaningful):")
    for r in regimes:
        print(f"    {r:<14}{musq[r]:>12.3e}{'' if musq[r] > 0 else '   <-- NEGATIVE, cosine unusable'}")
    print()
    print(f"{'regime pair':<28}{'cos':>9}{'95% CI':>22}{'mu_a.mu_b':>14}")
    aligns = {}
    for r in regimes:
        if r == ref:
            continue
        cos, lo, hi, mab = alignment(grads[r], grads[ref])
        aligns[r] = (cos, lo, hi)
        # The bootstrap CI is the trustworthy part; the ratio point estimate is
        # skewed and routinely lands outside its own interval. So judge alignment
        # from the CI FIRST and only fall back to the unusable flag when the CI is
        # also uninformative -- the earlier ordering suppressed a clearly-aligned
        # verdict (CI [+0.143,+0.492]) just because the point estimate exceeded 1.
        flag = ""
        if hi < 0:
            flag = "  <-- ANTI-ALIGNED"
        elif lo > 0:
            flag = "  <-- aligned"
        elif not np.isfinite(cos) or abs(cos) > 1.0:
            flag = "  <-- point est. unusable AND CI spans 0"
        print(f"{r + ' vs ' + ref:<28}{cos:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{mab:>14.3e}{flag}")
    print("\nNOTE: mu_a.mu_b is unbiased and interpretable even when the normalised")
    print("cosine is not -- read its SIGN and CI when the normaliser is marginal.")

    # ---------------- inclusion set, by the pre-registered 3x rule ----------------
    best = min(out[r]["b_simple"] for r in regimes)
    inc = [r for r in regimes if out[r]["b_simple"] <= 3 * best]
    print(f"\nINCLUSION SET (pre-registered rule: B_simple within 3x of best={best:,.0f}):")
    print(f"  {inc}")
    actor_eps_share = None
    print("\n" + "=" * 78)
    print("HEADLINE STATISTIC is the ratio |g_chunk|^2/|G_full|^2 against its")
    print("pure-noise value n (v4 measured 13.00 against a predicted 13 -- exact).")
    print("A ratio BELOW n means a systematic component exists; deviation% quantifies it.\n")

    sig = {r: (out[r]['cos'] - out[r]['ci']) > 0 for r in out}
    devs = {r: 100.0 * (out[r]['pure_noise_ratio'] - out[r]['ratio'])
            / out[r]['pure_noise_ratio'] for r in out}
    if sig["self"]:
        print("=> `self` shows a resolvable gradient. The stationary-self-play reading")
        print("   is REFUTED; the nulls have another cause. Stop and re-diagnose.")
    elif sig["random"] or sig["heuristic"]:
        print("=> `self` ~ 0 while another regime resolves. The SELF-PLAY OBJECTIVE is")
        print("   stationary but a real gradient exists elsewhere. Do EXP-C (opponent")
        print("   distribution) BEFORE EXP-B.")
    else:
        print("=> No regime resolves. Report the AMBIGUITY, not a winner: with "
              f"{target_chunks} chunks")
        print(f"   the cosine CI is ~+-{out['self']['ci']:.5f} while a B_simple of 2e5 "
              "predicts only 0.00064,")
        print("   so a genuinely-zero and a modestly-positive `self` are NOT separable")
        print("   by cosine here. Judge on the ratio deviations above:")
        print(f"     self {devs['self']:+.1f}%  random {devs['random']:+.1f}%  "
              f"heuristic {devs['heuristic']:+.1f}%")
        print("   If those are also comparable, exploration (EXP-B) is the next lever.")


if __name__ == "__main__":
    main()