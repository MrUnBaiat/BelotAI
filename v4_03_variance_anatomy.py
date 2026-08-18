"""
AUDIT v4 / EXP-3 -- where does the gradient variance actually live?

EXP-1 established that at epoch 2400 the 128-episode actor gradient is pure noise
(disjoint-chunk cosine -0.0005 +- 0.0099; magnitude ratio exactly n_chunks) and
that tr(Sigma) grew from 0.0735 at init to 21.57 at epoch 2400 -- a 293x variance
explosion with advantages unit-normalised in both cases.

My first mechanism guess was rare-action 1/pi blow-up. That is WRONG by
construction: for a Categorical the score function w.r.t. logits is (e_a - pi),
whose norm is bounded by sqrt(2). So the growth has to sit in the Jacobian
d logits / d theta, i.e. in the network's own parameters/activations.

HYPOTHESES TESTED HERE
  V3a  The variance is concentrated in the LSTM block. Section 5 of the handoff
       proved the carried LSTM state contributes NOTHING to strength (zero/garbage/
       swapped state all within CI of baseline) while still flipping ~20% of
       decisions. If the LSTM also dominates tr(Sigma), then the recurrence is
       pure variance with no return -- and deleting it is a free SNR win.
       FALSIFIED IF the LSTM's share of tr(Sigma) is near its share of parameters.

  V3b  The only systematic (non-noise) component of the update is the ENTROPY
       bonus, i.e. the policy is being driven by regularisation drift rather than
       by reward. FALSIFIED IF |G_pg| is comparable to or larger than |G_ent|.

  V3c  The variance is dominated by a small tail of timesteps.
       FALSIFIED IF the top 1% of timesteps carry roughly 1% of the variance.

Everything is computed from ONE honest training rollout under the real opponent
mix, using first-PPO-iteration gradients where ratio == 1 identically.
"""
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import train as T
from memory import make_minibatch
from model import RecurrentMAPPOModel
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_GAMES = 512
CHUNK = T.MINIBATCH_EPISODES
ENT_COEF = T.anneal(T.ENTROPY_START, T.ENTROPY_END, 2400, T.ENTROPY_ANNEAL_EPOCHS)

GROUPS = {
    "actor_mlp": "actor_feature_extractor",
    "lstm": "lstm",
    "actor_head": "actor.",
    "critic": "critic",
}


def group_of(name):
    for g, pref in GROUPS.items():
        if name.startswith(pref):
            return g
    return "other"


def collect(model, pool, seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    vec = VectorizedBelot(T.NUM_ENVS)
    eps, info = T.collect_rollout(model, vec, TARGET_GAMES, DEVICE, pool)
    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T.GAMMA, T.LAM)
    adv = torch.cat([ep.advantages for ep in eps])
    m, s = adv.mean(), adv.std()
    for ep in eps:
        ep.advantages = (ep.advantages - m) / (s + 1e-8)
    return eps, info


def forward(model, episodes):
    b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(episodes)
    t = lambda x: x.to(DEVICE)
    b_obs, b_gobs, b_masks = t(b_obs), t(b_gobs), t(b_masks)
    b_actions, b_old, b_adv, b_ret, pad = (t(b_actions), t(b_old), t(b_adv),
                                           t(b_ret), t(pad))
    B = b_obs.size(0)
    h0 = torch.zeros(1, B, T.HIDDEN, device=DEVICE)
    c0 = torch.zeros(1, B, T.HIDDEN, device=DEVICE)
    dist, values, _ = model(b_obs, b_gobs, (h0, c0), b_masks, is_sequence=True)
    return dist, values.squeeze(-1), b_actions, b_old, b_adv, b_ret, pad, b_masks


def grads_by_group(model, episodes, norm, mode):
    """mode: 'pg' = clipped surrogate only, 'ent' = entropy bonus only."""
    model.zero_grad(set_to_none=True)
    dist, values, b_actions, b_old, b_adv, b_ret, pad, _ = forward(model, episodes)
    if mode == "pg":
        ratio = torch.exp(dist.log_prob(b_actions) - b_old)
        s1 = ratio * b_adv
        s2 = torch.clamp(ratio, 1 - T.CLIP_EPSILON, 1 + T.CLIP_EPSILON) * b_adv
        loss = -(torch.min(s1, s2) * pad).sum() / norm
    else:
        loss = -ENT_COEF * (dist.entropy() * pad).sum() / norm
    loss.backward()
    out = {}
    for n, p in model.named_parameters():
        g = group_of(n)
        if g == "critic":
            continue
        v = (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        out.setdefault(g, []).append(v.detach().clone())
    return {g: torch.cat(v) for g, v in out.items()}


def noise_scale(chunk_grads, label):
    """McCandlish B_simple from n disjoint equal chunks."""
    n = len(chunk_grads)
    G = torch.stack(chunk_grads).mean(0)
    gsq_small = float(torch.stack([g.pow(2).sum() for g in chunk_grads]).mean())
    gsq_big = float(G.pow(2).sum())
    b_s, b_b = float(CHUNK), float(CHUNK * n)
    g2 = (b_b * gsq_big - b_s * gsq_small) / (b_b - b_s)
    trS = (gsq_small - gsq_big) / (1.0 / b_s - 1.0 / b_b)
    bs = trS / g2 if g2 > 0 else float("inf")
    pc = [float(torch.dot(chunk_grads[i], chunk_grads[j]) /
                (chunk_grads[i].norm() * chunk_grads[j].norm() + 1e-12))
          for i in range(n) for j in range(i + 1, n)]
    return dict(label=label, trS=trS, g2=g2, b_simple=bs, G=G,
                cos=float(np.mean(pc)),
                cos_ci=float(1.96 * np.std(pc) / np.sqrt(len(pc))))


def main():
    ck = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    fresh = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)

    # ---------- parameter norms: did the network simply grow? ----------
    print("=" * 78)
    print("PARAMETER NORMS  (trained epoch 2400  vs  random init)")
    print("=" * 78)
    print(f"{'group':<12}{'n_params':>12}{'|theta| trained':>18}{'|theta| init':>16}{'ratio':>9}")
    for g in list(GROUPS) + ["other"]:
        tn = [p for n, p in model.named_parameters() if group_of(n) == g]
        fn = [p for n, p in fresh.named_parameters() if group_of(n) == g]
        if not tn:
            continue
        npar = sum(p.numel() for p in tn)
        a = torch.cat([p.reshape(-1) for p in tn]).norm().item()
        b = torch.cat([p.reshape(-1) for p in fn]).norm().item()
        print(f"{g:<12}{npar:>12,}{a:>18.3f}{b:>16.3f}{a / b:>8.2f}x")

    eps, info = collect(model, [T.snapshot(model, DEVICE)])
    print(f"\nrollout: {len(eps)} episodes, {sum(len(e) for e in eps)} timesteps, "
          f"mix {info['games_by_opponent']}")

    random.shuffle(eps)
    n_chunks = len(eps) // CHUNK
    chunks = [eps[i * CHUNK:(i + 1) * CHUNK] for i in range(n_chunks)]
    norm = float(sum(len(e) for c in chunks for e in c)) / n_chunks

    pg = [grads_by_group(model, c, norm, "pg") for c in chunks]
    en = [grads_by_group(model, c, norm, "ent") for c in chunks]
    groups = sorted(pg[0])

    # ---------- V3a: variance by parameter group ----------
    print("\n" + "=" * 78)
    print("V3a  GRADIENT VARIANCE BY PARAMETER GROUP  (policy-gradient term)")
    print("=" * 78)
    print(f"{'group':<12}{'param share':>13}{'tr(Sigma)':>13}{'var share':>11}"
          f"{'|G|^2':>12}{'B_simple':>13}{'chunk cos':>12}")
    tot_par = sum(p.numel() for n, p in model.named_parameters() if group_of(n) != "critic")
    res = {g: noise_scale([c[g] for c in pg], g) for g in groups}
    tot_var = sum(r["trS"] for r in res.values())
    for g in groups:
        r = res[g]
        npar = sum(p.numel() for n, p in model.named_parameters() if group_of(n) == g)
        bs = "inf" if not np.isfinite(r["b_simple"]) else f"{r['b_simple']:,.0f}"
        print(f"{g:<12}{npar / tot_par:>12.1%}{r['trS']:>13.4e}{r['trS'] / tot_var:>10.1%}"
              f"{r['g2']:>12.2e}{bs:>13}{r['cos']:>+11.4f}")

    # ---------- V3b: policy gradient vs entropy gradient ----------
    print("\n" + "=" * 78)
    print("V3b  IS THE UPDATE DRIVEN BY REWARD OR BY THE ENTROPY BONUS?")
    print("=" * 78)
    all_pg = [torch.cat([c[g] for g in groups]) for c in pg]
    all_en = [torch.cat([c[g] for g in groups]) for c in en]
    rp, re = noise_scale(all_pg, "pg"), noise_scale(all_en, "ent")
    print(f"  |G_policy_gradient| (full rollout) : {rp['G'].norm():.6f}")
    print(f"  |G_entropy_bonus|   (full rollout) : {re['G'].norm():.6f}")
    print(f"  entropy share of the update vector : "
          f"{re['G'].norm() / (rp['G'].norm() + re['G'].norm()):.1%}")
    print(f"  chunk-cosine, policy-gradient term : {rp['cos']:+.4f} +- {rp['cos_ci']:.4f}")
    print(f"  chunk-cosine, entropy term         : {re['cos']:+.4f} +- {re['cos_ci']:.4f}")
    for r in (rp, re):
        bs = "inf" if not np.isfinite(r["b_simple"]) else f"{r['b_simple']:,.0f}"
        print(f"  B_simple, {r['label']:<24s}: {bs} episodes")
    print(f"  cosine(G_pg, G_entropy)            : "
          f"{float(torch.dot(rp['G'], re['G']) / (rp['G'].norm() * re['G'].norm() + 1e-12)):+.4f}")

    # ---------- V3c: tail concentration + per-phase stats ----------
    print("\n" + "=" * 78)
    print("V3c  PER-TIMESTEP ANATOMY (sampled-action prob, advantage, score norm)")
    print("=" * 78)
    with torch.no_grad():
        dist, values, b_actions, b_old, b_adv, b_ret, pad, b_masks = forward(model, eps[:512])
        p_a = torch.exp(dist.log_prob(b_actions))
        n_legal = b_masks.sum(-1)
        score = torch.sqrt((1 - p_a) ** 2 +
                           (dist.probs ** 2).sum(-1) - p_a ** 2)   # ||e_a - pi||
        bid = (b_masks[..., 32:].sum(-1) > 0) & pad.bool()
        play = (~bid) & pad.bool()
        contrib = (score * b_adv.abs())
        for name, m in (("bidding", bid), ("playing", play)):
            if m.sum() == 0:
                continue
            print(f"\n  {name}: n={int(m.sum())}  mean legal actions {n_legal[m].mean():.2f}")
            print(f"    pi(sampled a): mean {p_a[m].mean():.3f}  "
                  f"p05 {p_a[m].float().quantile(.05):.3f}  "
                  f"median {p_a[m].float().median():.3f}")
            print(f"    ||e_a - pi||  : mean {score[m].mean():.3f}  "
                  f"max {score[m].max():.3f}   (bound sqrt(2)=1.414)")
            print(f"    |advantage|   : mean {b_adv[m].abs().mean():.3f}  "
                  f"p99 {b_adv[m].abs().float().quantile(.99):.3f}  "
                  f"max {b_adv[m].abs().max():.3f}")
        c = contrib[pad.bool()].float()
        c2 = (c ** 2).sort(descending=True).values
        tot = c2.sum()
        for frac in (0.01, 0.05, 0.20):
            k = max(1, int(frac * len(c2)))
            print(f"\n  top {frac:>5.0%} of timesteps carry "
                  f"{float(c2[:k].sum() / tot):>6.1%} of sum(score*|adv|)^2")


if __name__ == "__main__":
    main()
