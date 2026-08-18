"""
AUDIT v4 / EXP-1 -- Is the PPO update direction signal or noise?

HYPOTHESIS (V1). The policy is not stuck in a local optimum; it is random-walking.
The measured symptom is that approx_kl per update is 0.015-0.017 and the policy
demonstrably moves, yet absolute strength is flat for 400 epochs and a 1400-epoch
head-to-head came back 50.0%. A local optimum produces SMALL policy motion; a
noise-dominated gradient produces large motion with no progress. If the gradient
estimated from one 128-episode minibatch is essentially uncorrelated with the
gradient from a disjoint 128-episode minibatch, then every optimizer step is
mostly sampling noise and the run cannot improve no matter how long it trains.

WHAT WOULD FALSIFY IT. A high cosine similarity between disjoint-minibatch
gradients (say > 0.3), or an estimated gradient noise scale B_simple comfortably
BELOW the 128-episode step size. Either means the batch is big enough and the
plateau must be explained by something else.

METHOD.
  * Collect one honest training rollout with the epoch-2400 checkpoint, using the
    real collect_rollout() and the real opponent mix, so the data distribution is
    exactly what training sees.
  * Compute GAE and normalise advantages exactly as update() does.
  * Take the FIRST PPO iteration gradient only, where ratio == 1 identically, so
    the actor gradient is the plain policy gradient -E[grad log pi * A].
  * Report pairwise cosine between disjoint 128-episode chunk gradients, cosine
    against the full-rollout gradient, and the McCandlish et al. (2018) gradient
    noise scale B_simple = tr(Sigma) / |G|^2 estimated from the two batch sizes.
  * Repeat for a RANDOM-INIT policy as a positive control. If the untrained policy
    shows a healthy SNR on the same pipeline, the measurement is sound and the
    trained policy has genuinely run out of extractable signal at this batch size.

Also reports actor-vs-critic gradient norms, because MAX_GRAD_NORM=0.5 is applied
to the CONCATENATION of both networks' parameters: if the critic gradient
dominates, joint clipping silently scales the actor's effective learning rate.
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
TARGET_GAMES = 512          # exactly one training rollout
CHUNK = T.MINIBATCH_EPISODES  # 128 -- the real optimizer step size
ENT_COEF = T.anneal(T.ENTROPY_START, T.ENTROPY_END, 2400, T.ENTROPY_ANNEAL_EPOCHS)

ACTOR_PREFIXES = ("actor_feature_extractor", "lstm", "actor")
CRITIC_PREFIXES = ("critic",)


def split_params(model):
    actor, critic = [], []
    for n, p in model.named_parameters():
        (critic if n.startswith(CRITIC_PREFIXES) else actor).append(p)
    return actor, critic


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in params])


def losses_for(model, episodes, valid_norm):
    """Actor / critic / entropy losses for a set of episodes, normalised by
    `valid_norm` (the timestep count of the WHOLE batch being compared), so that
    summing over disjoint chunks equals the full-batch loss exactly."""
    b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(episodes)
    b_obs, b_gobs = b_obs.to(DEVICE), b_gobs.to(DEVICE)
    b_masks, b_actions = b_masks.to(DEVICE), b_actions.to(DEVICE)
    b_old, b_adv, b_ret, pad = (b_old.to(DEVICE), b_adv.to(DEVICE),
                                b_ret.to(DEVICE), pad.to(DEVICE))
    B = b_obs.size(0)
    h0 = torch.zeros(1, B, T.HIDDEN, device=DEVICE)
    c0 = torch.zeros(1, B, T.HIDDEN, device=DEVICE)

    dist, values, _ = model(b_obs, b_gobs, (h0, c0), b_masks, is_sequence=True)
    values = values.squeeze(-1)
    logratio = dist.log_prob(b_actions) - b_old
    ratio = torch.exp(logratio)
    surr1 = ratio * b_adv
    surr2 = torch.clamp(ratio, 1 - T.CLIP_EPSILON, 1 + T.CLIP_EPSILON) * b_adv

    actor_loss = -(torch.min(surr1, surr2) * pad).sum() / valid_norm
    critic_loss = (F.mse_loss(values, b_ret, reduction='none') * pad).sum() / valid_norm
    entropy_loss = (dist.entropy() * pad).sum() / valid_norm
    return actor_loss, critic_loss, entropy_loss, float(ratio.max() - ratio.min())


def grad_of(model, episodes, valid_norm, which="actor"):
    actor_p, critic_p = split_params(model)
    model.zero_grad(set_to_none=True)
    a_l, c_l, e_l, _ = losses_for(model, episodes, valid_norm)
    loss = a_l - ENT_COEF * e_l if which == "actor" else T.VALUE_COEF * c_l
    loss.backward()
    return flat_grad(actor_p if which == "actor" else critic_p).detach().clone()


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def analyse(model, tag, frozen_pool):
    print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}")
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    vec = VectorizedBelot(T.NUM_ENVS)
    episodes, info = T.collect_rollout(model, vec, TARGET_GAMES, DEVICE, frozen_pool)
    print(f"episodes {len(episodes)}  timesteps {sum(len(e) for e in episodes)}  "
          f"mix {info['games_by_opponent']}")

    for ep in episodes:
        ep.returns, ep.advantages = ep.compute_gae(T.GAMMA, T.LAM)
    all_adv = torch.cat([ep.advantages for ep in episodes])
    m, s = all_adv.mean(), all_adv.std()
    for ep in episodes:
        ep.advantages = (ep.advantages - m) / (s + 1e-8)

    random.shuffle(episodes)
    n_chunks = len(episodes) // CHUNK
    chunks = [episodes[i * CHUNK:(i + 1) * CHUNK] for i in range(n_chunks)]
    total_steps = float(sum(len(e) for c in chunks for e in c))

    # --- per-chunk actor gradients, all normalised by the same denominator so
    #     their mean is exactly the full-batch gradient ---
    gs = [grad_of(model, c, total_steps / n_chunks, "actor") for c in chunks]
    G = torch.stack(gs).mean(0)

    pair_cos = [cos(gs[i], gs[j]) for i in range(n_chunks) for j in range(i + 1, n_chunks)]
    full_cos = [cos(g, G) for g in gs]

    gsq_small = float(torch.stack([g.pow(2).sum() for g in gs]).mean())
    gsq_big = float(G.pow(2).sum())
    b_small, b_big = float(CHUNK), float(CHUNK * n_chunks)

    # McCandlish et al. 2018 unbiased estimators
    g2 = (b_big * gsq_big - b_small * gsq_small) / (b_big - b_small)
    trS = (gsq_small - gsq_big) / (1.0 / b_small - 1.0 / b_big)
    b_simple = trS / g2 if g2 > 0 else float("inf")

    print(f"\nactor gradient, chunk size {CHUNK} episodes x {n_chunks} chunks")
    print(f"  pairwise cosine between disjoint chunks : "
          f"{np.mean(pair_cos):+.4f} +- {1.96 * np.std(pair_cos) / np.sqrt(len(pair_cos)):.4f}"
          f"   (min {np.min(pair_cos):+.3f} max {np.max(pair_cos):+.3f})")
    print(f"  cosine(chunk, full-rollout gradient)    : {np.mean(full_cos):+.4f}")
    print(f"  |g_chunk|^2 {gsq_small:.4e}   |G_full|^2 {gsq_big:.4e}   "
          f"ratio {gsq_small / max(gsq_big, 1e-30):.1f}x")
    print(f"  |true G|^2  {g2:.4e}   tr(Sigma) {trS:.4e}")
    print(f"  GRADIENT NOISE SCALE B_simple = {b_simple:,.0f} episodes"
          f"   (optimizer step uses {CHUNK})")
    if g2 > 0:
        snr = np.sqrt(g2 / (trS / b_small + 1e-30))
        print(f"  per-step SNR |G| / sqrt(tr(Sigma)/B)  = {snr:.3f}")

    # --- actor vs critic gradient norms under the JOINT clip ---
    actor_p, critic_p = split_params(model)
    model.zero_grad(set_to_none=True)
    a_l, c_l, e_l, _ = losses_for(model, chunks[0], float(sum(len(e) for e in chunks[0])))
    (a_l + T.VALUE_COEF * c_l - ENT_COEF * e_l).backward()
    na = flat_grad(actor_p).norm().item()
    nc = flat_grad(critic_p).norm().item()
    tot = (na ** 2 + nc ** 2) ** 0.5
    print(f"\njoint grad-norm clip (MAX_GRAD_NORM={T.MAX_GRAD_NORM})")
    print(f"  |grad actor| {na:.4f}   |grad critic| {nc:.4f}   |grad total| {tot:.4f}")
    print(f"  critic share of squared norm: {nc**2 / max(tot**2, 1e-30):.1%}")
    print(f"  clip active: {tot > T.MAX_GRAD_NORM}   "
          f"scale applied to actor: {min(1.0, T.MAX_GRAD_NORM / max(tot, 1e-12)):.3f}")
    return b_simple


def main():
    ckpt = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)
    trained = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)
    trained.load_state_dict(ckpt["model_state_dict"])

    # FROZEN_INIT points at best_model.pt, which is byte-identical to latest here,
    # so the real training pool member is a copy of the model itself.
    pool = [T.snapshot(trained, DEVICE)]
    analyse(trained, "TRAINED policy (epoch 2400) -- the plateaued one", pool)

    fresh = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)
    analyse(fresh, "POSITIVE CONTROL: random-init policy", [T.snapshot(fresh, DEVICE)])


if __name__ == "__main__":
    main()
