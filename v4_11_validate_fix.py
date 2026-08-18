"""
AUDIT v4 / EXP-11 -- targeted regression tests for train_v4.py.

The two invariants that have historically caught regressions in this project are
the reward-target identity (AUDIT_HANDOFF 3.2) and rollout/replay consistency
(3.1). train_v4.py rewrites collect_rollout() and update(), so both are re-run
against it here. A third test is specific to the primary fix.

T1  REWARD TARGET. Every stored episode's reward sum must equal its exact
    zero-sum target (gp_us - gp_them)/16. Catches any damage to reward
    retroaction or the terminal drain.

T2  ROLLOUT/REPLAY CONSISTENCY. Re-evaluating the stored (obs, action) pairs in
    sequence mode must reproduce the rollout log-probs and values exactly, so the
    first PPO ratio is 1.000000. Catches LSTM state-routing, obs/mask storage and
    padding damage.

T3  GRADIENT ACCUMULATION EQUIVALENCE (the fix itself). Accumulating
    MICROBATCH-sized backward passes, each normalised by the WHOLE step's
    timestep count, must produce bit-comparable gradients to a single backward
    pass over all the episodes at once. If this fails, C1 silently rescales the
    loss and every result from the new trainer is void.

T4  D3 REGRESSION. With an empty frozen pool the random opponent must still be
    sampled at the configured rate.
"""
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import train_v4 as T4
from memory import make_minibatch
from model import RecurrentMAPPOModel
from vec_env import VectorizedBelot

DEVICE = torch.device("cpu")
GAME_LOG = []
_orig = VectorizedBelot.finish_and_reset


def patched(self, e, info):
    GAME_LOG.append(list(info["game_points"]))
    _orig(self, e, info)


VectorizedBelot.finish_and_reset = patched


def t1_t2():
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    model = RecurrentMAPPOModel(hidden_dim=T4.HIDDEN)
    vec = VectorizedBelot(8)
    # Force pure self-play so every game stores exactly 4 episodes in seat order,
    # which is what makes the EXACT per-seat target check possible. Under the real
    # mix a mixed env stores only the 2 even seats, so episodes can no longer be
    # indexed 4-per-game -- that path is covered by audit_7_v2_rollout.py, and by
    # t1_mixed() below. (This is the D3 fix working: an empty pool used to give
    # 100% self-play, which is why the old form of this assert passed.)
    saved = (T4.OPP_SELF, T4.OPP_RANDOM, T4.OPP_FROZEN)
    T4.OPP_SELF, T4.OPP_RANDOM, T4.OPP_FROZEN = 1.0, 0.0, 0.0
    try:
        eps, _ = T4.collect_rollout(model, vec, 40, DEVICE)
    finally:
        T4.OPP_SELF, T4.OPP_RANDOM, T4.OPP_FROZEN = saved
    assert len(eps) % 4 == 0, "self-play envs must store 4 episodes per game"
    n_games = len(eps) // 4

    bad, maxerr = 0, 0.0
    for g in range(n_games):
        gp = GAME_LOG[g]
        t0 = (gp[0] - gp[1]) / 16.0
        for i in range(4):
            target = t0 if i % 2 == 0 else -t0
            err = abs(sum(eps[4 * g + i].rewards) - target)
            maxerr = max(maxerr, err)
            bad += err > 1e-9
    print(f"T1 reward target : games={n_games} episodes={len(eps)} "
          f"mismatched={bad} max|sum-target|={maxerr:.2e}")
    print(f"T1 RESULT: {'PASS' if bad == 0 else 'FAIL'}")

    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T4.GAMMA, T4.LAM)
    b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(eps)
    B = b_obs.size(0)
    with torch.no_grad():
        dist, values, _ = model(b_obs, b_gobs,
                                (torch.zeros(1, B, T4.HIDDEN), torch.zeros(1, B, T4.HIDDEN)),
                                b_masks, is_sequence=True)
    lp_diff = ((dist.log_prob(b_actions) - b_old).abs() * pad).max().item()
    stored = torch.zeros_like(values.squeeze(-1))
    for i, ep in enumerate(eps):
        stored[i, :len(ep)] = torch.tensor(ep.values, dtype=torch.float32)
    v_diff = ((values.squeeze(-1) - stored).abs() * pad).max().item()
    ratio = torch.exp(dist.log_prob(b_actions) - b_old)[pad.bool()]
    print(f"T2 replay        : max|dlogp| {lp_diff:.3e}  max|dV| {v_diff:.3e}  "
          f"ratio [{ratio.min():.6f}, {ratio.max():.6f}]")
    print(f"T2 RESULT: {'PASS' if lp_diff < 1e-3 and v_diff < 1e-3 else 'FAIL'}")
    return eps, model


def t3(eps, model):
    """One backward over everything vs micro-batched accumulation."""
    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T4.GAMMA, T4.LAM)
    adv = torch.cat([ep.advantages for ep in eps])
    m, s = adv.mean(), adv.std()
    for ep in eps:
        ep.advantages = (ep.advantages - m) / (s + 1e-8)
    total = float(sum(len(e) for e in eps))

    def losses(chunk, norm):
        (b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad) = make_minibatch(chunk)
        B = b_obs.size(0)
        dist, values, _ = model(b_obs, b_gobs,
                                (torch.zeros(1, B, T4.HIDDEN), torch.zeros(1, B, T4.HIDDEN)),
                                b_masks, is_sequence=True)
        values = values.squeeze(-1)
        ratio = torch.exp(dist.log_prob(b_actions) - b_old)
        s1 = ratio * b_adv
        s2 = torch.clamp(ratio, 1 - T4.CLIP_EPSILON, 1 + T4.CLIP_EPSILON) * b_adv
        a = -(torch.min(s1, s2) * pad).sum() / norm
        c = (F.mse_loss(values, b_ret, reduction='none') * pad).sum() / norm
        e = (dist.entropy() * pad).sum() / norm
        return a + T4.VALUE_COEF * c - 0.005 * e

    model.zero_grad(set_to_none=True)
    losses(eps, total).backward()
    g_one = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])

    model.zero_grad(set_to_none=True)
    for i in range(0, len(eps), T4.MICROBATCH_EPISODES):
        losses(eps[i:i + T4.MICROBATCH_EPISODES], total).backward()
    g_acc = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])

    rel = float((g_one - g_acc).norm() / (g_one.norm() + 1e-12))
    cos = float(torch.dot(g_one, g_acc) / (g_one.norm() * g_acc.norm()))
    print(f"T3 accumulation  : |g_one| {g_one.norm():.6f}  |g_acc| {g_acc.norm():.6f}  "
          f"relative diff {rel:.3e}  cosine {cos:.8f}")
    print(f"T3 RESULT: {'PASS' if rel < 1e-5 else 'FAIL'}")


def t4():
    random.seed(0)
    k = [T4._sample_opponent([])[0] for _ in range(20000)]
    fr = k.count("random") / len(k)
    print(f"T4 empty pool    : random share {fr:.1%} (configured {T4.OPP_RANDOM:.0%}), "
          f"self {k.count('self') / len(k):.1%}")
    print(f"T4 RESULT: {'PASS' if abs(fr - T4.OPP_RANDOM) < 0.02 else 'FAIL'}")


def t1_mixed():
    """Under the REAL opponent mix, every stored episode's reward sum must still
    be an exactly-representable zero-sum target k/16 for an achievable game-point
    difference, and mixed envs must store 2 episodes per game, not 4."""
    GAME_LOG.clear()
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    model = RecurrentMAPPOModel(hidden_dim=T4.HIDDEN)
    pool = [T4.snapshot(model, DEVICE)]
    vec = VectorizedBelot(8)
    eps, info = T4.collect_rollout(model, vec, 60, DEVICE, pool)
    n_games = sum(info["games_by_opponent"].values())
    expected = info["games_by_opponent"]["self"] * 4 + \
        (info["games_by_opponent"]["random"] + info["games_by_opponent"]["frozen"]) * 2

    legal = {round(abs(a - b) / 16.0, 10)
             for a in range(-26, 27) for b in range(-26, 27)}
    bad = sum(1 for ep in eps
              if round(abs(sum(ep.rewards)), 10) not in legal)
    print(f"T1b mixed        : games={n_games} mix={info['games_by_opponent']} "
          f"episodes={len(eps)} expected={expected} off-lattice={bad}")
    print(f"T1b RESULT: {'PASS' if len(eps) == expected and bad == 0 else 'FAIL'}")


if __name__ == "__main__":
    eps, model = t1_t2()
    t1_mixed()
    t3(eps, model)
    t4()
