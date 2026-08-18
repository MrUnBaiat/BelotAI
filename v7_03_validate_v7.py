"""
AUDIT v7 / EXP-C validation -- train_v7.py before any strength number.

train_v7 changes the opponent-sampling path AND the rollout stopping rule, both of
which feed episode storage. That is exactly the path T1/T2/T3 protect, so all
three are re-run, plus tests specific to the new code.

T1  reward-target exactness   every stored episode sums to (gp_us-gp_them)/16
T2  rollout/replay            first-iteration ratio identically 1.000000
T3  accumulation equivalence  micro-batched == one big backward
T4  scripted-opponent legality  the perturbed heuristic never plays illegally
                                inside the vectorised rollout
T5  mix matches config        realised self/random/script/frozen vs configured
T6  storage accounting        self games store 4 episodes, mixed store 2
T7  EPISODE budget            target_episodes stops on episodes, not games -- the
                              v5 section 5.2 confound this exists to remove
T8  global RNG discipline     the perturbed heuristic must not advance numpy's
                              global stream (it uses a private Generator), or
                              paired-deal reproducibility breaks project-wide
"""
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import train_v7 as T7
from memory import make_minibatch
from model import RecurrentMAPPOModel
from env import BelotEnv
from perturbed_heuristic import perturbed_heuristic_action
from vec_env import VectorizedBelot

DEVICE = torch.device("cpu")
GAME_LOG = []
_orig = VectorizedBelot.finish_and_reset


def patched(self, e, info):
    GAME_LOG.append(list(info["game_points"]))
    _orig(self, e, info)


VectorizedBelot.finish_and_reset = patched


def t1_t2_t3():
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    model = RecurrentMAPPOModel(hidden_dim=T7.HIDDEN)
    vec = VectorizedBelot(8)
    saved = (T7.OPP_SELF, T7.OPP_RANDOM, T7.OPP_FROZEN, T7.OPP_SCRIPT)
    T7.OPP_SELF, T7.OPP_RANDOM, T7.OPP_FROZEN, T7.OPP_SCRIPT = 1.0, 0.0, 0.0, 0.0
    try:
        eps, _ = T7.collect_rollout(model, vec, 40, DEVICE)
    finally:
        T7.OPP_SELF, T7.OPP_RANDOM, T7.OPP_FROZEN, T7.OPP_SCRIPT = saved

    bad, maxerr = 0, 0.0
    for g in range(len(eps) // 4):
        gp = GAME_LOG[g]
        t0 = (gp[0] - gp[1]) / 16.0
        for i in range(4):
            err = abs(sum(eps[4 * g + i].rewards) - (t0 if i % 2 == 0 else -t0))
            maxerr = max(maxerr, err); bad += err > 1e-9
    print(f"T1 reward target : {len(eps)} eps, {bad} mismatched, max {maxerr:.2e}"
          f"  -> {'PASS' if bad == 0 else 'FAIL'}")

    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T7.GAMMA, T7.LAM)
    b_obs, b_g, b_m, b_a, b_lp, b_adv, b_ret, pad = make_minibatch(eps)
    B = b_obs.size(0)
    with torch.no_grad():
        dist, values, _ = model(b_obs, b_g, (torch.zeros(1, B, T7.HIDDEN),
                                             torch.zeros(1, B, T7.HIDDEN)),
                                b_m, is_sequence=True)
    lp = ((dist.log_prob(b_a) - b_lp).abs() * pad).max().item()
    stored = torch.zeros_like(values.squeeze(-1))
    for i, ep in enumerate(eps):
        stored[i, :len(ep)] = torch.tensor(ep.values, dtype=torch.float32)
    vd = ((values.squeeze(-1) - stored).abs() * pad).max().item()
    print(f"T2 replay        : max|dlogp| {lp:.3e}  max|dV| {vd:.3e}"
          f"  -> {'PASS' if lp < 1e-3 and vd < 1e-3 else 'FAIL'}")

    adv = torch.cat([ep.advantages for ep in eps])
    m, s = adv.mean(), adv.std()
    for ep in eps:
        ep.advantages = (ep.advantages - m) / (s + 1e-8)
    total = float(sum(len(e) for e in eps))

    def loss(chunk, norm):
        o, g, msk, a, lpv, advv, ret, p = make_minibatch(chunk)
        Bc = o.size(0)
        d, v, _ = model(o, g, (torch.zeros(1, Bc, T7.HIDDEN),
                               torch.zeros(1, Bc, T7.HIDDEN)), msk, is_sequence=True)
        v = v.squeeze(-1)
        ratio = torch.exp(d.log_prob(a) - lpv)
        s1, s2 = ratio * advv, torch.clamp(ratio, 0.8, 1.2) * advv
        return (-(torch.min(s1, s2) * p).sum() / norm
                + T7.VALUE_COEF * (F.mse_loss(v, ret, reduction='none') * p).sum() / norm
                - 0.025 * (d.entropy() * p).sum() / norm)

    model.zero_grad(set_to_none=True); loss(eps, total).backward()
    g1 = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    for i in range(0, len(eps), T7.MICROBATCH_EPISODES):
        loss(eps[i:i + T7.MICROBATCH_EPISODES], total).backward()
    g2 = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])
    rel = float((g1 - g2).norm() / (g1.norm() + 1e-12))
    print(f"T3 accumulation  : relative diff {rel:.3e}"
          f"  -> {'PASS' if rel < 1e-5 else 'FAIL'}")


def t4_t5_t6():
    GAME_LOG.clear()
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    model = RecurrentMAPPOModel(hidden_dim=T7.HIDDEN)
    pool = [T7.snapshot(model, DEVICE) for _ in range(3)]
    vec = VectorizedBelot(8)

    bad_legal = [0]
    orig_step = VectorizedBelot.step_env

    def checked(self, e, action):
        if not self.envs[e].get_legal_actions()[action]:
            bad_legal[0] += 1
        return orig_step(self, e, action)

    VectorizedBelot.step_env = checked
    try:
        eps, info = T7.collect_rollout(model, vec, 400, DEVICE, pool)
    finally:
        VectorizedBelot.step_env = orig_step

    g = info["games_by_opponent"]
    n = sum(g.values())
    expected = g["self"] * 4 + (g["random"] + g["frozen"] + g["script"]) * 2
    print(f"T4 script legality: {bad_legal[0]} illegal actions"
          f"  -> {'PASS' if bad_legal[0] == 0 else 'FAIL'}")
    cfg = {"self": T7.OPP_SELF, "random": T7.OPP_RANDOM,
           "frozen": T7.OPP_FROZEN, "script": T7.OPP_SCRIPT}
    line = "  ".join(f"{k} {g[k]/n:.0%}(cfg {cfg[k]:.0%})" for k in cfg)
    ok_mix = all(abs(g[k] / n - cfg[k]) < 0.07 for k in cfg)
    print(f"T5 mix           : {line}  -> {'PASS' if ok_mix else 'FAIL'}")
    print(f"T6 storage       : {len(eps)} episodes vs expected {expected}"
          f"  -> {'PASS' if len(eps) == expected else 'FAIL'}")


def t7():
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    model = RecurrentMAPPOModel(hidden_dim=T7.HIDDEN)
    pool = [T7.snapshot(model, DEVICE)]
    vec = VectorizedBelot(8)
    target = 600
    eps, info = T7.collect_rollout(model, vec, 100000, DEVICE, pool,
                                   target_episodes=target)
    n = sum(info["games_by_opponent"].values())
    ok = target <= len(eps) <= target + 8   # may overshoot by up to one env-batch
    print(f"T7 episode budget: asked {target} episodes, got {len(eps)} "
          f"from {n} games  -> {'PASS' if ok else 'FAIL'}")


def t8():
    env = BelotEnv(); env.dealer = 0
    np.random.seed(4321); env.reset(); env.bolts_by_team = [0, 0]
    before = np.random.get_state()[1][:8].copy()
    for _ in range(40):
        if env.done:
            break
        env.step(perturbed_heuristic_action(env))
    after = np.random.get_state()[1][:8].copy()
    pure = np.array_equal(before, after)
    print(f"T8 global RNG    : untouched {pure}  -> {'PASS' if pure else 'FAIL'}")


if __name__ == "__main__":
    t1_t2_t3()
    t4_t5_t6()
    t7()
    t8()
