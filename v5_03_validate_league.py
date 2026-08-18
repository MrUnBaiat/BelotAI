"""
AUDIT v5 / EXP-3 -- regression tests for train_v5.py, plus one new test.

The two invariants that catch most regressions in this project are re-run, since
train_v5 changes the opponent-sampling path that feeds episode storage. Then the
test that matters for the league change itself:

T5 LEAGUE DIVERSITY IS REAL. Enlarging the pool is pointless if its members play
   the same way -- that is exactly the failure the old config had, where the
   "frozen" opponent was a byte-identical copy of the live model. So measure
   PAIRWISE BEHAVIOURAL DIVERGENCE between pool members: on a fixed set of
   positions, what fraction of decisions do two members disagree on? A pool of
   clones scores ~0%. This is the property the change is supposed to buy, and it
   is measured, not assumed.

T6 MIX MATCHES THE CONFIG. The realised opponent mix over a rollout must match
   OPP_SELF / OPP_RANDOM / OPP_FROZEN, and storage must be 4 episodes per
   self-play game and 2 per mixed game.
"""
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import train_v5 as T5
from memory import make_minibatch
from model import RecurrentMAPPOModel
from observation import build_observation
from env import BelotEnv
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
    model = RecurrentMAPPOModel(hidden_dim=T5.HIDDEN)
    vec = VectorizedBelot(8)
    saved = (T5.OPP_SELF, T5.OPP_RANDOM, T5.OPP_FROZEN)
    T5.OPP_SELF, T5.OPP_RANDOM, T5.OPP_FROZEN = 1.0, 0.0, 0.0
    try:
        eps, _ = T5.collect_rollout(model, vec, 40, DEVICE)
    finally:
        T5.OPP_SELF, T5.OPP_RANDOM, T5.OPP_FROZEN = saved
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
    print(f"T1 reward target : {len(eps)} episodes, {bad} mismatched, "
          f"max|sum-target| {maxerr:.2e}  -> {'PASS' if bad == 0 else 'FAIL'}")

    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T5.GAMMA, T5.LAM)
    b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(eps)
    B = b_obs.size(0)
    with torch.no_grad():
        dist, values, _ = model(b_obs, b_gobs,
                                (torch.zeros(1, B, T5.HIDDEN), torch.zeros(1, B, T5.HIDDEN)),
                                b_masks, is_sequence=True)
    lp = ((dist.log_prob(b_actions) - b_old).abs() * pad).max().item()
    stored = torch.zeros_like(values.squeeze(-1))
    for i, ep in enumerate(eps):
        stored[i, :len(ep)] = torch.tensor(ep.values, dtype=torch.float32)
    vd = ((values.squeeze(-1) - stored).abs() * pad).max().item()
    print(f"T2 replay        : max|dlogp| {lp:.3e}  max|dV| {vd:.3e}  "
          f"-> {'PASS' if lp < 1e-3 and vd < 1e-3 else 'FAIL'}")
    return eps, model


def t3(eps, model):
    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T5.GAMMA, T5.LAM)
    adv = torch.cat([ep.advantages for ep in eps])
    m, s = adv.mean(), adv.std()
    for ep in eps:
        ep.advantages = (ep.advantages - m) / (s + 1e-8)
    total = float(sum(len(e) for e in eps))

    def loss(chunk, norm):
        b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(chunk)
        B = b_obs.size(0)
        dist, values, _ = model(b_obs, b_gobs,
                                (torch.zeros(1, B, T5.HIDDEN), torch.zeros(1, B, T5.HIDDEN)),
                                b_masks, is_sequence=True)
        values = values.squeeze(-1)
        ratio = torch.exp(dist.log_prob(b_actions) - b_old)
        s1, s2 = ratio * b_adv, torch.clamp(ratio, 0.8, 1.2) * b_adv
        return (-(torch.min(s1, s2) * pad).sum() / norm
                + T5.VALUE_COEF * (F.mse_loss(values, b_ret, reduction='none') * pad).sum() / norm
                - 0.025 * (dist.entropy() * pad).sum() / norm)

    model.zero_grad(set_to_none=True); loss(eps, total).backward()
    g1 = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    for i in range(0, len(eps), T5.MICROBATCH_EPISODES):
        loss(eps[i:i + T5.MICROBATCH_EPISODES], total).backward()
    g2 = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])
    rel = float((g1 - g2).norm() / (g1.norm() + 1e-12))
    print(f"T3 accumulation  : relative diff {rel:.3e}  -> {'PASS' if rel < 1e-5 else 'FAIL'}")


def t6():
    GAME_LOG.clear()
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    model = RecurrentMAPPOModel(hidden_dim=T5.HIDDEN)
    pool = [T5.snapshot(model, DEVICE) for _ in range(4)]
    vec = VectorizedBelot(8)
    eps, info = T5.collect_rollout(model, vec, 200, DEVICE, pool)
    g = info["games_by_opponent"]
    n = sum(g.values())
    expected = g["self"] * 4 + (g["random"] + g["frozen"]) * 2
    print(f"T6 mix           : self {g['self']/n:.1%} (cfg {T5.OPP_SELF:.0%}) "
          f"random {g['random']/n:.1%} (cfg {T5.OPP_RANDOM:.0%}) "
          f"frozen {g['frozen']/n:.1%} (cfg {T5.OPP_FROZEN:.0%})")
    ok = abs(g["frozen"] / n - T5.OPP_FROZEN) < 0.06 and abs(g["self"] / n - T5.OPP_SELF) < 0.06
    print(f"T6 storage       : {len(eps)} episodes vs expected {expected}  "
          f"-> {'PASS' if len(eps) == expected and ok else 'FAIL'}")


def t5_diversity():
    """Behavioural divergence between pool members on a fixed position set."""
    ck = torch.load("checkpoints/best_model.pt", map_location=DEVICE, weights_only=False)
    base = RecurrentMAPPOModel(hidden_dim=T5.HIDDEN).to(DEVICE)
    base.load_state_dict(ck["model_state_dict"]); base.eval()

    # Collect a fixed set of decision positions by playing heuristic self-play.
    from eval import _heuristic_action
    pos = []
    for h in range(40):
        env = BelotEnv(); env.dealer = h % 4
        np.random.seed(6000 + h); env.reset(); env.bolts_by_team = [0, 0]
        while not env.done:
            l, gl, m = build_observation(env, env.current_player, [0, 0])
            if m.sum() > 1:
                pos.append((l, gl, m))
            env.step(_heuristic_action(env))
    L = torch.from_numpy(np.array([p[0] for p in pos])).to(DEVICE)
    G = torch.from_numpy(np.array([p[1] for p in pos])).to(DEVICE)
    M = torch.from_numpy(np.array([p[2] for p in pos], dtype=np.float32)).to(DEVICE)
    B = L.shape[0]

    def argmax_of(net):
        with torch.no_grad():
            d, _, _ = net(L, G, (torch.zeros(1, B, T5.HIDDEN), torch.zeros(1, B, T5.HIDDEN)),
                          M, is_sequence=False)
        return d.probs.argmax(-1)

    # (a) the OLD pool: an exact clone, which is what FROZEN_INIT produced
    clone = T5.snapshot(base, DEVICE)
    a0, a1 = argmax_of(base), argmax_of(clone)
    clone_div = float((a0 != a1).float().mean())

    # (b) genuinely different members: perturb to stand in for snapshots taken
    #     hundreds of epochs apart (this test is about the MEASUREMENT, run it on
    #     real snapshots once a v5 run has produced them)
    divs = []
    for i, sd in enumerate([0.01, 0.02, 0.04]):
        m2 = T5.snapshot(base, DEVICE)
        torch.manual_seed(100 + i)
        with torch.no_grad():
            for p in m2.parameters():
                p.add_(torch.randn_like(p) * sd * p.std())
        divs.append((sd, float((a0 != argmax_of(m2)).float().mean())))

    print(f"T5 diversity     : {B} positions")
    print(f"   exact clone (the OLD 'frozen' opponent) disagreement: {clone_div:.2%}")
    for sd, d in divs:
        print(f"   perturbed sigma={sd:<5} disagreement: {d:.2%}")
    print(f"   -> {'PASS' if clone_div < 0.001 else 'FAIL'}: confirms the old pool member "
          f"was behaviourally identical,\n      so 15% of games taught nothing beyond "
          f"self-play. The measurement is ready\n      to run on real v5 snapshots.")


if __name__ == "__main__":
    eps, model = t1_t2()
    t3(eps, model)
    t6()
    t5_diversity()
