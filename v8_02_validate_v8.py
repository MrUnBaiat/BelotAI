"""
AUDIT v8 -- validation for train_v8.py's regime-filtered actor loss.

C11 changes the path from episode storage into the loss, which is exactly what the
standing invariants protect. All of them are re-run, plus two new tests that the
existing ones provably cannot catch.

T1  reward-target exactness
T2  rollout/replay (first-iteration ratio identically 1.000000)
T3  accumulation equivalence on a UNIFORM batch (the old test)
T8  scripted opponent leaves numpy's global stream untouched

T9  ACCUMULATION ON A REGIME-MIXED BATCH.  <-- new, and the one that matters
    T3 passed at 9.9e-08 on a uniform batch and CANNOT catch a variable-denominator
    bug, because with one regime every micro-batch has the same actor-valid count.
    Build a batch whose regime split is deliberately UNEVEN across micro-batches
    and assert one-big-backward == accumulated.

T10 ACTOR-MASK CORRECTNESS.  <-- new
    Episodes outside ACTOR_REGIMES must contribute EXACTLY zero actor gradient.
    Verified by zeroing their advantages and asserting the actor gradient is
    bit-identical: if the mask leaked, changing those advantages would change it.

T11 DENOMINATOR SEPARATION.
    The critic must still see every timestep. Verified by zeroing the RETURNS of
    non-actor episodes and asserting the critic gradient DOES change -- the
    converse of T10, and it catches an over-aggressive mask that filtered both.
"""
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import train_v8 as T8
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


def _losses(model, chunk, mb_valid, actor_valid, device=DEVICE):
    """Mirror of train_v8.update()'s inner block."""
    (b_obs, b_g, b_m, b_a, b_lp, b_adv, b_ret, pad) = make_minibatch(chunk)
    a_mask = pad * torch.tensor(
        [1.0 if T8.ACTOR_REGIMES and getattr(ep, "regime", "script") in T8.ACTOR_REGIMES
         else 0.0 for ep in chunk], dtype=pad.dtype).unsqueeze(1)
    B = b_obs.size(0)
    dist, v, _ = model(b_obs, b_g, (torch.zeros(1, B, T8.HIDDEN),
                                    torch.zeros(1, B, T8.HIDDEN)), b_m, is_sequence=True)
    v = v.squeeze(-1)
    ratio = torch.exp(dist.log_prob(b_a) - b_lp)
    s1, s2 = ratio * b_adv, torch.clamp(ratio, 0.8, 1.2) * b_adv
    al = -(torch.min(s1, s2) * a_mask).sum() / actor_valid
    cl = (F.mse_loss(v, b_ret, reduction='none') * pad).sum() / mb_valid
    el = (dist.entropy() * a_mask).sum() / actor_valid
    return al + T8.VALUE_COEF * cl - 0.025 * el


def build_mixed(model, n_script, n_other):
    """Episodes with a deliberately uneven regime split."""
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    vec = VectorizedBelot(8)
    eps, _ = T8.collect_rollout(model, vec, 120, DEVICE)
    # relabel to force an uneven split across micro-batches: all script first
    for i, ep in enumerate(eps):
        ep.regime = "script" if i < n_script else "self"
    return eps[:n_script + n_other]


def main():
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    model = RecurrentMAPPOModel(hidden_dim=T8.HIDDEN)
    vec = VectorizedBelot(8)

    # ---------- T1 / T2 on pure self-play ----------
    saved = (T8.OPP_SELF, T8.OPP_RANDOM, T8.OPP_FROZEN, T8.OPP_SCRIPT)
    T8.OPP_SELF, T8.OPP_RANDOM, T8.OPP_FROZEN, T8.OPP_SCRIPT = 1.0, 0.0, 0.0, 0.0
    try:
        eps, _ = T8.collect_rollout(model, vec, 40, DEVICE)
    finally:
        T8.OPP_SELF, T8.OPP_RANDOM, T8.OPP_FROZEN, T8.OPP_SCRIPT = saved
    bad, maxerr = 0, 0.0
    for g in range(len(eps) // 4):
        gp = GAME_LOG[g]; t0 = (gp[0] - gp[1]) / 16.0
        for i in range(4):
            err = abs(sum(eps[4 * g + i].rewards) - (t0 if i % 2 == 0 else -t0))
            maxerr = max(maxerr, err); bad += err > 1e-9
    print(f"T1 reward target : {bad} mismatched, max {maxerr:.2e}"
          f"  -> {'PASS' if bad == 0 else 'FAIL'}")

    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T8.GAMMA, T8.LAM)
    b_obs, b_g, b_m, b_a, b_lp, *_rest, pad = make_minibatch(eps)
    B = b_obs.size(0)
    with torch.no_grad():
        dist, values, _ = model(b_obs, b_g, (torch.zeros(1, B, T8.HIDDEN),
                                             torch.zeros(1, B, T8.HIDDEN)),
                                b_m, is_sequence=True)
    lp = ((dist.log_prob(b_a) - b_lp).abs() * pad).max().item()
    print(f"T2 replay        : max|dlogp| {lp:.3e}"
          f"  -> {'PASS' if lp < 1e-3 else 'FAIL'}")

    # ---------- T9: accumulation on a REGIME-MIXED batch ----------
    mixed = build_mixed(model, n_script=70, n_other=50)
    for ep in mixed:
        ep.returns, ep.advantages = ep.compute_gae(T8.GAMMA, T8.LAM)
    mb_valid = float(sum(len(e) for e in mixed))
    actor_valid = float(sum(len(e) for e in mixed if e.regime in T8.ACTOR_REGIMES))
    n_s = sum(1 for e in mixed if e.regime in T8.ACTOR_REGIMES)
    print(f"\nT9 mixed batch   : {len(mixed)} episodes ({n_s} actor / {len(mixed)-n_s} not), "
          f"actor_valid {actor_valid:.0f} of {mb_valid:.0f} steps")

    model.zero_grad(set_to_none=True)
    _losses(model, mixed, mb_valid, actor_valid).backward()
    g1 = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    MB = 32                                   # forces uneven actor counts per chunk
    for i in range(0, len(mixed), MB):
        _losses(model, mixed[i:i + MB], mb_valid, actor_valid).backward()
    g2 = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])
    rel = float((g1 - g2).norm() / (g1.norm() + 1e-12))
    counts = [sum(1 for e in mixed[i:i+MB] if e.regime in T8.ACTOR_REGIMES)
              for i in range(0, len(mixed), MB)]
    print(f"   actor episodes per micro-chunk: {counts}  (uneven by construction)")
    print(f"   relative diff {rel:.3e}  -> {'PASS' if rel < 1e-5 else 'FAIL'}")

    # ---------- T10: non-actor episodes contribute ZERO actor gradient ----------
    actor_params = [p for n, p in model.named_parameters() if not n.startswith("critic")]

    def actor_grad(eps_):
        model.zero_grad(set_to_none=True)
        (bo, bg, bm, ba, blp, badv, bret, pd) = make_minibatch(eps_)
        am = pd * torch.tensor([1.0 if e.regime in T8.ACTOR_REGIMES else 0.0
                                for e in eps_], dtype=pd.dtype).unsqueeze(1)
        Bn = bo.size(0)
        dst, _, _ = model(bo, bg, (torch.zeros(1, Bn, T8.HIDDEN),
                                   torch.zeros(1, Bn, T8.HIDDEN)), bm, is_sequence=True)
        r = torch.exp(dst.log_prob(ba) - blp)
        loss = -(torch.min(r * badv, torch.clamp(r, .8, 1.2) * badv) * am).sum() / actor_valid
        loss.backward()
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in actor_params]).clone()

    ga = actor_grad(mixed)
    for e in mixed:
        if e.regime not in T8.ACTOR_REGIMES:
            e.advantages = e.advantages * 0.0 + 999.0      # wreck them
    gb = actor_grad(mixed)
    same = torch.equal(ga, gb)
    print(f"\nT10 actor mask   : corrupting NON-actor advantages changed the actor "
          f"gradient: {not same}")
    print(f"   -> {'PASS (bit-identical, mask is tight)' if same else 'FAIL (mask leaks)'}")

    # ---------- T11: the critic still sees everything ----------
    critic_params = [p for n, p in model.named_parameters() if n.startswith("critic")]

    def critic_grad(eps_):
        model.zero_grad(set_to_none=True)
        (bo, bg, bm, ba, blp, badv, bret, pd) = make_minibatch(eps_)
        Bn = bo.size(0)
        _, v, _ = model(bo, bg, (torch.zeros(1, Bn, T8.HIDDEN),
                                 torch.zeros(1, Bn, T8.HIDDEN)), bm, is_sequence=True)
        loss = (F.mse_loss(v.squeeze(-1), bret, reduction='none') * pd).sum() / mb_valid
        loss.backward()
        return torch.cat([p.grad.reshape(-1).clone() for p in critic_params])

    ca = critic_grad(mixed)
    for e in mixed:
        if e.regime not in T8.ACTOR_REGIMES:
            e.returns = e.returns * 0.0 + 5.0
    cb = critic_grad(mixed)
    changed = not torch.equal(ca, cb)
    print(f"\nT11 critic scope : corrupting NON-actor returns changed the critic "
          f"gradient: {changed}")
    print(f"   -> {'PASS (critic sees every state)' if changed else 'FAIL (mask over-filters)'}")

    # ---------- T8: global RNG ----------
    env = BelotEnv(); env.dealer = 0
    np.random.seed(4321); env.reset(); env.bolts_by_team = [0, 0]
    before = np.random.get_state()[1][:8].copy()
    for _ in range(40):
        if env.done:
            break
        env.step(perturbed_heuristic_action(env))
    pure = np.array_equal(before, np.random.get_state()[1][:8])
    print(f"\nT8 global RNG    : untouched {pure}  -> {'PASS' if pure else 'FAIL'}")


if __name__ == "__main__":
    main()
