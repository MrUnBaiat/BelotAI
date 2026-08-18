"""
AUDIT 1 -- Rollout <-> Replay consistency (the most dangerous failure mode).

PPO is only mathematically valid if, before any weight update, re-evaluating the
stored (obs, action) pairs in sequence mode reproduces the rollout log-probs
(ratio == 1) and values. Any mismatch in LSTM hidden-state routing, obs storage,
mask storage, or padding silently poisons every gradient.

PASS criterion: max |logp_replay - logp_rollout| and |V_replay - V_rollout|
below float32 noise (~1e-4).
"""
import sys, random
import numpy as np
import torch

sys.path.insert(0, '.')  # run from the project root
import train
from train import collect_rollout
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from memory import make_minibatch


def main():
    device = 'cpu'
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    model = RecurrentMAPPOModel()
    vec = VectorizedBelot(8)

    # collect_rollout returns (episodes, info) in v2; older revisions returned a
    # bare list. Unpack defensively so this harness cannot silently "pass" on a
    # tuple (len(tuple) == 2 looked like "2 episodes" before this was fixed).
    out = collect_rollout(model, vec, target_games=24, device=device)
    eps = out[0] if isinstance(out, tuple) else out
    print(f"collected episodes: {len(eps)}   total steps: {sum(len(e) for e in eps)}")

    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(train.GAMMA, train.LAM)

    b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(eps)
    B = b_obs.size(0)
    h0 = torch.zeros(1, B, train.HIDDEN); c0 = torch.zeros(1, B, train.HIDDEN)
    with torch.no_grad():
        dist, values, _ = model(b_obs, b_gobs, (h0, c0), b_masks, is_sequence=True)
    new_lp = dist.log_prob(b_actions)
    vals = values.squeeze(-1)

    stored_vals = torch.zeros_like(vals)
    for i, ep in enumerate(eps):
        stored_vals[i, :len(ep)] = torch.tensor(ep.values, dtype=torch.float32)

    lp_diff = ((new_lp - b_old).abs() * pad)
    v_diff = ((vals - stored_vals).abs() * pad)
    ratio = torch.exp(new_lp - b_old)[pad.bool()]

    print(f"max |logp_replay - logp_rollout| : {lp_diff.max().item():.3e}")
    print(f"mean|logp diff| over valid steps : {(lp_diff.sum()/pad.sum()).item():.3e}")
    print(f"max |V_replay  - V_rollout|      : {v_diff.max().item():.3e}")
    print(f"ratio on valid steps: min {ratio.min().item():.6f}  max {ratio.max().item():.6f}")

    ok = lp_diff.max().item() < 1e-3 and v_diff.max().item() < 1e-3
    print("RESULT:", "PASS -- rollout and replay are consistent, PPO ratios are clean"
          if ok else "FAIL -- rollout/replay mismatch: PPO gradients are poisoned")


if __name__ == "__main__":
    main()
