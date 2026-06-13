import numpy as np
import torch

from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from memory import Episode
import train as T

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")

# Small, fast config
T.NUM_ENVS = 6
T.TARGET_GAMES = 5
T.MINIBATCH_EPISODES = 7
T.PPO_ITERS = 2

model = RecurrentMAPPOModel().to(device)
vec = VectorizedBelot(T.NUM_ENVS)

eps = T.collect_rollout(model, vec, T.TARGET_GAMES, device)

# --- Structural checks ---
assert len(eps) == T.TARGET_GAMES * 4, f"expected {T.TARGET_GAMES*4} timelines, got {len(eps)}"
print(f"Collected {len(eps)} complete episodes from {T.TARGET_GAMES} games.")

# Every episode is non-empty, internally consistent in length, finite-rewarded.
lens = []
for ep in eps:
    L = len(ep)
    lens.append(L)
    assert L >= 8, f"episode too short: {L}"
    assert len(ep.obs) == len(ep.global_obs) == len(ep.masks) == L
    assert len(ep.rewards) == len(ep.values) == len(ep.logprobs) == L
    assert all(np.isfinite(r) for r in ep.rewards)
    assert ep.obs[0].shape == (513,) and ep.global_obs[0].shape == (332,)
print(f"Episode lengths: min={min(lens)} max={max(lens)} mean={np.mean(lens):.1f}")

# Zero-sum sanity: the strategic true-up makes each game's 4 seats sum to ~0 in reward.
# Group episodes back into games of 4 (collection appends 4 at a time per finished game).
for gi in range(T.TARGET_GAMES):
    grp = eps[gi*4:(gi+1)*4]
    total = sum(sum(ep.rewards) for ep in grp)
    assert abs(total) < 1e-3, f"game {gi} reward not zero-sum: {total}"
print("Per-game reward sums are zero-sum (true-up wired correctly).")

# --- GAE + a real update step must run and stay finite ---
a, c, e = T.update(model, torch.optim.Adam(model.parameters(), lr=3e-4), eps, device)
assert all(np.isfinite(x) for x in (a, c, e)), (a, c, e)
print(f"Update OK | Actor {a:.4f} | Critic {c:.4f} | Entropy {e:.4f}")

# --- Batched forward path shape check (N rows in one pass) ---
agents, local, glob, masks = vec.observe_active()
h = torch.zeros(1, T.NUM_ENVS, 512); cstate = torch.zeros(1, T.NUM_ENVS, 512)
dist, value, (nh, nc) = model(torch.from_numpy(local), torch.from_numpy(glob),
                              (h, cstate), torch.from_numpy(masks), is_sequence=False)
assert tuple(value.squeeze(-1).shape) == (T.NUM_ENVS,)
assert tuple(dist.sample().shape) == (T.NUM_ENVS,)
assert nh.shape == (1, T.NUM_ENVS, 512)
print(f"Batched forward pass shapes OK (N={T.NUM_ENVS}).")

print("\nALL SMOKE CHECKS PASSED")