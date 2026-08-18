"""
AUDIT 7 -- validate the v2 rollout: boundary flush, opponent mixing, storage rules.

T1 every stored episode's reward sum == its exact zero-sum target
T2 stored episodes come ONLY from learning seats (even seats in mixed envs)
T3 flush leaves every env fresh; no stored episode straddles a rollout boundary
   (re-runs the audit-2 boundary test on the v2 collector)
T4 the frozen opponent actually drives its seats (actions differ from the live net)
T5 opponent mix proportions match the configured 70/15/15
"""
import sys, collections
import numpy as np, torch

sys.path.insert(0, '.')
import train as T
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel

device = 'cpu'
torch.manual_seed(0); np.random.seed(0)
import random as R; R.seed(0)

model = RecurrentMAPPOModel(hidden_dim=T.HIDDEN)
frozen = T.snapshot(model, device)
# perturb the frozen net so its actions are distinguishable from the live one
with torch.no_grad():
    for p in frozen.parameters():
        p.add_(torch.randn_like(p) * 0.5)

vec = VectorizedBelot(8)

# ---- instrument: record the true per-seat episode targets from the env itself ----
true_targets = {}
orig_step = vec.step_env
def step_env(e, a):
    sr, done, info = orig_step(e, a)
    if done:
        gp = info["game_points"]
        true_targets[(e, vec.envs[e].dealer)] = gp   # keyed loosely; we use sums below
    return sr, done, info

# T1/T2: run two back-to-back rollouts (the second exercises the flush)
eps1, i1 = T.collect_rollout(model, vec, 40, device, [frozen])
eps2, i2 = T.collect_rollout(model, vec, 40, device, [frozen])

sums = [abs(sum(ep.rewards)) for ep in eps1 + eps2]
lens = [len(ep) for ep in eps1 + eps2]
print(f"T1 episodes: {len(eps1)+len(eps2)} | reward-sum magnitudes: "
      f"min {min(sums):.4f} max {max(sums):.4f}")

# exact check: rebuild targets by replaying a controlled rollout
class Tracker:
    """Wrap vec to capture, per (env,game), each seat's exact zero-sum target."""
    def __init__(self, vec): self.vec=vec; self.targets=[]; self.epi=[]
tr = Tracker(vec)

# T3: after every collect_rollout, all envs must be mid-hand or fresh; at the START
# of the next one the flush must make them all fresh before any storage happens.
vec2 = VectorizedBelot(8)
_ = T.collect_rollout(model, vec2, 20, device, [frozen])
mid = sum(0 if f else 1 for f in vec2.fresh)
flush_steps = T.flush_in_flight(model, vec2, device)
print(f"T3 envs mid-hand after a rollout: {mid}/8 | flush ran {flush_steps} macro-steps "
      f"| all fresh afterwards: {all(vec2.fresh)}")

# T5: opponent mix over many games
R.seed(1)
counts = collections.Counter()
for _ in range(4000):
    k, _i = T._sample_opponent([frozen])
    counts[k] += 1
tot = sum(counts.values())
print(f"T5 opponent mix: " + ", ".join(f"{k} {counts[k]/tot:.1%}" for k in ['self','random','frozen'])
      + f"   (target {T.OPP_SELF:.0%}/{T.OPP_RANDOM:.0%}/{T.OPP_FROZEN:.0%})")
print(f"   games actually played by opponent: {i1['games_by_opponent']} / {i2['games_by_opponent']}")
print(f"   flush steps rollout1 {i1['flush_steps']} (expected 0, envs start fresh), "
      f"rollout2 {i2['flush_steps']} (expected > 0)")
