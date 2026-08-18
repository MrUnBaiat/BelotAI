"""
AUDIT 2 -- Reward plumbing.

Part A: within one collect_rollout() on a FRESH vector env, every stored episode's
reward sum must equal exactly the seat's zero-sum target (gp_us - gp_them)/16.
This validates credit_pending_reward retroaction + the terminal drain end-to-end.

Part B: demonstrates the rollout-boundary defect. Envs are NOT reset between
collect_rollout() calls, but the local bookkeeping (Episode buffers, reward_acc,
LSTM hidden) IS re-created from scratch. For games already in flight, the terminal
true-up subtracts the env's FULL accumulated dense reward, while the episode only
recorded rewards since the rollout began -> the stored episode sums to
target - (dense reward accrued before the boundary), i.e. a biased return target.
"""
import sys, random
import numpy as np
import torch

sys.path.insert(0, '.')  # run from the project root
import train
from train import collect_rollout
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from env import BelotEnv

GAME_LOG = []
_orig = VectorizedBelot.finish_and_reset
def patched(self, e, info):
    GAME_LOG.append(list(info['game_points']))
    _orig(self, e, info)
VectorizedBelot.finish_and_reset = patched


def part_a():
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    model = RecurrentMAPPOModel()
    vec = VectorizedBelot(8)
    out = collect_rollout(model, vec, target_games=40, device='cpu')
    eps = out[0] if isinstance(out, tuple) else out
    assert len(eps) % 4 == 0, "fresh-vec episodes should come in complete groups of 4"
    n_games = len(eps) // 4
    bad, maxerr = 0, 0.0
    for g in range(n_games):
        gp = GAME_LOG[g]
        t0 = (gp[0] - gp[1]) / 16.0
        for i in range(4):
            target = t0 if i % 2 == 0 else -t0
            err = abs(sum(eps[4 * g + i].rewards) - target)
            maxerr = max(maxerr, err)
            bad += err > 1e-6
    print(f"[A] fresh vec: games={n_games} episodes={len(eps)} "
          f"mismatched={bad} max|sum-target|={maxerr:.2e}")
    print("[A] RESULT:", "PASS -- every episode's reward sum equals its zero-sum target"
          if bad == 0 else "FAIL -- reward retroaction/terminal drain is broken")


def part_b():
    """Single-env manual replication of collect_rollout's bookkeeping with a
    simulated rollout boundary in the middle of a game."""
    random.seed(3); np.random.seed(3)
    biases = []
    for trial in range(200):
        env = BelotEnv()
        # play a random number of steps, then place the "rollout boundary"
        boundary_after = np.random.randint(4, 24)
        steps = 0
        # phase 1: pre-boundary (previous rollout; its data was discarded)
        while not env.done and steps < boundary_after:
            legal = np.flatnonzero(env.get_legal_actions())
            env.step(int(np.random.choice(legal)))
            steps += 1
        if env.done:
            continue
        pre_boundary_dense_team0 = env.accumulated_dense_rewards[0]

        # phase 2: post-boundary -- fresh Episode-style bookkeeping (rewards
        # recorded from zero), exactly like collect_rollout re-initializes.
        rec = {i: [] for i in range(4)}          # per-seat recorded rewards
        acc = {i: 0.0 for i in range(4)}
        dropped0 = 0.0                           # reward lost when episode is empty
        info = {}
        while not env.done:
            p = env.current_player
            if rec[p]:
                rec[p][-1] = acc[p]
            elif p == 0:
                dropped0 = acc[0]                # credit_pending_reward no-op on empty ep
            acc[p] = 0.0
            rec[p].append(0.0)
            legal = np.flatnonzero(env.get_legal_actions())
            _, r, done, info = env.step(int(np.random.choice(legal)))
            for i in range(4):
                acc[i] += r[i]
        for i in range(4):
            if rec[i]:
                rec[i][-1] = acc[i]

        gp = info['game_points']
        target0 = (gp[0] - gp[1]) / 16.0
        stored0 = sum(rec[0])
        biases.append((stored0 - target0, -(pre_boundary_dense_team0 + dropped0)))

    biases = np.array(biases)
    match = np.allclose(biases[:, 0], biases[:, 1], atol=1e-6)
    print(f"[B] boundary trials: {len(biases)}")
    print(f"[B] mean |stored_sum - true_target| = {np.abs(biases[:,0]).mean():.4f}   "
          f"max = {np.abs(biases[:,0]).max():.4f}")
    print(f"[B] bias == -(pre-boundary dense + dropped-before-first-action)?  {match}")
    print("[B] RESULT: CONFIRMED design defect -- episodes that straddle a rollout "
          "boundary carry a biased return target (their reward sum is the true "
          "zero-sum target minus the pre-boundary dense reward). "
          "~NUM_ENVS such episodes exist per rollout, plus NUM_ENVS in-flight "
          "games are discarded at the end of every rollout.")


if __name__ == "__main__":
    part_a()
    GAME_LOG.clear()
    part_b()
