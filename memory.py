"""
Episode-centric rollout storage.

The old AgentBuffer kept one flat list per agent and re-derived episode
boundaries from `done` flags. That only stays correct if consecutive entries
belong to the same game -- which is exactly the invariant vectorization breaks,
because turns from different games arrive interleaved.

So we flip the model: each `Episode` IS one contiguous (env, agent, game)
timeline. There are no intra-episode `done` flags to reason about, and the GAE
bootstrap is unconditionally 0.0 because every stored Episode is complete (any
game still in flight at the collection budget is discarded, never stored).

Because the policy and critic are parameter-shared, episodes from every env and
every seat are pooled into one training set for the update.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class Episode:
    """One complete trajectory: a single seat's view of a single game."""

    def __init__(self):
        self.obs = []
        self.global_obs = []
        self.masks = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []          # placeholders, filled by credit_pending_reward
        self.returns = None        # set during the update phase
        self.advantages = None

    def add(self, obs, global_obs, mask, action, logprob, value):
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.masks.append(mask)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(0.0)   # overwritten on this seat's NEXT action / terminal drain

    def credit_pending_reward(self, reward):
        """
        Reward retroaction: the consequence of an action only resolves later
        (trick completion / end-of-hand true-up), so the reward accumulated since
        this seat last acted is attributed to its most recent transition.
        """
        if self.rewards:
            self.rewards[-1] = reward

    def __len__(self):
        return len(self.actions)

    def compute_gae(self, gamma=0.99, lam=0.95):
        """GAE for a COMPLETE episode -> bootstrap value is exactly 0.0."""
        values = self.values + [0.0]
        gae = 0.0
        advantages = [0.0] * len(self.rewards)
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        adv = torch.tensor(advantages, dtype=torch.float32)
        ret = adv + torch.tensor(self.values, dtype=torch.float32)
        return ret, adv


def make_minibatch(episodes):
    """
    Pad a list of Episodes into rectangular (B, T, ...) tensors for the
    sequence-mode forward pass, plus a (B, T) validity mask used to zero out the
    contribution of padded timesteps in every loss term.
    """
    obs       = [torch.tensor(np.array(ep.obs), dtype=torch.float32) for ep in episodes]
    gobs      = [torch.tensor(np.array(ep.global_obs), dtype=torch.float32) for ep in episodes]
    masks     = [torch.tensor(np.array(ep.masks), dtype=torch.float32) for ep in episodes]
    actions   = [torch.tensor(ep.actions, dtype=torch.long) for ep in episodes]
    logprobs  = [torch.tensor(ep.logprobs, dtype=torch.float32) for ep in episodes]
    advantages = [ep.advantages for ep in episodes]
    returns    = [ep.returns for ep in episodes]
    lengths    = [len(ep) for ep in episodes]

    b_obs      = pad_sequence(obs, batch_first=True)
    b_gobs     = pad_sequence(gobs, batch_first=True)
    # pad action masks with 1.0 so a padded (all-illegal) row never collapses to
    # -inf logits and NaNs the log-prob. The pad_mask zeroes these steps anyway.
    b_masks    = pad_sequence(masks, batch_first=True, padding_value=1.0)
    b_actions  = pad_sequence(actions, batch_first=True)
    b_logprobs = pad_sequence(logprobs, batch_first=True)
    b_adv      = pad_sequence(advantages, batch_first=True)
    b_ret      = pad_sequence(returns, batch_first=True)
    pad_mask   = pad_sequence([torch.ones(L) for L in lengths], batch_first=True)

    return b_obs, b_gobs, b_masks, b_actions, b_logprobs, b_adv, b_ret, pad_mask