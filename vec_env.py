"""
In-process lockstep vectorization.

`VectorizedBelot` owns N independent `BelotEnv` instances inside a single
process. It never tries to step them "simultaneously" as one tensor op (the game
logic is branchy and turn-based); instead it batches the ONE expensive shared
operation -- the policy forward pass -- by exposing the active agent's
observation from every env at once.

Each env always has exactly one active agent, and finished envs are auto-reset by
the caller, so the batch handed to the network is always exactly N rows wide.
"""

import numpy as np
from env import BelotEnv
from observation import build_observation


class VectorizedBelot:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [BelotEnv() for _ in range(num_envs)]          # each is reset() in its ctor
        self.match_scores = [[0, 0] for _ in range(num_envs)]      # per-env running match score

    def active_agents(self):
        """Absolute id of the player to move in each env."""
        return [env.current_player for env in self.envs]

    def observe_active(self):
        """
        Build a batched observation for the active agent of every env.

        Returns:
            agents : list[int]          length N, the active player per env
            local  : np.ndarray (N,513)
            glob   : np.ndarray (N,332)
            masks  : np.ndarray (N,38)  float legal-action masks
        """
        agents = self.active_agents()
        local = np.zeros((self.num_envs, 513), dtype=np.float32)
        glob = np.zeros((self.num_envs, 332), dtype=np.float32)
        masks = np.zeros((self.num_envs, 38), dtype=np.float32)

        for e, a in enumerate(agents):
            l, g, m = build_observation(self.envs[e], a, self.match_scores[e])
            local[e] = l
            glob[e] = g
            masks[e] = m.astype(np.float32)

        return agents, local, glob, masks

    def step_env(self, e, action):
        """Advance env `e` by one action. Returns (step_rewards, done, info)."""
        _, step_rewards, done, info = self.envs[e].step(action)
        return step_rewards, done, info

    def finish_and_reset(self, e, info):
        """
        Close out a finished game in env `e`: fold its game points into the
        running match score, deal a fresh hand, and -- if the match crossed 101 --
        wipe the persistent match score and bolt counters (the same lifecycle the
        original wrapper applied at the top of reset()).
        """
        gp = info.get("game_points", [0, 0, 0, 0])
        self.match_scores[e][0] += gp[0]
        self.match_scores[e][1] += gp[1]

        self.envs[e].reset()

        if self.match_scores[e][0] >= 101 or self.match_scores[e][1] >= 101:
            self.match_scores[e] = [0, 0]
            self.envs[e].bolts_by_team = [0, 0]