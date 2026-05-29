import numpy as np
import torch
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
from env import BelotEnv

class BelotAECEnv(AECEnv):
    metadata = {'render.modes': ['human'], "name": "belot_v1"}

    def __init__(self):
        super().__init__()
        self.belot = BelotEnv()
        self.agents = [f"player_{i}" for i in range(4)]
        self.possible_agents = self.agents[:]
        
        # Action space: 38 distinct actions
        self.action_spaces = {agent: spaces.Discrete(38) for agent in self.possible_agents}
        
        # Changed Observation Space to 513
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0, high=1, shape=(513,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(38,), dtype=np.int8)
            }) for agent in self.possible_agents
        }
        
        # Track global scores across episodes (until 101)
        self.match_scores = [0, 0] # Team 0, Team 1
        
    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent): return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.belot.reset()
        if self.match_scores[0] >= 101 or self.match_scores[1] >= 101:
            self.match_scores = [0, 0] 
            self.belot.bolts_by_team = [0, 0] # Reset bolts for the new match too!
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # PettingZoo Agent Selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = f"player_{self.belot.current_player}"

    def observe(self, agent):
        abs_id = int(agent[-1])
        team_us = abs_id % 2
        team_them = 1 - team_us
        
        obs = np.zeros(513, dtype=np.float32)
        idx = 0
        
        # 1. Private Hand (32)
        for card in self.belot.hands[abs_id]: obs[idx + card] = 1.0
        idx += 32
        
        # 2. Face-Up Card (32)
        if self.belot.face_up_card is not None and self.belot.phase == "BIDDING":
            obs[idx + self.belot.face_up_card] = 1.0
        idx += 32
        
        # 3. Current Trump (5)
        if self.belot.trump is None: obs[idx] = 1.0
        else: obs[idx + 1 + self.belot.trump] = 1.0
        idx += 5
        
        # 4. Relative Declarer (5)
        if self.belot.declarer is None: obs[idx] = 1.0
        else: obs[idx + 1 + (self.belot.declarer - abs_id) % 4] = 1.0
        idx += 5
        
        # 5. Phase (3)
        if self.belot.phase == "BIDDING":
            if self.belot.bidding_round == 1: obs[idx] = 1.0
            else: obs[idx + 1] = 1.0
        else: obs[idx + 2] = 1.0
        idx += 3
        
        # 6. Current Trick (108) - Left (1), Partner (2), Right (3)
        for rel in [1, 2, 3]:
            abs_p = (abs_id + rel) % 4
            for seq_idx, (p, c) in enumerate(self.belot.current_trick): # 2 for loops seems too much
                if p == abs_p:
                    obs[idx + c] = 1.0
                    obs[idx + 32 + seq_idx] = 1.0 # 4-dim sequence
            idx += 36
            
        # 7. Game Stats (6)
        obs[idx] = self.match_scores[team_us] / 101.0
        obs[idx + 1] = self.match_scores[team_them] / 101.0
        obs[idx + 2] = self.belot.raw_points_by_team[team_us] / 162.0
        obs[idx + 3] = self.belot.raw_points_by_team[team_them] / 162.0
        obs[idx + 4] = self.belot.bolts_by_team[team_us] / 2.0
        obs[idx + 5] = self.belot.bolts_by_team[team_them] / 2.0
        idx += 6
        
        # 8. Relative Dealer (4)
        obs[idx + (self.belot.dealer - abs_id) % 4] = 1.0
        idx += 4
        
        # 9. Last Trick (144) - Me (0), Left (1), Partner (2), Right (3)
        for rel in [0, 1, 2, 3]:
            abs_p = (abs_id + rel) % 4
            for seq_idx, (p, c) in enumerate(self.belot.last_trick): # Again, 2 for loops seems too much
                if p == abs_p:
                    obs[idx + c] = 1.0
                    obs[idx + 32 + seq_idx] = 1.0
            idx += 36
            
        # 10. Belief State Matrix Calc (96)
        unseen = np.ones(32, dtype=bool)
        unseen[self.belot.hands[abs_id]] = False
        unseen[self.belot.graveyard] = False
        for p, c in self.belot.current_trick: unseen[c] = False
        if self.belot.phase == "BIDDING" and self.belot.face_up_card is not None:
            unseen[self.belot.face_up_card] = False
            
        other_players = [(abs_id + 1) % 4, (abs_id + 2) % 4, (abs_id + 3) % 4]
        
        # Vectorized Prob Distribution
        W = np.zeros((3, 32), dtype=np.float32)
        for i, p in enumerate(other_players):
            valid = unseen & ~self.belot.impossible_cards[p] & ~self.belot.known_cards[p]
            W[i, valid] = 1.0
            
        col_sums = W.sum(axis=0)
        col_sums[col_sums == 0] = 1.0 
        P = W / col_sums 
        
        remaining = np.array([len(self.belot.hands[p]) - self.belot.known_cards[p].sum() for p in other_players], dtype=np.float32)
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        
        P = P * (remaining[:, np.newaxis] / row_sums[:, np.newaxis])
        P = np.clip(P, 0.0, 1.0)
        
        belief_matrix = np.zeros((3, 32), dtype=np.float32)
        for i, p in enumerate(other_players):
            belief_matrix[i, self.belot.known_cards[p]] = 1.0
            unresolved_mask = ~self.belot.known_cards[p]
            belief_matrix[i, unresolved_mask] = P[i, unresolved_mask]
            
            obs[idx : idx + 32] = belief_matrix[i]
            idx += 32
            
        # 11. Trick Number (8)
        trick_idx = min(self.belot.tricks_played, 7) # Bounds protection
        obs[idx + trick_idx] = 1.0
        idx += 8
        
        # Action Masking Extraction
        # Note: If it's not the agent's turn, PettingZoo expects a 0 mask to prevent learning on dummy turns.
        if f"player_{self.belot.current_player}" == agent and not self.belot.done:
            legal_mask = self.belot.get_legal_actions().astype(np.int8)
        else:
            legal_mask = np.zeros(38, dtype=np.int8)
            
        # 12. Valid Actions Feature Injection (38)
        obs[idx : idx + 38] = legal_mask.astype(np.float32)
        idx += 38
        
        # 13. The Graveyard (32)
        for c in self.belot.graveyard: obs[idx + c] = 1.0
        idx += 32
        
        # Integrity Assert
        assert idx == 513, f"Expected exactly 513 features, but built vector with {idx}"
            
        return {"observation": obs, "action_mask": legal_mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._clear_rewards()
        
        # Step the underlying environment
        _, step_rewards, done, _ = self.belot.step(action)
        
        if done:
            # Map final game points to rewards and update match score
            for i in range(4):
                agent_name = f"player_{i}"
                self.rewards[agent_name] = step_rewards[i]
                self.terminations[agent_name] = True
            
            # Update global match scores (assuming Team 0 is P0/P2, Team 1 is P1/P3)
            self.match_scores[0] += step_rewards[0] 
            self.match_scores[1] += step_rewards[1]
        
        if self._agent_selector.is_last():
            # If all are done, do nothing. Otherwise step to next.
            pass
            
        if done: self.agent_selection = self._agent_selector.next()
        else: self.agent_selection = f"player_{self.belot.current_player}" 
        self._accumulate_rewards()