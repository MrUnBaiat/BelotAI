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
        
        # Observation space: 219 flat vector + action mask
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0, high=1, shape=(219,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(38,), dtype=np.int8)
            }) for agent in self.possible_agents
        }
        
        # Track global scores across episodes (until 101)
        self.match_scores = [0, 0] # Team 0, Team 1
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]
        
    def action_space(self, agent):
        return self.action_spaces[agent]

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
        """Constructs the 219-dim relative state vector for the given agent."""
        abs_id = int(agent[-1])
        team_us = abs_id % 2
        team_them = 1 - team_us
        
        obs = np.zeros(219, dtype=np.float32)
        idx = 0
        
        # 1. Private Hand: 32-dim multi-hot
        hand = self.belot.hands[abs_id]
        for card in hand:
            obs[idx + card] = 1.0
        idx += 32
        
        # 2. Face-Up Card: 32-dim one-hot
        if self.belot.face_up_card is not None and self.belot.phase == "BIDDING":
            obs[idx + self.belot.face_up_card] = 1.0
        idx += 32
        
        # 3. Current Trump: 5-dim one-hot (None, C, D, H, S)
        if self.belot.trump is None:
            obs[idx] = 1.0
        else:
            obs[idx + 1 + self.belot.trump] = 1.0
        idx += 5
        
        # 4. Relative Contract Holder: 5-dim one-hot (None, Self, Left, Partner, Right)
        if self.belot.declarer is None:
            obs[idx] = 1.0
        else:
            rel_declarer = (self.belot.declarer - abs_id) % 4
            obs[idx + 1 + rel_declarer] = 1.0
        idx += 5
        
        # 5. Current Phase: 3-dim one-hot (Bidding 1, Bidding 2, Playing)
        if self.belot.phase == "BIDDING":
            if self.belot.bidding_round == 1: obs[idx] = 1.0
            else: obs[idx + 1] = 1.0
        else:
            obs[idx + 2] = 1.0
        idx += 3
        
        # 6. Current Trick Cards: 128-dim (4 relative seats * 32 cards)
        for player, card in self.belot.current_trick:
            rel_player = (player - abs_id) % 4
            obs[idx + (rel_player * 32) + card] = 1.0
        idx += 128
        
        # 7. Trick Leader: 4-dim one-hot relative ID
        if len(self.belot.current_trick) > 0:
            leader_abs = self.belot.current_trick[0][0]
        else:
            leader_abs = self.belot.current_player
        rel_leader = (leader_abs - abs_id) % 4
        obs[idx + rel_leader] = 1.0
        idx += 4
        
        # 8. Match Scores: 2 dims (Us / 101, Them / 101)
        obs[idx] = self.match_scores[team_us] / 101.0
        obs[idx + 1] = self.match_scores[team_them] / 101.0
        idx += 2
        
        # 9. Bile Points (Raw Points): 2 dims (Us / 162, Them / 162)
        obs[idx] = self.belot.raw_points_by_team[team_us] / 162.0
        obs[idx + 1] = self.belot.raw_points_by_team[team_them] / 162.0
        idx += 2
        
        # 10. Bolt Counters: 2 dims
        obs[idx] = self.belot.bolts_by_team[team_us] / 2.0
        obs[idx + 1] = self.belot.bolts_by_team[team_them] /2.0
        idx += 2
        
        # 11. Relative Dealer: 4-dim one-hot
        rel_dealer = (self.belot.dealer - abs_id) % 4
        obs[idx + rel_dealer] = 1.0
        idx += 4
        
        # Action Masking Extraction
        # Note: If it's not the agent's turn, PettingZoo expects a 0 mask to prevent learning on dummy turns.
        if f"player_{self.belot.current_player}" == agent and not self.belot.done:
            legal_mask = self.belot.get_legal_actions().astype(np.int8)
        else:
            legal_mask = np.zeros(38, dtype=np.int8)
            
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
            
        if done:
            self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = f"player_{self.belot.current_player}" # Is this necessary?   
        self._accumulate_rewards()