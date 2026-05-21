import torch
import numpy as np

class AgentBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.masks = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.h_states = []
        self.c_states = []
        
    def store(self, obs, mask, action, logprob, reward, value, done, h, c):
        self.obs.append(obs)
        self.masks.append(mask)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.h_states.append(h)
        self.c_states.append(c)

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """Calculates Generalized Advantage Estimation (GAE)."""
        values = self.values + [next_value]
        gae = 0
        returns = []
        advantages = []
        
        # Traverse backwards through the sequence
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

class MultiAgentMemory:
    def __init__(self, agents):
        self.agents = agents
        self.buffers = {agent: AgentBuffer() for agent in agents}
        
    def clear_all(self):
        for buf in self.buffers.values():
            buf.clear()