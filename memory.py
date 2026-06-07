import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class AgentBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.global_obs = [] # Added Global State Array
        self.masks = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.h_states = []
        self.c_states = []
        
    def store(self, obs, global_obs, mask, action, logprob, reward, value, done, h, c):
        self.obs.append(obs)
        self.global_obs.append(global_obs) # Store Global State
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
    
    def get_padded_batch(self, advantages, returns):
        ep_obs, ep_gobs, ep_masks, ep_actions, ep_logprobs = [], [], [], [], []
        ep_adv, ep_ret, pad_masks = [], [], []
        
        cur_obs, cur_gobs, cur_masks, cur_actions, cur_logprobs, cur_adv, cur_ret = [], [], [], [], [], [], []
        
        # 1. Split flat lists by episode using the 'done' flags
        for i, done in enumerate(self.dones):
            cur_obs.append(self.obs[i])
            cur_gobs.append(self.global_obs[i]) # Aggregate Global States
            cur_masks.append(self.masks[i])
            cur_actions.append(torch.tensor(self.actions[i]))
            cur_logprobs.append(torch.tensor(self.logprobs[i]))
            cur_adv.append(advantages[i])
            cur_ret.append(returns[i])
            
            # If game ends, or we hit the end of the buffer, package the episode
            if done or i == len(self.dones) - 1:
                ep_obs.append(torch.stack(cur_obs))
                ep_gobs.append(torch.stack(cur_gobs)) # Stack Global States
                ep_masks.append(torch.stack(cur_masks))
                ep_actions.append(torch.stack(cur_actions))
                ep_logprobs.append(torch.stack(cur_logprobs))
                ep_adv.append(torch.stack(cur_adv))
                ep_ret.append(torch.stack(cur_ret))
                
                # Create a sequence of 1s representing valid data steps
                pad_masks.append(torch.ones(len(cur_actions)))
                
                cur_obs, cur_gobs, cur_masks, cur_actions, cur_logprobs, cur_adv, cur_ret = [], [], [], [], [], [], []

        # 2. Pad sequences to create rectangular tensors: (Batch, Seq_Len, ...)
        b_obs = pad_sequence(ep_obs, batch_first=True)
        b_gobs = pad_sequence(ep_gobs, batch_first=True) # Pad Global States
        b_masks = pad_sequence(ep_masks, batch_first=True, padding_value=1.0) # To avoid calculating logarithms of zero on the dummy padded steps.
        b_actions = pad_sequence(ep_actions, batch_first=True)
        b_logprobs = pad_sequence(ep_logprobs, batch_first=True)
        b_adv = pad_sequence(ep_adv, batch_first=True)
        b_ret = pad_sequence(ep_ret, batch_first=True)
        
        # 3. Pad the valid-data mask with 0s
        b_pad_mask = pad_sequence(pad_masks, batch_first=True)

        return b_obs, b_gobs, b_masks, b_actions, b_logprobs, b_adv, b_ret, b_pad_mask
    
class MultiAgentMemory:
    def __init__(self, agents):
        self.agents = agents
        self.buffers = {agent: AgentBuffer() for agent in agents}
        
    def clear_all(self):
        for buf in self.buffers.values():
            buf.clear()