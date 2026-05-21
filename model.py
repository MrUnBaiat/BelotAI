import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class RecurrentPPOModel(nn.Module):
    def __init__(self, input_dim=219, hidden_dim=256, action_dim=38):
        super(RecurrentPPOModel, self).__init__()
        
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Recurrent Core (LSTMCell for sequential step-by-step unrolling)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Actor Critic Heads
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs, hc, action_mask=None):
        """
        obs: Tensor (Batch, 219)
        hc: Tuple of Tensors ((Batch, 256), (Batch, 256)) representing hidden and cell states
        action_mask: Tensor (Batch, 38)
        """
        features = self.feature_extractor(obs)
        
        # Data flow through LSTMCell
        h, c = self.lstm(features, hc)
        
        # Critic Value Evaluation
        value = self.critic(h)
        
        # Actor Action Generation
        logits = self.actor(h)
        
        if action_mask is not None:
            # Mask illegal actions by setting their logit to -1e9
            # action_mask is expected to be 1 for legal, 0 for illegal
            bool_mask = action_mask.bool()
            logits = logits.masked_fill(~bool_mask, -1e9)
            
        # Create categorical distribution for action sampling
        dist = Categorical(logits=logits)
        
        return dist, value, (h, c)