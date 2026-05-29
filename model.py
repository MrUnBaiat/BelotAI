import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class RecurrentPPOModel(nn.Module):
    def __init__(self, input_dim=513, hidden_dim=512, action_dim=38):
        super(RecurrentPPOModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs, hc, action_mask=None, is_sequence=False):
        if not is_sequence:
            # --- Single Step Rollout Mode ---
            features = self.feature_extractor(obs).unsqueeze(1) 
            lstm_out, new_hc = self.lstm(features, hc)
            features_out = lstm_out.squeeze(1) 
        else:
            # --- Batch Sequence Training Mode ---
            features = self.feature_extractor(obs) 
            features_out, new_hc = self.lstm(features, hc)
            # DO NOT reshape/flatten features_out here!
            
        # logits will naturally be (Batch, Seq_Len, 38) in sequence mode
        logits = self.actor(features_out)
        
        # value will naturally be (Batch, Seq_Len, 1) in sequence mode
        value = self.critic(features_out)
        
        if action_mask is not None:
            bool_mask = action_mask.bool()
            logits = logits.masked_fill(~bool_mask, -1e9)
            
        dist = Categorical(logits=logits)
        
        return dist, value, new_hc