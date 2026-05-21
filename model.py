import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class RecurrentPPOModel(nn.Module):
    def __init__(self, input_dim=219, hidden_dim=256, action_dim=38):
        super(RecurrentPPOModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Upgrade to nn.LSTM (handles both sequence batching and single steps)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs, hc, action_mask=None, is_sequence=False):
        """
        obs: (Batch, 219) if not sequence else (Batch, Sequence_Length, 219)
        hc: (h, c) hidden states where shape is (1, Batch, 256)
        """
        if not is_sequence:
            # --- Single Step Rollout Mode ---
            features = self.feature_extractor(obs).unsqueeze(1) # Add dummy time dim -> (B, 1, 256)
            lstm_out, new_hc = self.lstm(features, hc)
            features_out = lstm_out.squeeze(1) # Remove time dim -> (B, 256)
        else:
            # --- Batch Sequence Training Mode (Blazing Fast CUDA Unrolling) ---
            batch_size, seq_len, _ = obs.shape
            # Flatten across batch and time to pass through MLPs quickly
            flat_features = self.feature_extractor(obs.view(-1, 219))
            features = flat_features.view(batch_size, seq_len, -1)
            
            features_out, new_hc = self.lstm(features, hc)
            # Flatten output back to feed into linear heads
            features_out = features_out.reshape(-1, features_out.shape[-1])
            
        logits = self.actor(features_out)
        value = self.critic(features_out)
        
        if action_mask is not None:
            if is_sequence:
                action_mask = action_mask.view(-1, action_mask.shape[-1])
            bool_mask = action_mask.bool()
            logits = logits.masked_fill(~bool_mask, -1e9)
            
        dist = Categorical(logits=logits)
        
        return dist, value, new_hc