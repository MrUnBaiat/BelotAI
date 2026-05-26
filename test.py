import os
import torch
from env_wrapper import BelotAECEnv
from model import RecurrentPPOModel

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # --- 1. Load the Models ---
    model_1000 = RecurrentPPOModel().to(device)
    model_500 = RecurrentPPOModel().to(device)

    # Ensure the checkpoints exist before attempting to load
    ckpt_1000_path = "checkpoints/model_epoch_3500.pt"
    ckpt_500_path = "checkpoints/model_epoch_3500.pt"

    if not os.path.exists(ckpt_1000_path) or not os.path.exists(ckpt_500_path):
        raise FileNotFoundError("Could not find the specified checkpoints in the 'checkpoints' folder.")

    model_1000.load_state_dict(torch.load(ckpt_1000_path, map_location=device)['model_state_dict'])
    model_500.load_state_dict(torch.load(ckpt_500_path, map_location=device)['model_state_dict'])

    # Set to evaluation mode (disables dropout/batchnorm if you ever add them)
    model_1000.eval()
    model_500.eval()

    env = BelotAECEnv()

    # --- 2. Setup Match Tracking ---
    total_matches = 100
    wins_1000 = 0
    wins_500 = 0
    draws = 0

    print(f"Starting {total_matches} full matches (First to 101 Points)...")
    print("Team 0 (Players 0 & 2): Epoch 1000")
    print("Team 1 (Players 1 & 3): Epoch 500\n")

    # --- 3. The Evaluation Loop ---
    for match in range(1, total_matches + 1):
        # Reset the environment and strictly wipe global game variables for the new match
        env.reset()
        env.match_scores = [0, 0] 
        env.belot.bolts_by_team = [0, 0] 

        # Keep playing hands until someone hits 101 Game Points
        while max(env.match_scores) < 101:
            # We must reset the hand (deck, tricks) but keep match_scores intact
            env.reset() 
            
            # Re-initialize hidden states for the new hand
            hidden_states = {
                agent: (torch.zeros(1, 1, 256, device=device), torch.zeros(1, 1, 256, device=device))
                for agent in env.possible_agents
            }

            for agent in env.agent_iter():
                obs_dict, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    env.step(None) # Dead step required by PettingZoo API
                    continue

                # Prepare inputs
                obs = torch.tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.tensor(obs_dict["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
                hc = hidden_states[agent]

                with torch.no_grad():
                    # Route to the correct model based on Player ID
                    player_idx = int(agent.split('_')[1])
                    active_model = model_1000 if player_idx % 2 == 0 else model_500

                    # Forward pass
                    dist, _, new_hc = active_model(obs, hc, mask, is_sequence=False)

                    # GREEDY ACTION: Take the mathematically best choice, ignoring illegal masked actions
                    action = torch.argmax(dist.logits, dim=-1).item()

                env.step(action)
                hidden_states[agent] = new_hc

        # Evaluate the victor of the full match
        if env.match_scores[0] > env.match_scores[1]:
            wins_1000 += 1
        elif env.match_scores[1] > env.match_scores[0]:
            wins_500 += 1
        else:
            draws += 1 

        if match % 10 == 0:
            print(f"Match {match}/{total_matches} Complete | Epoch 1000 Wins: {wins_1000} | Epoch 500 Wins: {wins_500} | Draws: {draws}")

    # --- 4. Final Report ---
    print("\n" + "="*30)
    print("      FINAL RESULTS")
    print("="*30)
    print(f"Model Epoch 1000 Winrate: {(wins_1000 / total_matches) * 100:.1f}%")
    print(f"Model Epoch 500 Winrate:  {(wins_500 / total_matches) * 100:.1f}%")
    print(f"Draws: {draws}")
    print("="*30)

if __name__ == "__main__":
    evaluate()