import os
import torch
from env_wrapper import BelotAECEnv
from model import RecurrentMAPPOModel
from evaluation.modelCombinedCriticActor.model_ppo import RecurrentPPOModel  # The new file from Step 1

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # --- 1. Load the Models ---
    # Team 0: MAPPO (CTDE + Dense Rewards)
    model_mappo = RecurrentMAPPOModel().to(device)
    # Team 1: PPO (Standard + End-of-Episode Rewards)
    model_ppo = RecurrentPPOModel().to(device)

    # Note: Replace these paths with your actual checkpoint paths
    ckpt_mappo_path = "model_epoch_2600.pt" 
    ckpt_ppo_path = "model_epoch_9950.pt"

    if not os.path.exists(ckpt_mappo_path) or not os.path.exists(ckpt_ppo_path):
        raise FileNotFoundError("Could not find one or both checkpoints. Please check the paths.")

    model_mappo.load_state_dict(torch.load(ckpt_mappo_path, map_location=device)['model_state_dict'])
    model_ppo.load_state_dict(torch.load(ckpt_ppo_path, map_location=device)['model_state_dict'])

    # Set to evaluation mode
    model_mappo.eval()
    model_ppo.eval()

    env = BelotAECEnv()

    # --- 2. Setup Match Tracking ---
    total_matches = 500
    wins_mappo = 0
    wins_ppo = 0
    draws = 0

    print(f"Starting {total_matches} full matches (First to 101 Points)...")
    print("Team 0 (Players 0 & 2): MAPPO Architecture")
    print("Team 1 (Players 1 & 3): Standard PPO Architecture\n")

    # --- 3. The Evaluation Loop ---
    for match in range(1, total_matches + 1):
        # Force a hard reset for the global game variables at the start of a match
        env.reset()
        env.match_scores = [0, 0] 
        env.belot.bolts_by_team = [0, 0] 

        # Keep playing hands until a team hits 101 Game Points
        while max(env.match_scores) < 101:
            env.reset() 
            
            # Re-initialize hidden states for the new hand. 
            # Both models use 512 hidden dimensions, so this structure works for both.
            hidden_states = {
                agent: (torch.zeros(1, 1, 512, device=device), torch.zeros(1, 1, 512, device=device))
                for agent in env.possible_agents
            }

            for agent in env.agent_iter():
                obs_dict, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    env.step(None) # Dead step required by PettingZoo API
                    continue

                # Prepare common inputs
                obs = torch.tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.tensor(obs_dict["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
                hc = hidden_states[agent]

                player_idx = int(agent.split('_')[1])
                
                with torch.no_grad():
                    # --- ARCHITECTURE ROUTING ---
                    if player_idx % 2 == 0:
                        # Team 1: PPO only requires the Local Observation
                        dist, _, new_hc = model_ppo(obs, hc, mask, is_sequence=False)
                    else:
                        # Team 0: MAPPO requires the Global Observation
                        g_obs = torch.tensor(obs_dict["global_observation"], dtype=torch.float32, device=device).unsqueeze(0)
                        dist, _, new_hc = model_mappo(obs, g_obs, hc, mask, is_sequence=False)

                    # GREEDY ACTION: Take the mathematically best choice
                    action = torch.argmax(dist.logits, dim=-1).item()

                env.step(action)
                hidden_states[agent] = new_hc

        # Evaluate the victor of the full match
        if env.match_scores[0] > env.match_scores[1]:
            wins_mappo += 1
        elif env.match_scores[1] > env.match_scores[0]:
            wins_ppo += 1
        else:
            draws += 1 

        if match % 10 == 0:
            print(f"Match {match}/{total_matches} Complete | MAPPO Wins: {wins_mappo} | PPO Wins: {wins_ppo} | Draws: {draws}")

    # --- 4. Final Report ---
    print("\n" + "="*35)
    print("          FINAL RESULTS")
    print("="*35)
    print(f"MAPPO (Team 0) Winrate: {(wins_mappo / total_matches) * 100:.1f}%")
    print(f"PPO   (Team 1) Winrate: {(wins_ppo / total_matches) * 100:.1f}%")
    print(f"Draws: {draws}")
    print("="*35)

if __name__ == "__main__":
    evaluate()