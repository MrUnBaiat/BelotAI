import os
import torch
import numpy as np
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from evaluation.modelCombinedCriticActor.model_ppo import RecurrentPPOModel

def evaluate_vectorized(total_matches=500, num_envs=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device} using Vectorized Environment ({num_envs} parallel envs)")

    # --- 1. Load the Models ---
    model_mappo = RecurrentMAPPOModel().to(device)
    model_ppo = RecurrentPPOModel().to(device)

    # Update paths to your local weights
    ckpt_mappo_path = "model_epoch_1100.pt"
    ckpt_ppo_path = "model_epoch_9450.pt"

    if not os.path.exists(ckpt_mappo_path) or not os.path.exists(ckpt_ppo_path):
        raise FileNotFoundError("Could not find one or both checkpoints. Please verify paths.")

    model_mappo.load_state_dict(torch.load(ckpt_mappo_path, map_location=device)['model_state_dict'])
    model_ppo.load_state_dict(torch.load(ckpt_ppo_path, map_location=device)['model_state_dict'])

    model_mappo.eval()
    model_ppo.eval()

    # --- 2. Setup Vectorized Environment & Tracking ---
    vec_env = VectorizedBelot(num_envs=num_envs)

    # Tensor tracking for LSTMs: (num_envs, num_players, hidden_dim)
    hx = torch.zeros(num_envs, 4, 512, device=device)
    cx = torch.zeros(num_envs, 4, 512, device=device)

    wins_mappo = 0
    wins_ppo = 0
    draws = 0
    matches_played = 0

    print(f"Starting {total_matches} parallel match evaluations...")
    print("Team 0 (Players 0 & 2): MAPPO (Vectorized Model)")
    print("Team 1 (Players 1 & 3): Standard PPO Model\n")

    # --- 3. Evaluation Loop ---
    while matches_played < total_matches:
        agents, local, glob, masks = vec_env.observe_active()

        # Route environments dynamically based on the active player's team
        team0_envs = [e for e, a in enumerate(agents) if a % 2 == 0]
        team1_envs = [e for e, a in enumerate(agents) if a % 2 != 0]

        actions = np.zeros(num_envs, dtype=np.int32)

        # Team 0: MAPPO Forward Pass (Batched)
        if len(team0_envs) > 0:
            obs_t0 = torch.tensor(local[team0_envs], dtype=torch.float32, device=device)
            glob_t0 = torch.tensor(glob[team0_envs], dtype=torch.float32, device=device)
            mask_t0 = torch.tensor(masks[team0_envs], dtype=torch.float32, device=device)
            
            p_ids_t0 = [agents[e] for e in team0_envs]
            h_t0 = hx[team0_envs, p_ids_t0].unsqueeze(0)  # (1, K, 512)
            c_t0 = cx[team0_envs, p_ids_t0].unsqueeze(0)  # (1, K, 512)

            with torch.no_grad():
                dist, _, (new_h, new_c) = model_mappo(obs_t0, glob_t0, (h_t0, c_t0), mask_t0, is_sequence=False)
                act_t0 = torch.argmax(dist.logits, dim=-1).cpu().numpy()

            actions[team0_envs] = act_t0
            hx[team0_envs, p_ids_t0] = new_h.squeeze(0)
            cx[team0_envs, p_ids_t0] = new_c.squeeze(0)

        # Team 1: PPO Forward Pass (Batched)
        if len(team1_envs) > 0:
            obs_t1 = torch.tensor(local[team1_envs], dtype=torch.float32, device=device)
            mask_t1 = torch.tensor(masks[team1_envs], dtype=torch.float32, device=device)
            
            p_ids_t1 = [agents[e] for e in team1_envs]
            h_t1 = hx[team1_envs, p_ids_t1].unsqueeze(0)  # (1, K, 512)
            c_t1 = cx[team1_envs, p_ids_t1].unsqueeze(0)  # (1, K, 512)

            with torch.no_grad():
                dist, _, (new_h, new_c) = model_ppo(obs_t1, (h_t1, c_t1), mask_t1, is_sequence=False)
                act_t1 = torch.argmax(dist.logits, dim=-1).cpu().numpy()

            actions[team1_envs] = act_t1
            hx[team1_envs, p_ids_t1] = new_h.squeeze(0)
            cx[team1_envs, p_ids_t1] = new_c.squeeze(0)

        # Step individual environments sequentially inside the vector
        for e in range(num_envs):
            step_rewards, done, info = vec_env.step_env(e, actions[e])
            if done:
                gp = info.get("game_points", [0, 0, 0, 0])
                final_score_0 = vec_env.match_scores[e][0] + gp[0]
                final_score_1 = vec_env.match_scores[e][1] + gp[1]

                # Check if the cumulative match score crosses the 101 barrier
                if final_score_0 >= 101 or final_score_1 >= 101:
                    if final_score_0 > final_score_1:
                        wins_mappo += 1
                    elif final_score_1 > final_score_0:
                        wins_ppo += 1
                    else:
                        draws += 1
                    
                    matches_played += 1
                    if matches_played % 10 == 0 or matches_played == total_matches:
                        print(f"Match {matches_played}/{total_matches} Complete | MAPPO Wins: {wins_mappo} | PPO Wins: {wins_ppo} | Draws: {draws}")
                    
                    if matches_played >= total_matches:
                        break

                # Clean up and reset hand lifecycle + zero out recurrent memory for this env
                vec_env.finish_and_reset(e, info)
                hx[e] = 0.0
                cx[e] = 0.0

    # --- 4. Final Report ---
    print("\n" + "="*35)
    print("          FINAL RESULTS")
    print("="*35)
    print(f"MAPPO (Team 0) Winrate: {(wins_mappo / total_matches) * 100:.1f}%")
    print(f"PPO   (Team 1) Winrate: {(wins_ppo / total_matches) * 100:.1f}%")
    print(f"Draws: {draws}")
    print("="*35)

if __name__ == "__main__":
    evaluate_vectorized(total_matches=500, num_envs=32)