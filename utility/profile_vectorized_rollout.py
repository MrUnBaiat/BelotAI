import time
import torch
import numpy as np
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from memory import Episode

def zero_state(device):
    return (torch.zeros(1, 1, 512, device=device),
            torch.zeros(1, 1, 512, device=device))

def run_vectorized_bottleneck_audit(num_envs=64, target_games=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Vectorized Bottleneck Audit...")
    print(f"Config: NUM_ENVS={num_envs} | TARGET_GAMES={target_games} | DEVICE={device}\n")
    
    # Initialize vectorized environment and model exactly like train.py
    vec = VectorizedBelot(num_envs)
    model = RecurrentMAPPOModel().to(device)
    model.eval()
    
    # Timing Accumulators
    time_observe = 0.0
    time_inference = 0.0
    time_step = 0.0
    time_overhead = 0.0
    
    # Replicate train.py state tracking dictionaries
    hidden     = {(e, a): zero_state(device) for e in range(num_envs) for a in range(4)}
    active_ep  = {(e, a): Episode()          for e in range(num_envs) for a in range(4)}
    reward_acc = {(e, a): 0.0                for e in range(num_envs) for a in range(4)}
    
    completed = []
    games_done = 0
    total_steps = 0
    
    start_total = time.perf_counter()
    
    while games_done < target_games:
        # 1. Profile Batched Observation Construction
        t0 = time.perf_counter()
        agents, local, glob, masks = vec.observe_active()
        time_observe += (time.perf_counter() - t0)
        
        # 2. Profile Gathering Overhead (Assembling batch tensors)
        t0 = time.perf_counter()
        h_batch = torch.cat([hidden[(e, agents[e])][0] for e in range(num_envs)], dim=1)
        c_batch = torch.cat([hidden[(e, agents[e])][1] for e in range(num_envs)], dim=1)
        
        local_t = torch.from_numpy(local).to(device)
        glob_t  = torch.from_numpy(glob).to(device)
        mask_t  = torch.from_numpy(masks).to(device)
        time_overhead += (time.perf_counter() - t0)
        
        # 3. Profile Batched Model Inference (GPU Matrix Multiplies)
        t0 = time.perf_counter()
        with torch.no_grad():
            dist, value, (new_h, new_c) = model(
                local_t, glob_t, (h_batch, c_batch), mask_t, is_sequence=False
            )
            actions = dist.sample()
            
        actions_np = actions.cpu().numpy()
        values_np  = value.squeeze(-1).cpu().numpy()
        time_inference += (time.perf_counter() - t0)
        
        # 4. Profile Serial Environment Progression & Tracking
        for e in range(num_envs):
            a = agents[e]
            key = (e, a)
            
            # State Routing Overhead
            t0 = time.perf_counter()
            active_ep[key].credit_pending_reward(reward_acc[key])
            reward_acc[key] = 0.0
            active_ep[key].add(
                local[e].copy(), glob[e].copy(), masks[e].copy(),
                int(actions_np[e]), 0.0, float(values_np[e])
            )
            hidden[key] = (new_h[:, e:e + 1, :].contiguous(),
                           new_c[:, e:e + 1, :].contiguous())
            time_overhead += (time.perf_counter() - t0)
            
            # Raw Environment Step Execution
            t0 = time.perf_counter()
            step_rewards, done, info = vec.step_env(e, int(actions_np[e]))
            time_step += (time.perf_counter() - t0)
            
            # Post-Step Bookkeeping & Reset Overhead
            t0 = time.perf_counter()
            for i in range(4):
                reward_acc[(e, i)] += step_rewards[i]
                
            if done:
                for i in range(4):
                    k = (e, i)
                    active_ep[k].credit_pending_reward(reward_acc[k])
                    if len(active_ep[k]) > 0:
                        completed.append(active_ep[k])
                games_done += 1
                
                vec.finish_and_reset(e, info)
                for i in range(4):
                    active_ep[(e, i)] = Episode()
                    reward_acc[(e, i)] = 0.0
                    hidden[(e, i)] = zero_state(device)
                    
            time_overhead += (time.perf_counter() - t0)
            total_steps += 1
            
            if games_done >= target_games:
                break

    end_total = time.perf_counter()
    total_duration = end_total - start_total
    
    print("="*60)
    print("             VECTORIZED BOTTLENECK AUDIT RESULTS            ")
    print("="*60)
    print(f"Total Rollout Duration: {total_duration:.4f} seconds")
    print(f"Total Completed Games:  {games_done}")
    print(f"Total Atomic Steps:     {total_steps} individual transitions")
    print(f"Avg Performance:        {(total_duration / total_steps)*1000:.4f} ms/step")
    print("-"*60)
    
    def print_row(phase_name, duration):
        percentage = (duration / total_duration) * 100
        avg_ms = (duration / total_steps) * 1000
        print(f"{phase_name:<25} | {duration:>8.4f}s | {percentage:>5.1f}% | {avg_ms:>7.4f} ms/step")

    print_row("Observation Build (CPU)", time_observe)
    print_row("Batched Inference (GPU)", time_inference)
    print_row("Raw Env Engines (CPU)", time_step)
    print_row("Overhead & Reset Logic", time_overhead)
    print("="*60)

if __name__ == "__main__":
    run_vectorized_bottleneck_audit()
    
''' Result:
Starting Vectorized Bottleneck Audit...
Config: NUM_ENVS=32 | TARGET_GAMES=128 | DEVICE=cuda

============================================================
             VECTORIZED BOTTLENECK AUDIT RESULTS            
============================================================
Total Rollout Duration: 1.1019 seconds
Total Completed Games:  128
Total Atomic Steps:     4472 individual transitions
Avg Performance:        0.2464 ms/step
------------------------------------------------------------
Observation Build (CPU)   |   0.3790s |  34.4% |  0.0848 ms/step
Batched Inference (GPU)   |   0.5099s |  46.3% |  0.1140 ms/step
Raw Env Engines (CPU)     |   0.0547s |   5.0% |  0.0122 ms/step
Overhead & Reset Logic    |   0.1549s |  14.1% |  0.0346 ms/step
============================================================
'''