import time
import torch
import numpy as np
from env_wrapper import BelotAECEnv
from model import RecurrentMAPPOModel

def run_bottleneck_audit(num_episodes=128):
    print(f"Starting bottleneck audit over {num_episodes} episodes on CPU...")
    
    # Initialize components exactly like the rollout phase in train.py
    env = BelotAECEnv()
    model = RecurrentMAPPOModel()
    model.eval() # Ensure eval mode for inference
    
    # Accumulators for timing (in seconds)
    time_observe = 0.0
    time_inference = 0.0
    time_step = 0.0
    time_overhead = 0.0
    
    total_steps = 0
    start_total = time.perf_counter()
    
    for ep in range(num_episodes):
        t0 = time.perf_counter()
        env.reset()
        time_overhead += (time.perf_counter() - t0)
        
        hidden_states = {
            agent: (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512)) 
            for agent in env.possible_agents
        }
        
        for agent in env.agent_iter():
            # 1. Profile Observation Construction (env.last() invokes observe())
            t0 = time.perf_counter()
            obs_dict, reward, termination, truncation, info = env.last()
            time_observe += (time.perf_counter() - t0)
            
            if termination or truncation:
                t0 = time.perf_counter()
                env.step(None)
                time_step += (time.perf_counter() - t0)
                continue
                
            # 2. Profile Overhead (Tensor packing/unpacking)
            t0 = time.perf_counter()
            obs = torch.tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
            g_obs = torch.tensor(obs_dict["global_observation"], dtype=torch.float32).unsqueeze(0) 
            mask = torch.tensor(obs_dict["action_mask"], dtype=torch.float32).unsqueeze(0)
            hc = hidden_states[agent]
            time_overhead += (time.perf_counter() - t0)
            
            # 3. Profile Model Inference
            t0 = time.perf_counter()
            with torch.no_grad():
                dist, value, new_hc = model(obs, g_obs, hc, mask, is_sequence=False)
                action = dist.sample()
            time_inference += (time.perf_counter() - t0)
            
            # 4. Profile Raw Environment Step Logic
            t0 = time.perf_counter()
            env.step(action.item())
            hidden_states[agent] = new_hc
            time_step += (time.perf_counter() - t0)
            
            total_steps += 1

    end_total = time.perf_counter()
    total_duration = end_total - start_total
    
    # Accounted time across tracked segments
    accounted_time = time_observe + time_inference + time_step + time_overhead
    
    print("\n" + "="*50)
    print("                BOTTLENECK AUDIT RESULTS            ")
    print("="*50)
    print(f"Total Rollout Time:   {total_duration:.4f} seconds")
    print(f"Total Actions Taken:  {total_steps} steps")
    print(f"Avg Time Per Step:    {(total_duration / total_steps)*1000:.4f} ms")
    print("-"*50)
    
    # Helper to print formatting rows
    def print_row(phase_name, duration):
        percentage = (duration / total_duration) * 100
        avg_ms = (duration / total_steps) * 1000
        print(f"{phase_name:<20} | {duration:>8.4f}s | {percentage:>5.1f}% | {avg_ms:>7.4f} ms/step")

    print_row("Observation (observe)", time_observe)
    print_row("Inference (model)", time_inference)
    print_row("Env Engine (step)", time_step)
    print_row("Overhead / Resets", time_overhead)
    print("="*50)
    
    # Diagnostics summary
    if time_inference > time_observe:
        print("\n[AUDIT PASS] Inference dominates. Vectorizing the forward pass WILL yield substantial gains.")
    else:
        print("\n[AUDIT WARNING] Observation building dominates. Your vectorization gains will be significantly capped.")
        print("Recommendation: Vectorize the network *only* if you plan to optimize or JIT the belief-matrix loop next.")

if __name__ == "__main__":
    run_bottleneck_audit()
    
''' Result:
Starting bottleneck audit over 128 episodes on CPU...

==================================================
                BOTTLENECK AUDIT RESULTS            
==================================================
Total Rollout Time:   9.6258 seconds
Total Actions Taken:  4310 steps
Avg Time Per Step:    2.2334 ms
--------------------------------------------------
Observation (observe) |   0.8509s |   8.8% |  0.1974 ms/step
Inference (model)    |   8.3219s |  86.5% |  1.9308 ms/step
Env Engine (step)    |   0.1714s |   1.8% |  0.0398 ms/step
Overhead / Resets    |   0.2635s |   2.7% |  0.0611 ms/step
==================================================

[AUDIT PASS] Inference dominates. Vectorizing the forward pass WILL yield substantial gains.
'''