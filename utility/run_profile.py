import os
import gc
import time
import torch
import torch.optim as optim

import train as belot_train
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from eval import evaluate

# Define 4 distinct tactical profiles to benchmark
PROFILES = {
    "1_Baseline": {
        "LR": 3e-4, "ENTROPY_COEF": 0.02, "PPO_ITERS": 4, "CLIP_EPSILON": 0.2, "TARGET_KL": 0.03
    },
    "2_Aggressive_Learning": {
        "LR": 7e-4, "ENTROPY_COEF": 0.01, "PPO_ITERS": 5, "CLIP_EPSILON": 0.3, "TARGET_KL": 0.04
    },
    "3_Conservative_Stable": {
        "LR": 1e-4, "ENTROPY_COEF": 0.03, "PPO_ITERS": 3, "CLIP_EPSILON": 0.15, "TARGET_KL": 0.02
    },
    "4_High_Exploration": {
        "LR": 3e-4, "ENTROPY_COEF": 0.05, "PPO_ITERS": 4, "CLIP_EPSILON": 0.2, "TARGET_KL": 0.03
    }
}

def run_sprint(name, config):
    print(f"\n--- Profiling Config: {name} ---")
    for k, v in config.items():
        setattr(belot_train, k, v)
        print(f"  {k}: {v}")

    # Enforce a short sprint
    belot_train.EPOCHS = 20
    belot_train.EVAL_EVERY = 20
    belot_train.EVAL_GAMES = 40  # Light evaluation for speed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")  # Maximize L4 TF32 performance

    model = RecurrentMAPPOModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=belot_train.LR)
    vec = VectorizedBelot(belot_train.NUM_ENVS)

    start_time = time.time()
    
    # Run the 20-epoch sprint
    for epoch in range(belot_train.EPOCHS):
        episodes = belot_train.collect_rollout(model, vec, belot_train.TARGET_GAMES, device)
        metrics = belot_train.update(model, optimizer, episodes, device)

    # Evaluate at the finish line
    res = evaluate(model, num_games=belot_train.EVAL_GAMES, device=device)
    elapsed = time.time() - start_time
    
    print(f"Finished in {elapsed:.1f}s | Win Rate: {res['win_rate']:.3f} | Point Diff: {res['avg_point_diff']:.2f} | ExpVar: {metrics.get('explained_variance', 0):.3f}")

    # Aggressive VRAM cleanup
    del model, optimizer, vec, episodes
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "win_rate": res["win_rate"],
        "point_diff": res["avg_point_diff"],
        "exp_var": metrics.get("explained_variance", 0),
        "time": elapsed
    }

if __name__ == "__main__":
    global_start = time.time()
    results = {}

    for name, config in PROFILES.items():
        # Check if we are dangerously close to the 10-minute wall (8.5 mins buffer)
        if time.time() - global_start > 510:
            print("\n[!] Approaching 10-minute limit! Skipping remaining profiles.")
            break
        results[name] = run_sprint(name, config)

    print("\n================ PROFILING SUMMARY ================")
    best_profile = None
    best_score = -float("inf")

    for name, res in results.items():
        # We look for a blend of high point differential and healthy Explained Variance
        score = res["point_diff"]
        print(f"Profile {name}: Score={score:.2f} | Win%={res['win_rate']:.2f} | Time={res['time']:.1f}s")
        if score > best_score:
            best_score = score
            best_profile = name

    print("===================================================")
    print(f"🏆 OPTIMAL PROFILE: {best_profile}")
    print("Incorporate these parameters into your final train.py run!")