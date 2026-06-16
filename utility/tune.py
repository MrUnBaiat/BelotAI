import os
import gc
import torch
import optuna
import numpy as np
import torch.optim as optim

# Import your original training module
import train as belot_train
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from eval import evaluate

def objective(trial):
    """
    Optuna objective function. Dynamically overrides the global configs 
    in train.py, runs a shortened training loop, and returns the best eval metric.
    """
    # 1. Define the Hyperparameter Search Space
    belot_train.LR = trial.suggest_float("LR", 1e-5, 1e-3, log=True)
    belot_train.ENTROPY_COEF = trial.suggest_float("ENTROPY_COEF", 0.005, 0.05, log=True)
    belot_train.VALUE_COEF = trial.suggest_categorical("VALUE_COEF", [0.5, 1.0])
    belot_train.CLIP_EPSILON = trial.suggest_categorical("CLIP_EPSILON", [0.1, 0.2, 0.3])
    belot_train.PPO_ITERS = trial.suggest_int("PPO_ITERS", 2, 6)
    belot_train.LAM = trial.suggest_float("LAM", 0.90, 0.99)
    belot_train.TARGET_KL = trial.suggest_float("TARGET_KL", 0.015, 0.05)
    
    # 2. Modify configs for a faster profiling run
    # We don't want to run 10,000 epochs per trial. 150-200 is usually 
    # enough to see if the policy is learning or collapsing.
    belot_train.EPOCHS = 150
    belot_train.EVAL_EVERY = 25
    belot_train.EVAL_GAMES = 100 # Reduced eval load for speed

    # 3. Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high") # Leverage L4 TF32 cores

    model = RecurrentMAPPOModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=belot_train.LR)
    vec = VectorizedBelot(belot_train.NUM_ENVS)

    best_eval_metric = -float("inf")

    # 4. Trial Training Loop
    for epoch in range(belot_train.EPOCHS):
        # Use the logic directly from your train.py
        episodes = belot_train.collect_rollout(model, vec, belot_train.TARGET_GAMES, device)
        metrics = belot_train.update(model, optimizer, episodes, device)

        # 5. Periodic Evaluation & Pruning
        if epoch > 0 and epoch % belot_train.EVAL_EVERY == 0:
            res = evaluate(model, num_games=belot_train.EVAL_GAMES, device=device)
            current_metric = res["avg_point_diff"]

            if current_metric > best_eval_metric:
                best_eval_metric = current_metric

            # Report intermediate metric to Optuna
            trial.report(current_metric, epoch)
            
            # Prune (early-stop) unpromising trials to save L4 GPU time
            if trial.should_prune():
                cleanup(model, optimizer, vec, episodes)
                raise optuna.exceptions.TrialPruned()

    cleanup(model, optimizer, vec, episodes)
    return best_eval_metric

def cleanup(model, optimizer, vec, episodes=None):
    """Prevents PyTorch memory leaks between trials on the L4 GPU."""
    del model
    del optimizer
    del vec
    if episodes:
        del episodes
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print("Starting Hyperparameter Tuning...")
    
    # Create an Optuna study aiming to maximize the avg_point_diff
    study = optuna.create_study(
        study_name="belot_mappo_tuning", 
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50) # Don't prune before epoch 50
    )
    
    # Run for 30 trials (Adjust based on your Colab time limit)
    study.optimize(objective, n_trials=30)

    # Output results
    print("\n==================================================")
    print("Tuning Complete!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Point Differential: {study.best_trial.value:.2f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("==================================================")