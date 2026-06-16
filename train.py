"""
Vectorized MAPPO training for Belot.

Architecture (in-process lockstep vectorization):
  - N independent BelotEnv instances live in one process.
  - Each macro-step gathers the active agent's obs from all N envs, runs ONE
    batched (N, ...) forward pass, then steps every env once.

The three silent killers are handled explicitly in collect_rollout() (tagged
inline): #1 hidden-state routing, #2 buffer contiguity, #3 terminal reward/reset.

Collection runs until TARGET_GAMES complete; in-flight games are discarded so the
GAE stays a clean next_value=0.0 with no truncation bootstrap.
"""

import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel
from memory import Episode, make_minibatch
from eval import evaluate

# ----------------------------- Config -----------------------------
NUM_ENVS           = 64      # stable batch width N for every forward pass
TARGET_GAMES       = 512     # completed games per rollout (bigger batch, less discard waste)
PPO_ITERS          = 4       # optimization epochs over the collected data
MINIBATCH_EPISODES = 256     # ~8 minibatches/epoch; memory is not the constraint here
CLIP_EPSILON       = 0.2
GAMMA              = 0.999   # ~9-step episodes: keep the terminal true-up near-undiscounted
LAM                = 0.95
HIDDEN             = 512
LR                 = 3e-4
VALUE_COEF         = 0.5
ENTROPY_COEF       = 0.04    # heavily-masked action space -> modest entropy bonus
MAX_GRAD_NORM      = 0.5
TARGET_KL          = 0.03    # early-stop guard for PPO iters (set None to disable)
EPOCHS             = 10000
SEED               = 0

EVAL_EVERY         = 25      # epochs between evaluations
EVAL_GAMES         = 200
CHECKPOINT_DIR     = "checkpoints"   # on Colab, point this at /content/drive/MyDrive/...


def zero_state(device):
    return (torch.zeros(1, 1, HIDDEN, device=device),
            torch.zeros(1, 1, HIDDEN, device=device))


def explained_variance(values, returns):
    var_ret = returns.var()
    return float(1.0 - (returns - values).var() / (var_ret + 1e-8))


# ============================================================================
# 1. ROLLOUT PHASE
# ============================================================================
def collect_rollout(model, vec, target_games, device):
    N = vec.num_envs

    hidden     = {(e, a): zero_state(device) for e in range(N) for a in range(4)}  # killer #1
    active_ep  = {(e, a): Episode()          for e in range(N) for a in range(4)}  # killer #2
    reward_acc = {(e, a): 0.0                for e in range(N) for a in range(4)}  # killer #3

    completed = []
    games_done = 0

    while games_done < target_games:
        agents, local, glob, masks = vec.observe_active()

        # killer #1: assemble LSTM state for each env's currently active seat
        h_batch = torch.cat([hidden[(e, agents[e])][0] for e in range(N)], dim=1)
        c_batch = torch.cat([hidden[(e, agents[e])][1] for e in range(N)], dim=1)

        local_t = torch.from_numpy(local).to(device)
        glob_t  = torch.from_numpy(glob).to(device)
        mask_t  = torch.from_numpy(masks).to(device)

        with torch.no_grad():
            dist, value, (new_h, new_c) = model(
                local_t, glob_t, (h_batch, c_batch), mask_t, is_sequence=False
            )
            actions = dist.sample()
            logprobs = dist.log_prob(actions)

        actions_np  = actions.cpu().numpy()
        logprobs_np = logprobs.cpu().numpy()
        values_np   = value.squeeze(-1).cpu().numpy()

        for e in range(N):
            a = agents[e]
            key = (e, a)

            active_ep[key].credit_pending_reward(reward_acc[key])   # killer #3 (retroaction)
            reward_acc[key] = 0.0

            active_ep[key].add(                                      # killer #2 (contiguity)
                local[e].copy(), glob[e].copy(), masks[e].copy(),
                int(actions_np[e]), float(logprobs_np[e]), float(values_np[e]),
            )

            hidden[key] = (new_h[:, e:e + 1, :].contiguous(),       # killer #1 (routing)
                           new_c[:, e:e + 1, :].contiguous())

            step_rewards, done, info = vec.step_env(e, int(actions_np[e]))
            for i in range(4):
                reward_acc[(e, i)] += step_rewards[i]

            if done:
                for i in range(4):                                  # killer #3 (terminal drain)
                    k = (e, i)
                    active_ep[k].credit_pending_reward(reward_acc[k])
                    if len(active_ep[k]) > 0:
                        completed.append(active_ep[k])
                games_done += 1

                vec.finish_and_reset(e, info)                       # auto-reset only this slot
                for i in range(4):
                    active_ep[(e, i)] = Episode()
                    reward_acc[(e, i)] = 0.0
                    hidden[(e, i)] = zero_state(device)

                if games_done >= target_games:
                    break

    return completed


# ============================================================================
# 2. UPDATE PHASE
# ============================================================================
def update(model, optimizer, episodes, device):
    for ep in episodes:
        ep.returns, ep.advantages = ep.compute_gae(GAMMA, LAM)

    # Critic diagnostic from rollout-time predictions (before any weight change).
    all_vals = torch.cat([torch.tensor(ep.values, dtype=torch.float32) for ep in episodes])
    all_rets = torch.cat([ep.returns for ep in episodes])
    ev = explained_variance(all_vals, all_rets)

    all_adv = torch.cat([ep.advantages for ep in episodes])
    mean, std = all_adv.mean(), all_adv.std()
    for ep in episodes:
        ep.advantages = (ep.advantages - mean) / (std + 1e-8)

    actor_acc = critic_acc = entropy_acc = kl_acc = clip_acc = 0.0
    steps = 0
    stop = False

    for _ in range(PPO_ITERS):
        if stop:
            break
        random.shuffle(episodes)
        iter_kl, iter_steps = 0.0, 0

        for start in range(0, len(episodes), MINIBATCH_EPISODES):
            mb = episodes[start:start + MINIBATCH_EPISODES]
            b_obs, b_gobs, b_masks, b_actions, b_old_logprobs, b_adv, b_ret, pad_mask = \
                make_minibatch(mb)

            b_obs = b_obs.to(device);     b_gobs = b_gobs.to(device)
            b_masks = b_masks.to(device); b_actions = b_actions.to(device)
            b_old_logprobs = b_old_logprobs.to(device)
            b_adv = b_adv.to(device);     b_ret = b_ret.to(device)
            pad_mask = pad_mask.to(device)

            B = b_obs.size(0)
            h0 = torch.zeros(1, B, HIDDEN, device=device)
            c0 = torch.zeros(1, B, HIDDEN, device=device)

            dist, values, _ = model(b_obs, b_gobs, (h0, c0), b_masks, is_sequence=True)
            values = values.squeeze(-1)
            new_logprobs = dist.log_prob(b_actions)
            entropies = dist.entropy()

            logratio = new_logprobs - b_old_logprobs
            ratio = torch.exp(logratio)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * b_adv

            valid = pad_mask.sum()
            actor_loss   = -(torch.min(surr1, surr2) * pad_mask).sum() / valid
            critic_loss  = (F.mse_loss(values, b_ret, reduction='none') * pad_mask).sum() / valid
            entropy_loss = (entropies * pad_mask).sum() / valid

            total_loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (((ratio - 1) - logratio) * pad_mask).sum() / valid
                clip_frac = ((torch.abs(ratio - 1.0) > CLIP_EPSILON).float() * pad_mask).sum() / valid

            actor_acc   += actor_loss.item()
            critic_acc  += critic_loss.item()
            entropy_acc += entropy_loss.item()
            kl_acc      += approx_kl.item()
            clip_acc    += clip_frac.item()
            steps += 1
            iter_kl += approx_kl.item(); iter_steps += 1

        # Early-stop PPO iters if the policy has moved too far from the rollout data.
        if TARGET_KL is not None and iter_steps > 0 and (iter_kl / iter_steps) > 1.5 * TARGET_KL:
            stop = True

    if steps == 0:
        return {}
    return {
        "actor": actor_acc / steps,
        "critic": critic_acc / steps,
        "entropy": entropy_acc / steps,
        "approx_kl": kl_acc / steps,
        "clip_frac": clip_acc / steps,
        "explained_variance": ev,
    }


# ============================================================================
# 3. DRIVER
# ============================================================================
def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")  # enable TF32 matmuls on the L4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter("runs/belot_ppo_vectorized")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = RecurrentMAPPOModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    vec = VectorizedBelot(NUM_ENVS)

    start_epoch = 0
    best_metric = -float("inf")
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = ckpt.get('best_metric', -float("inf"))
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh training run.")

    for epoch in range(start_epoch, EPOCHS):
        episodes = collect_rollout(model, vec, TARGET_GAMES, device)
        metrics = update(model, optimizer, episodes, device)

        writer.add_scalar("Loss/Actor",            metrics["actor"],              epoch)
        writer.add_scalar("Loss/Critic",           metrics["critic"],             epoch)
        writer.add_scalar("Loss/Entropy",          metrics["entropy"],            epoch)
        writer.add_scalar("Diag/ApproxKL",         metrics["approx_kl"],          epoch)
        writer.add_scalar("Diag/ClipFraction",     metrics["clip_frac"],          epoch)
        writer.add_scalar("Diag/ExplainedVariance", metrics["explained_variance"], epoch)

        if epoch % EVAL_EVERY == 0:
            res = evaluate(model, num_games=EVAL_GAMES, device=device)
            writer.add_scalar("Eval/WinRateVsRandom",   res["win_rate"],       epoch)
            writer.add_scalar("Eval/PointDiffVsRandom", res["avg_point_diff"], epoch)
            print(f"Epoch {epoch} | win% {res['win_rate']:.3f} | "
                  f"pointdiff {res['avg_point_diff']:.2f} | EV {metrics['explained_variance']:.2f}")

            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
            }
            torch.save(ckpt_data, latest_ckpt)
            if res["avg_point_diff"] > best_metric:
                best_metric = res["avg_point_diff"]
                ckpt_data['best_metric'] = best_metric
                torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    writer.close()


if __name__ == "__main__":
    train()