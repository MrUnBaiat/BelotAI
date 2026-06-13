"""
Vectorized MAPPO training for Belot.

Architecture (in-process lockstep vectorization):
  - N independent BelotEnv instances live in one process.
  - Each macro-step gathers the active agent's obs from all N envs, runs ONE
    batched (N, ...) forward pass, then steps every env once. Inference -- the
    ~80%-of-runtime cost we profiled -- is amortized across N games.

The three silent killers from the design phase are handled explicitly in
collect_rollout() and are tagged inline:
  #1 Hidden-state routing  : per-(env, seat) LSTM state, gathered/scattered by key.
  #2 Buffer contiguity     : each (env, seat) writes only into its own Episode.
  #3 Terminal reward/reset : drain all four seats' final rewards (incl. true-up)
                             atomically at done, THEN auto-reset only that slot.

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

# ----------------------------- Config -----------------------------
NUM_ENVS           = 64     # stable batch width N for every forward pass
TARGET_GAMES       = 128    # completed games gathered per rollout
PPO_ITERS          = 4      # optimization epochs over the collected data
MINIBATCH_EPISODES = 128    # episodes per gradient step (bounds GPU memory)
CLIP_EPSILON       = 0.2
GAMMA              = 0.99
LAM                = 0.95
HIDDEN             = 512
LR                 = 3e-4
EPOCHS             = 10000
ENTROPY_COEF       = 0.05


def zero_state(device):
    return (torch.zeros(1, 1, HIDDEN, device=device),
            torch.zeros(1, 1, HIDDEN, device=device))


# ============================================================================
# 1. ROLLOUT PHASE
# ============================================================================
def collect_rollout(model, vec, target_games, device):
    N = vec.num_envs

    # Per-(env, seat) state. Keying by (env, seat) -- never by batch row -- is what
    # keeps routing/contiguity correct when a batch mixes seats across games.
    hidden     = {(e, a): zero_state(device) for e in range(N) for a in range(4)}  # killer #1
    active_ep  = {(e, a): Episode()          for e in range(N) for a in range(4)}  # killer #2
    reward_acc = {(e, a): 0.0                for e in range(N) for a in range(4)}  # killer #3

    completed = []
    games_done = 0

    while games_done < target_games:
        # ---- GATHER: one active obs per env -> batch of exactly N ----
        agents, local, glob, masks = vec.observe_active()

        # killer #1: assemble the LSTM state for each env's *currently active* seat
        h_batch = torch.cat([hidden[(e, agents[e])][0] for e in range(N)], dim=1)  # (1, N, H)
        c_batch = torch.cat([hidden[(e, agents[e])][1] for e in range(N)], dim=1)

        local_t = torch.from_numpy(local).to(device)
        glob_t  = torch.from_numpy(glob).to(device)
        mask_t  = torch.from_numpy(masks).to(device)

        # ---- ONE batched GPU forward pass ----
        with torch.no_grad():
            dist, value, (new_h, new_c) = model(
                local_t, glob_t, (h_batch, c_batch), mask_t, is_sequence=False
            )
            actions = dist.sample()
            logprobs = dist.log_prob(actions)

        # Pull scalars to host once per macro-step (not once per env).
        actions_np  = actions.cpu().numpy()
        logprobs_np = logprobs.cpu().numpy()
        values_np   = value.squeeze(-1).cpu().numpy()

        # ---- PER-ENV: store, route, step, accumulate, (maybe) terminate ----
        for e in range(N):
            a = agents[e]
            key = (e, a)

            # killer #3 (retroaction half): credit reward earned since this seat's
            # last action to its previous transition, then clear its accumulator.
            active_ep[key].credit_pending_reward(reward_acc[key])
            reward_acc[key] = 0.0

            # killer #2: append into THIS (env, seat) timeline only -> contiguous.
            active_ep[key].add(
                local[e].copy(), glob[e].copy(), masks[e].copy(),
                int(actions_np[e]), float(logprobs_np[e]), float(values_np[e]),
            )

            # killer #1: scatter the new hidden state back to its (env, seat) slot.
            hidden[key] = (new_h[:, e:e + 1, :].contiguous(),
                           new_c[:, e:e + 1, :].contiguous())

            step_rewards, done, info = vec.step_env(e, int(actions_np[e]))

            # This step's team reward is owed to every seat until that seat next acts.
            for i in range(4):
                reward_acc[(e, i)] += step_rewards[i]

            if done:
                # killer #3 (terminal half): step_rewards already carries the
                # end-of-hand true-up, so a single pass closes all four timelines.
                for i in range(4):
                    k = (e, i)
                    active_ep[k].credit_pending_reward(reward_acc[k])
                    if len(active_ep[k]) > 0:
                        completed.append(active_ep[k])
                games_done += 1

                # Auto-reset ONLY this slot; batch width stays N.
                vec.finish_and_reset(e, info)
                for i in range(4):
                    active_ep[(e, i)] = Episode()
                    reward_acc[(e, i)] = 0.0
                    hidden[(e, i)] = zero_state(device)

                if games_done >= target_games:
                    break  # budget met -> discard everything still in flight

    return completed


# ============================================================================
# 2. UPDATE PHASE
# ============================================================================
def update(model, optimizer, episodes, device):
    # GAE per complete episode (bootstrap 0.0), then one global advantage norm.
    for ep in episodes:
        ep.returns, ep.advantages = ep.compute_gae(GAMMA, LAM)

    all_adv = torch.cat([ep.advantages for ep in episodes])
    mean, std = all_adv.mean(), all_adv.std()
    for ep in episodes:
        ep.advantages = (ep.advantages - mean) / (std + 1e-8)

    actor_acc = critic_acc = entropy_acc = 0.0
    steps = 0

    for _ in range(PPO_ITERS):
        random.shuffle(episodes)
        for start in range(0, len(episodes), MINIBATCH_EPISODES):
            mb = episodes[start:start + MINIBATCH_EPISODES]

            b_obs, b_gobs, b_masks, b_actions, b_old_logprobs, b_adv, b_ret, pad_mask = \
                make_minibatch(mb)

            b_obs   = b_obs.to(device);   b_gobs = b_gobs.to(device)
            b_masks = b_masks.to(device); b_actions = b_actions.to(device)
            b_old_logprobs = b_old_logprobs.to(device)
            b_adv = b_adv.to(device);     b_ret = b_ret.to(device)
            pad_mask = pad_mask.to(device)

            # Fresh zero hidden state per episode -> LSTM re-rolled from t=0,
            # matching how each episode was generated during rollout.
            B = b_obs.size(0)
            h0 = torch.zeros(1, B, HIDDEN, device=device)
            c0 = torch.zeros(1, B, HIDDEN, device=device)

            dist, values, _ = model(b_obs, b_gobs, (h0, c0), b_masks, is_sequence=True)
            values = values.squeeze(-1)
            new_logprobs = dist.log_prob(b_actions)
            entropies = dist.entropy()

            ratio = torch.exp(new_logprobs - b_old_logprobs)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * b_adv

            valid = pad_mask.sum()
            actor_loss   = -(torch.min(surr1, surr2) * pad_mask).sum() / valid
            critic_loss  = (F.mse_loss(values, b_ret, reduction='none') * pad_mask).sum() / valid
            entropy_loss = (entropies * pad_mask).sum() / valid

            total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            actor_acc   += actor_loss.item()
            critic_acc  += critic_loss.item()
            entropy_acc += entropy_loss.item()
            steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0
    return actor_acc / steps, critic_acc / steps, entropy_acc / steps


# ============================================================================
# 3. DRIVER
# ============================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    run_name = "belot_ppo_vectorized"
    writer = SummaryWriter(f"runs/{run_name}")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = RecurrentMAPPOModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    vec = VectorizedBelot(NUM_ENVS)

    start_epoch = 0
    latest_ckpt = os.path.join(checkpoint_dir, "latest_model.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting fresh training run.")

    for epoch in range(start_epoch, EPOCHS):
        # Model stays on `device` for both phases: the rollout forward pass is now
        # batched (N wide), so there is no longer any reason to ping-pong to CPU.
        episodes = collect_rollout(model, vec, TARGET_GAMES, device)
        actor_loss, critic_loss, entropy_loss = update(model, optimizer, episodes, device)

        # ---- Telemetry: training diagnostics only ----
        writer.add_scalar("Loss/Actor",   actor_loss,   epoch)
        writer.add_scalar("Loss/Critic",  critic_loss,  epoch)
        writer.add_scalar("Loss/Entropy", entropy_loss, epoch)

        if epoch % 50 == 0:
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(ckpt_data, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
            torch.save(ckpt_data, latest_ckpt)
            print(f"Epoch {epoch} | Saved | "
                  f"Actor {actor_loss:.4f} | Critic {critic_loss:.4f} | Entropy {entropy_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    train()