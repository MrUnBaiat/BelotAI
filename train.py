import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env_wrapper import BelotAECEnv
from model import RecurrentMAPPOModel
from memory import MultiAgentMemory

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"Training on device: {device}")
    
    run_name = "belot_ppo_v2_513dim"
    writer = SummaryWriter(f"runs/{run_name}")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = BelotAECEnv()
    model = RecurrentMAPPOModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    memory = MultiAgentMemory(env.possible_agents)
    
    # --- Load Checkpoint (If exists) ---
    start_epoch = 0
    latest_ckpt = os.path.join(checkpoint_dir, "latest_model.pt")
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting fresh training run.")

    epochs = 10000
    ppo_iters = 4
    episodes_per_batch = 128
    clip_epsilon = 0.2
    
    for epoch in range(start_epoch, epochs):
        memory.clear_all()
        
        # ==========================================
        # 1. ROLLOUT PHASE: CPU (Zero Transfer Latency)
        # ==========================================
        model.to(cpu_device) # Move model to CPU for fast step-by-step playing
        
        for _ in range(episodes_per_batch):
            env.reset()
            hidden_states = {
                agent: (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512)) 
                for agent in env.possible_agents
            }
            
            for agent in env.agent_iter():
                obs_dict, reward, termination, truncation, info = env.last()
                
                # REWARD RETROACTION: I did not understand this, but I think it is needed
                # At time `t`, env.last() provides the accumulated reward derived from the action 
                # taken at `t-1`. We assign this back to the transition generated in the previous step.
                if len(memory.buffers[agent].rewards) > 0:
                    memory.buffers[agent].rewards[-1] = reward # Why not +=?
                
                if termination or truncation:
                    if len(memory.buffers[agent].rewards) > 0:
                        memory.buffers[agent].dones[-1] = True
                    env.step(None) 
                    continue
                
                # Keep everything on CPU! No .to(device) here.
                obs = torch.tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
                g_obs = torch.tensor(obs_dict["global_observation"], dtype=torch.float32).unsqueeze(0) 
                mask = torch.tensor(obs_dict["action_mask"], dtype=torch.float32).unsqueeze(0)
                hc = hidden_states[agent]
                
                with torch.no_grad():
                    dist, value, new_hc = model(obs, g_obs, hc, mask, is_sequence=False)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                
                # Store transition at `t`. Reward is stored as 0.0 but gets overwritten
                # on the agent's next turn with the actual result of this action.
                memory.buffers[agent].store(
                    obs.squeeze(), g_obs.squeeze(), mask.squeeze(), action.item(), 
                    logprob.item(), 0.0, value.item(), False, 
                    hc[0].squeeze(), hc[1].squeeze()
                )
                
                env.step(action.item())
                hidden_states[agent] = new_hc

        # ==========================================
        # 2. UPDATE PHASE: GPU (Massive Parallelism)
        # ==========================================
        model.to(device) # Move model back to GPU for heavy training
        
        # Calculate GAE (This stays on CPU since it's just basic math lists)
        agent_returns = {}
        agent_advantages = {}
        for agent in env.possible_agents:
            buf = memory.buffers[agent]
            if len(buf.rewards) == 0: continue
            ret, adv = buf.compute_gae(next_value=0.0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            agent_returns[agent] = ret
            agent_advantages[agent] = adv

        epoch_actor_loss, epoch_critic_loss, epoch_entropy, update_steps = 0, 0, 0, 0

        for _ in range(ppo_iters):
            for agent in env.possible_agents:
                buf = memory.buffers[agent]
                if len(buf.obs) == 0: continue
                
                # Get properly batched and padded sequences
                b_obs, b_gobs, b_masks, b_actions, b_old_logprobs, b_advantages, b_returns, pad_mask = buf.get_padded_batch(
                    agent_advantages[agent], agent_returns[agent]
                )
                
                # Move everything to GPU
                b_obs = b_obs.to(device)
                b_gobs = b_gobs.to(device) # Move Global
                b_masks = b_masks.to(device)
                b_actions = b_actions.to(device)
                b_old_logprobs = b_old_logprobs.to(device)
                b_returns = b_returns.to(device)
                b_advantages = b_advantages.to(device)
                pad_mask = pad_mask.to(device)
                
                # 2. Initialize FRESH hidden states for this specific batch of episodes
                batch_size = b_obs.size(0)
                # Updated sequence initial states for 512 dimensions
                h_0 = torch.zeros(1, batch_size, 512, device=device)
                c_0 = torch.zeros(1, batch_size, 512, device=device)
                curr_hc = (h_0, c_0)
                
                # 3. Execute batched sequence through GPU
                dist, values, _ = model(b_obs, b_gobs, curr_hc, b_masks, is_sequence=True)
                
                values = values.squeeze(-1) 
                new_logprobs = dist.log_prob(b_actions)
                entropies = dist.entropy()
                
                # 4. Calculate PPO losses
                ratio = torch.exp(new_logprobs - b_old_logprobs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advantages
                
                # 5. MASK THE LOSSES: Multiply by pad_mask to ignore padded 0s, then average over valid steps only
                valid_steps = pad_mask.sum()
                
                actor_loss = -(torch.min(surr1, surr2) * pad_mask).sum() / valid_steps
                
                # Use reduction='none' so we can mask the MSE loss per-element before summing
                critic_loss = (F.mse_loss(values, b_returns, reduction='none') * pad_mask).sum() / valid_steps
                
                entropy_loss = (entropies * pad_mask).sum() / valid_steps
                
                total_loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                # Accumulate for telemetry
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy_loss.item()
                update_steps += 1

        # --- 4. TELEMETRY LOGGING ---
        if update_steps > 0:
            writer.add_scalar("Loss/Actor", epoch_actor_loss / update_steps, epoch)
            writer.add_scalar("Loss/Critic", epoch_critic_loss / update_steps, epoch)
            writer.add_scalar("Loss/Entropy", epoch_entropy / update_steps, epoch)
            writer.add_scalar("Scores/Team_Us", env.match_scores[0], epoch)
            writer.add_scalar("Scores/Team_Them", env.match_scores[1], epoch)
            writer.add_scalar("Game/Bolts_Us", env.belot.bolts_by_team[0], epoch)
            writer.add_scalar("Game/Bolts_Them", env.belot.bolts_by_team[1], epoch)

        # --- 5. MODEL PERSISTENCE ---
        if epoch % 50 == 0:
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(ckpt_data, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
            torch.save(ckpt_data, latest_ckpt)
            print(f"Epoch {epoch} | Saved Checkpoint | Team Us: {env.match_scores[0]} | Team Them: {env.match_scores[1]}")

    writer.close()

if __name__ == "__main__":
    train()