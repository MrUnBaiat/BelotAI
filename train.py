import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env_wrapper import BelotAECEnv
from model import RecurrentPPOModel
from memory import MultiAgentMemory

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"Training on device: {device}")
    
    # --- Setup Telemetry and Persistence ---
    run_name = "belot_ppo_v1"
    writer = SummaryWriter(f"runs/{run_name}")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = BelotAECEnv()
    model = RecurrentPPOModel().to(device)
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
    episodes_per_batch = 64
    clip_epsilon = 0.2
    
    for epoch in range(start_epoch, epochs):
        memory.clear_all()
        
        # ==========================================
        # 1. ROLLOUT PHASE: CPU (Zero Transfer Latency)
        # ==========================================
        model.to(cpu_device) # Move model to CPU for fast step-by-step playing
        
        for _ in range(episodes_per_batch):
            env.reset()
            # Initialize hidden states on CPU
            hidden_states = {
                agent: (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256)) 
                for agent in env.possible_agents
            }
            
            for agent in env.agent_iter():
                obs_dict, reward, termination, truncation, info = env.last()
                scaled_reward = reward / 10.0 
                
                if termination or truncation:
                    if len(memory.buffers[agent].rewards) > 0:
                        memory.buffers[agent].rewards[-1] = scaled_reward
                        memory.buffers[agent].dones[-1] = True
                    env.step(None) 
                    continue
                
                # Keep everything on CPU! No .to(device) here.
                obs = torch.tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor(obs_dict["action_mask"], dtype=torch.float32).unsqueeze(0)
                hc = hidden_states[agent]
                
                with torch.no_grad():
                    dist, value, new_hc = model(obs, hc, mask, is_sequence=False)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                
                memory.buffers[agent].store(
                    obs.squeeze(), mask.squeeze(), action.item(), 
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

        # PPO BPTT Loop...
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy = 0
        update_steps = 0

        for _ in range(ppo_iters):
            for agent in env.possible_agents:
                buf = memory.buffers[agent]
                if len(buf.obs) == 0: continue
                
                # NOW we push the massive stacked blocks of data to the GPU all at once
                b_obs = torch.stack(buf.obs).unsqueeze(0).to(device)
                b_masks = torch.stack(buf.masks).unsqueeze(0).to(device)
                b_actions = torch.tensor(buf.actions, device=device)
                b_old_logprobs = torch.tensor(buf.logprobs, device=device)
                b_returns = agent_returns[agent].to(device)
                b_advantages = agent_advantages[agent].to(device)
                
                h_0 = buf.h_states[0].unsqueeze(0).unsqueeze(0).to(device)
                c_0 = buf.c_states[0].unsqueeze(0).unsqueeze(0).to(device)
                curr_hc = (h_0, c_0)
                
                # Execute batched sequence through GPU
                dist, values, _ = model(b_obs, curr_hc, b_masks, is_sequence=True)
                
                values = values.squeeze()
                new_logprobs = dist.log_prob(b_actions)
                entropies = dist.entropy()
                
                # Calculate PPO losses seamlessly across the whole sequence at once
                ratio = torch.exp(new_logprobs - b_old_logprobs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, b_returns)
                entropy_loss = entropies.mean()
                
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
            
            # Log Game Specific Metrics (assuming 1 game = 1 episode for now)
            writer.add_scalar("Scores/Team_Us", env.match_scores[0], epoch)
            writer.add_scalar("Scores/Team_Them", env.match_scores[1], epoch)
            writer.add_scalar("Game/Bolts_Us", env.belot.bolts_by_team[0], epoch)
            writer.add_scalar("Game/Bolts_Them", env.belot.bolts_by_team[1], epoch)

        # --- 5. MODEL PERSISTENCE ---
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
            
            # Keep a floating "latest" pointer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, latest_ckpt)
            
            print(f"Epoch {epoch} | Saved Checkpoint | Team Us: {env.match_scores[0]} | Team Them: {env.match_scores[1]}")

    writer.close()

if __name__ == "__main__":
    train()