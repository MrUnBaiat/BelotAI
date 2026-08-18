"""
Vectorized MAPPO training for Belot.

Architecture (in-process lockstep vectorization):
  - N independent BelotEnv instances live in one process.
  - Each macro-step gathers the active agent's obs from all N envs, runs ONE
    batched (N, ...) forward pass, then steps every env once.

The three silent killers are handled explicitly in collect_rollout() (tagged
inline): #1 hidden-state routing, #2 buffer contiguity, #3 terminal reward/reset.

v2 changes (all validated by the audit scripts):
  * BOUNDARY FLUSH -- envs persist across rollouts while episode buffers do not,
    so any hand still in flight at a rollout boundary used to get a terminal
    true-up computed against dense rewards it never recorded (measured bias:
    mean 0.19, max 0.62 on a +-1 scale). flush_in_flight() now plays those hands
    out WITHOUT storing them, so every stored Episode starts at a fresh deal.
  * OPPONENT MIXING -- pure self-play sat at an equilibrium (latest vs best
    measured 50.0% over 150 matches). Each env now draws an opponent for the odd
    team at every game: current policy / uniform-random / a frozen snapshot.
    In mixed envs only the even seats are learning seats and only their episodes
    are stored; rewards still flow through every seat, so returns stay exact.
  * DIAGNOSTICS -- PPO iterations actually completed under the KL early stop,
    entropy split by phase, and ratio/advantage sanity, all logged.
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
from eval import evaluate_matches

# ----------------------------- Config -----------------------------
NUM_ENVS           = 64      # stable batch width N for every forward pass
TARGET_GAMES       = 512     # completed games per rollout
PPO_ITERS          = 4       # optimization epochs over the collected data
MINIBATCH_EPISODES = 128     # ~2x the gradient steps/epoch vs v1; KL stayed ~0.003
CLIP_EPSILON       = 0.2
GAMMA              = 0.999   # ~9-step episodes: keep the terminal true-up near-undiscounted
LAM                = 0.95
HIDDEN             = 512
VALUE_COEF         = 0.5
MAX_GRAD_NORM      = 0.5
TARGET_KL          = 0.02    # early-stop guard for PPO iters (set None to disable)
EPOCHS             = 10000
SEED               = 0

# Annealed schedules (measured entropy was pinned at the bonus-imposed floor).
LR_START           = 3e-4
LR_END             = 1e-4
LR_ANNEAL_EPOCHS   = 3000
ENTROPY_START      = 0.04
ENTROPY_END        = 0.005
ENTROPY_ANNEAL_EPOCHS = 2000

# Opponent mixing for the ODD team (seats 1 & 3), sampled per env per game.
OPP_SELF           = 0.70    # current policy in every seat (classic self-play)
OPP_RANDOM         = 0.15    # uniform-random legal opponent
OPP_FROZEN         = 0.15    # a frozen snapshot from the pool
FROZEN_INIT        = "checkpoints/best_model.pt"   # seeds the training pool (None to skip)
# The EVAL reference must be immutable: best_model.pt is overwritten whenever the
# model improves, so using it as the yardstick silently redefines the yardstick on
# every restart and makes "vs reference" incomparable across runs. This file is
# created once from FROZEN_INIT and then never written again.
EVAL_REFERENCE     = "checkpoints/reference_model.pt"
FROZEN_POOL_MAX    = 3
SNAPSHOT_EVERY     = 100     # epochs between adding the live policy to the pool

EVAL_EVERY         = 50      # epochs between evaluations (fewer, bigger evals)
EVAL_MATCHES       = 250     # full matches to 101, per opponent (~2750 hands)
SELECTION_METRIC   = "hand_diff_vs_reference_v2"   # bump when the metric changes
CHECKPOINT_DIR     = "checkpoints"   # on Colab, point this at /content/drive/MyDrive/...


def zero_state(device, n=1):
    return (torch.zeros(1, n, HIDDEN, device=device),
            torch.zeros(1, n, HIDDEN, device=device))


def explained_variance(values, returns):
    var_ret = returns.var()
    return float(1.0 - (returns - values).var() / (var_ret + 1e-8))


def anneal(start, end, epoch, horizon):
    f = min(max(epoch / float(horizon), 0.0), 1.0)
    return start + f * (end - start)


# ============================================================================
# 0. BOUNDARY FLUSH
# ============================================================================
def flush_in_flight(model, vec, device):
    """
    Play every mid-hand env to the end of its current hand WITHOUT storing
    anything. Guarantees each rollout starts with all envs at a fresh deal, so
    no stored Episode can straddle a rollout boundary.

    The discarded actions come from the current policy with one hidden state per
    env (not per seat) -- the data is thrown away, so this only perturbs the
    distribution of reset states, never a training target.
    """
    N = vec.num_envs
    if all(vec.fresh):
        return 0
    hid = {e: zero_state(device) for e in range(N)}
    steps = 0
    while not all(vec.fresh):
        _, local, glob, masks = vec.observe_active()
        h = torch.cat([hid[e][0] for e in range(N)], dim=1)
        c = torch.cat([hid[e][1] for e in range(N)], dim=1)
        with torch.no_grad():
            dist, _, (nh, nc) = model(
                torch.from_numpy(local).to(device),
                torch.from_numpy(glob).to(device),
                (h, c),
                torch.from_numpy(masks).to(device),
                is_sequence=False,
            )
            acts = dist.sample().cpu().numpy()
        for e in range(N):
            if vec.fresh[e]:
                continue
            hid[e] = (nh[:, e:e + 1, :].contiguous(), nc[:, e:e + 1, :].contiguous())
            _, done, info = vec.step_env(e, int(acts[e]))
            if done:
                vec.finish_and_reset(e, info)
                hid[e] = zero_state(device)
        steps += 1
    return steps


def _sample_opponent(frozen_pool):
    r = random.random()
    if r < OPP_SELF or not frozen_pool:
        return "self", None
    if r < OPP_SELF + OPP_RANDOM:
        return "random", None
    return "frozen", random.randrange(len(frozen_pool))


# ============================================================================
# 1. ROLLOUT PHASE
# ============================================================================
def collect_rollout(model, vec, target_games, device, frozen_pool=None):
    frozen_pool = frozen_pool or []
    N = vec.num_envs

    flush_steps = flush_in_flight(model, vec, device)   # every env now at a fresh deal

    hidden     = {(e, a): zero_state(device) for e in range(N) for a in range(4)}  # killer #1
    hidden_opp = {(e, a): zero_state(device) for e in range(N) for a in range(4)}  # frozen net
    active_ep  = {(e, a): Episode()          for e in range(N) for a in range(4)}  # killer #2
    reward_acc = {(e, a): 0.0                for e in range(N) for a in range(4)}  # killer #3

    opp_kind = [None] * N          # opponent driving seats 1 & 3 in this env's game
    opp_idx  = [None] * N
    for e in range(N):
        opp_kind[e], opp_idx[e] = _sample_opponent(frozen_pool)

    def is_learning(e, a):
        """Seats whose transitions are stored and trained on."""
        return opp_kind[e] == "self" or a % 2 == 0

    completed = []
    games_done = 0
    n_by_kind = {"self": 0, "random": 0, "frozen": 0}

    while games_done < target_games:
        agents, local, glob, masks = vec.observe_active()

        drive = ["cur" if is_learning(e, agents[e]) else opp_kind[e] for e in range(N)]
        cur_rows = [e for e in range(N) if drive[e] == "cur"]

        actions_np = np.zeros(N, dtype=np.int64)
        logprobs_np = np.zeros(N, dtype=np.float32)
        values_np = np.zeros(N, dtype=np.float32)

        # --- current policy: one batched pass over the rows it drives ---
        if cur_rows:
            idx = torch.as_tensor(cur_rows, device=device)
            # killer #1: assemble LSTM state for each env's currently active seat
            h_batch = torch.cat([hidden[(e, agents[e])][0] for e in cur_rows], dim=1)
            c_batch = torch.cat([hidden[(e, agents[e])][1] for e in cur_rows], dim=1)
            local_t = torch.from_numpy(local).to(device).index_select(0, idx)
            glob_t  = torch.from_numpy(glob).to(device).index_select(0, idx)
            mask_t  = torch.from_numpy(masks).to(device).index_select(0, idx)

            with torch.no_grad():
                dist, value, (new_h, new_c) = model(
                    local_t, glob_t, (h_batch, c_batch), mask_t, is_sequence=False
                )
                acts = dist.sample()
                lps = dist.log_prob(acts)

            a_np, l_np = acts.cpu().numpy(), lps.cpu().numpy()
            v_np = value.squeeze(-1).cpu().numpy()
            for j, e in enumerate(cur_rows):
                actions_np[e], logprobs_np[e], values_np[e] = a_np[j], l_np[j], v_np[j]
                hidden[(e, agents[e])] = (new_h[:, j:j + 1, :].contiguous(),   # killer #1
                                          new_c[:, j:j + 1, :].contiguous())

        # --- frozen snapshots: one batched pass per distinct pool member ---
        for k, fnet in enumerate(frozen_pool):
            rows = [e for e in range(N) if drive[e] == "frozen" and opp_idx[e] == k]
            if not rows:
                continue
            idx = torch.as_tensor(rows, device=device)
            h_b = torch.cat([hidden_opp[(e, agents[e])][0] for e in rows], dim=1)
            c_b = torch.cat([hidden_opp[(e, agents[e])][1] for e in rows], dim=1)
            with torch.no_grad():
                dist, _, (nh, nc) = fnet(
                    torch.from_numpy(local).to(device).index_select(0, idx),
                    torch.from_numpy(glob).to(device).index_select(0, idx),
                    (h_b, c_b),
                    torch.from_numpy(masks).to(device).index_select(0, idx),
                    is_sequence=False,
                )
                acts = dist.sample().cpu().numpy()
            for j, e in enumerate(rows):
                actions_np[e] = acts[j]
                hidden_opp[(e, agents[e])] = (nh[:, j:j + 1, :].contiguous(),
                                              nc[:, j:j + 1, :].contiguous())

        # --- uniform-random opponent ---
        for e in range(N):
            if drive[e] == "random":
                legal = np.flatnonzero(masks[e])
                actions_np[e] = int(np.random.choice(legal))

        for e in range(N):
            a = agents[e]
            key = (e, a)

            if drive[e] == "cur" and is_learning(e, a):
                active_ep[key].credit_pending_reward(reward_acc[key])  # killer #3 (retroaction)
                reward_acc[key] = 0.0
                active_ep[key].add(                                     # killer #2 (contiguity)
                    local[e].copy(), glob[e].copy(), masks[e].copy(),
                    int(actions_np[e]), float(logprobs_np[e]), float(values_np[e]),
                )

            step_rewards, done, info = vec.step_env(e, int(actions_np[e]))
            for i in range(4):
                reward_acc[(e, i)] += step_rewards[i]

            if done:
                for i in range(4):                                  # killer #3 (terminal drain)
                    k = (e, i)
                    active_ep[k].credit_pending_reward(reward_acc[k])
                    if is_learning(e, i) and len(active_ep[k]) > 0:
                        completed.append(active_ep[k])
                games_done += 1
                n_by_kind[opp_kind[e]] += 1

                vec.finish_and_reset(e, info)                       # auto-reset only this slot
                opp_kind[e], opp_idx[e] = _sample_opponent(frozen_pool)
                for i in range(4):
                    active_ep[(e, i)] = Episode()
                    reward_acc[(e, i)] = 0.0
                    hidden[(e, i)] = zero_state(device)
                    hidden_opp[(e, i)] = zero_state(device)

                if games_done >= target_games:
                    break

    info_out = {"flush_steps": flush_steps, "games_by_opponent": n_by_kind}
    return completed, info_out


# ============================================================================
# 2. UPDATE PHASE
# ============================================================================
def update(model, optimizer, episodes, device, entropy_coef=ENTROPY_START):
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
    ent_bid_acc = ent_play_acc = 0.0
    bid_steps = play_steps = 0
    steps = 0
    iters_completed = 0
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

            total_loss = actor_loss + VALUE_COEF * critic_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (((ratio - 1) - logratio) * pad_mask).sum() / valid
                clip_frac = ((torch.abs(ratio - 1.0) > CLIP_EPSILON).float() * pad_mask).sum() / valid
                # phase split: a bidding step is the only kind with legal actions >= 32
                bid_mask = (b_masks[..., 32:].sum(-1) > 0).float() * pad_mask
                play_mask = pad_mask - bid_mask
                nb, npl = bid_mask.sum(), play_mask.sum()
                if nb > 0:
                    ent_bid_acc += float((entropies * bid_mask).sum() / nb); bid_steps += 1
                if npl > 0:
                    ent_play_acc += float((entropies * play_mask).sum() / npl); play_steps += 1

            actor_acc   += actor_loss.item()
            critic_acc  += critic_loss.item()
            entropy_acc += entropy_loss.item()
            kl_acc      += approx_kl.item()
            clip_acc    += clip_frac.item()
            steps += 1
            iter_kl += approx_kl.item(); iter_steps += 1

        iters_completed += 1
        # Early-stop PPO iters if the policy has moved too far from the rollout data.
        if TARGET_KL is not None and iter_steps > 0 and (iter_kl / iter_steps) > 1.5 * TARGET_KL:
            stop = True

    if steps == 0:
        return {}
    return {
        "actor": actor_acc / steps,
        "critic": critic_acc / steps,
        "entropy": entropy_acc / steps,
        "entropy_bidding": ent_bid_acc / max(bid_steps, 1),
        "entropy_playing": ent_play_acc / max(play_steps, 1),
        "approx_kl": kl_acc / steps,
        "clip_frac": clip_acc / steps,
        "explained_variance": ev,
        "iters_completed": iters_completed,
        "grad_steps": steps,
        "episodes": len(episodes),
    }


# ============================================================================
# 3. DRIVER
# ============================================================================
def load_frozen(path, device):
    net = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
    net.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def snapshot(model, device):
    net = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
    net.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")  # enable TF32 matmuls on the L4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter("runs/belot_ppo_v2")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_START)
    vec = VectorizedBelot(NUM_ENVS)

    start_epoch = 0
    best_metric = -float("inf")
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # A best_metric written under a DIFFERENT metric definition is on a
        # different scale and would silently block every future save (v1 stored
        # "point diff vs random" ~ +5.7; v2 compares "hand diff vs reference"
        # ~ +0.3, so nothing ever beat it). Only inherit it if the tag matches.
        if ckpt.get('selection_metric') == SELECTION_METRIC:
            best_metric = ckpt.get('best_metric', -float("inf"))
        else:
            best_metric = -float("inf")
            print(f"  selection metric changed ({ckpt.get('selection_metric')} -> "
                  f"{SELECTION_METRIC}); resetting best_metric")
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh training run.")

    frozen_pool = []
    if FROZEN_INIT and os.path.exists(FROZEN_INIT):
        frozen_pool.append(load_frozen(FROZEN_INIT, device))
        print(f"Frozen opponent pool seeded from {FROZEN_INIT}")

    # Pin the eval reference once, then reuse that exact file forever.
    eval_reference = None
    if EVAL_REFERENCE:
        if not os.path.exists(EVAL_REFERENCE) and FROZEN_INIT and os.path.exists(FROZEN_INIT):
            import shutil
            shutil.copyfile(FROZEN_INIT, EVAL_REFERENCE)
            print(f"Pinned eval reference: copied {FROZEN_INIT} -> {EVAL_REFERENCE}")
        if os.path.exists(EVAL_REFERENCE):
            eval_reference = load_frozen(EVAL_REFERENCE, device)
            print(f"Eval reference loaded from {EVAL_REFERENCE} (immutable)")

    for epoch in range(start_epoch, EPOCHS):
        lr = anneal(LR_START, LR_END, epoch, LR_ANNEAL_EPOCHS)
        for g in optimizer.param_groups:
            g['lr'] = lr
        ent_coef = anneal(ENTROPY_START, ENTROPY_END, epoch, ENTROPY_ANNEAL_EPOCHS)

        episodes, roll_info = collect_rollout(model, vec, TARGET_GAMES, device, frozen_pool)
        metrics = update(model, optimizer, episodes, device, entropy_coef=ent_coef)
        if not metrics:
            continue

        writer.add_scalar("Loss/Actor",             metrics["actor"],               epoch)
        writer.add_scalar("Loss/Critic",            metrics["critic"],              epoch)
        writer.add_scalar("Loss/Entropy",           metrics["entropy"],             epoch)
        writer.add_scalar("Entropy/Bidding",        metrics["entropy_bidding"],     epoch)
        writer.add_scalar("Entropy/Playing",        metrics["entropy_playing"],     epoch)
        writer.add_scalar("Diag/ApproxKL",          metrics["approx_kl"],           epoch)
        writer.add_scalar("Diag/ClipFraction",      metrics["clip_frac"],           epoch)
        writer.add_scalar("Diag/ExplainedVariance", metrics["explained_variance"],  epoch)
        writer.add_scalar("Diag/PPOItersCompleted", metrics["iters_completed"],     epoch)
        writer.add_scalar("Diag/GradSteps",         metrics["grad_steps"],          epoch)
        writer.add_scalar("Diag/EpisodesCollected", metrics["episodes"],            epoch)
        writer.add_scalar("Diag/FlushSteps",        roll_info["flush_steps"],       epoch)
        writer.add_scalar("Sched/LR",               lr,                             epoch)
        writer.add_scalar("Sched/EntropyCoef",      ent_coef,                       epoch)
        for k, v in roll_info["games_by_opponent"].items():
            writer.add_scalar(f"Rollout/GamesVs_{k}", v, epoch)

        if SNAPSHOT_EVERY and epoch > 0 and epoch % SNAPSHOT_EVERY == 0:
            frozen_pool.append(snapshot(model, device))
            if len(frozen_pool) > FROZEN_POOL_MAX:
                # keep the eval reference (index 0) and drop the oldest snapshot
                frozen_pool.pop(1 if len(frozen_pool) > 1 else 0)
            print(f"Epoch {epoch}: added snapshot to frozen pool (size {len(frozen_pool)})")

        if epoch % EVAL_EVERY == 0:
            # vs random saturated at 0.97-1.00 match win, so it carries no signal;
            # the fixed greedy heuristic is an absolute yardstick that does not
            # saturate and stays comparable across every run you will ever do.
            r_heur = evaluate_matches(model, num_matches=EVAL_MATCHES, device=device,
                                      opponent="heuristic")
            writer.add_scalar("Eval/MatchWinVsHeuristic", r_heur["match_win_rate"], epoch)
            writer.add_scalar("Eval/HandDiffVsHeuristic", r_heur["avg_hand_diff"],  epoch)
            line = (f"Epoch {epoch} | vs heuristic: match% {r_heur['match_win_rate']:.3f} "
                    f"handdiff {r_heur['avg_hand_diff']:+.2f} "
                    f"+-{r_heur['hand_diff_ci95']:.2f}")

            metric = r_heur["avg_hand_diff"]
            if eval_reference is not None:
                r_ref = evaluate_matches(model, num_matches=EVAL_MATCHES, device=device,
                                         opponent="model", frozen=eval_reference)
                writer.add_scalar("Eval/MatchWinVsReference", r_ref["match_win_rate"], epoch)
                writer.add_scalar("Eval/HandDiffVsReference", r_ref["avg_hand_diff"],  epoch)
                line += (f" | vs reference: match% {r_ref['match_win_rate']:.3f} "
                         f"handdiff {r_ref['avg_hand_diff']:+.2f} "
                         f"+-{r_ref['hand_diff_ci95']:.2f}")
                metric = r_ref["avg_hand_diff"]     # progress vs a FIXED strong opponent
            line += (f" | EV {metrics['explained_variance']:.2f} "
                     f"| iters {metrics['iters_completed']}/{PPO_ITERS} "
                     f"| KL {metrics['approx_kl']:.4f}")
            print(line, flush=True)

            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'selection_metric': SELECTION_METRIC,
            }
            torch.save(ckpt_data, latest_ckpt)
            if metric > best_metric:
                best_metric = metric
                ckpt_data['best_metric'] = best_metric
                torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    writer.close()


if __name__ == "__main__":
    train()