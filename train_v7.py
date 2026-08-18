"""
Belot MAPPO -- AUDIT v4 training configuration.  READY TO RUN ON GPU.

    python train_v4.py

This is train.py with five changes, each traceable to an executed measurement in
the v4 audit. Nothing here is speculative tuning; every constant that moved has a
number behind it. See AUDIT_V4_FINDINGS.md for the full evidence.

--------------------------------------------------------------------------------
C1  EPISODES PER OPTIMIZER STEP        (the primary fix)
--------------------------------------------------------------------------------
MEASURED (v4_01, v4_04, v4_05): at epoch 2400 the actor gradient computed from a
128-episode minibatch is pure sampling noise. Cosine between DISJOINT 128-episode
gradients = -0.0005 +- 0.0099. Two independent signatures confirm it: with n=13
chunks the magnitude ratio |g_chunk|^2/|G_full|^2 was exactly 13.0 (pure noise
predicts exactly n) and cosine-to-mean was 0.2752 (pure noise predicts 1/sqrt(13)
= 0.2774). The gradient noise scale B_simple sits in the 1e4-2e5 episode range;
training was stepping at 128.

FIX: one optimizer step now spans the WHOLE rollout, accumulated over
MICROBATCH_EPISODES-sized forward/backward chunks so peak memory is unchanged,
and the rollout itself is 4x bigger. ~7,000 episodes per step instead of 128.

C1b, AND IT IS NOT OPTIONAL: the learning rate must be scaled with the batch.
A pilot at 2048 games/rollout with the ORIGINAL learning rate measured
approx_kl = 0.0000 per optimizer step -- 4 clean steps per epoch move the policy
essentially not at all, so the run would have gone flat for a trivial reason
(no parameter motion) rather than because the SNR diagnosis is wrong. See
LR_BATCH_SCALE below.

--------------------------------------------------------------------------------
C2  WEIGHT AVERAGING (EMA)
--------------------------------------------------------------------------------
MEASURED (v4_09): see AUDIT_V4_FINDINGS.md section 3 for the noise-ball result.
A random walk near a local optimum degrades performance because performance is
locally concave there, so the iterates orbit a better centre. Averaging the
weights recovers the centre at zero data cost. EMA is evaluated and checkpointed
alongside the live model; selection uses whichever is better.
Set EMA_DECAY = 0 to disable.

--------------------------------------------------------------------------------
C3  SELECTION METRIC  (defect D2)
--------------------------------------------------------------------------------
MEASURED (v4_08): over epochs 1800-2400 the vs-reference score rose significantly
(+0.073 [+0.042, +0.112] per 100 epochs) while the vs-heuristic score did not move
at all (+0.021 [-0.082, +0.117]). The old code selected best_model.pt on
vs-reference -- a score against a checkpoint sitting in its own training pool. It
was selecting for opponent-specific exploitation. Selection now uses the
vs-heuristic scalar, per AUDIT_HANDOFF section 8.4. vs-reference is still logged,
never selected on.

--------------------------------------------------------------------------------
C4  THE POOL AND THE YARDSTICK ARE NO LONGER DESCENDANTS OF THE MODEL  (defect D1)
--------------------------------------------------------------------------------
VERIFIED (v4_08): FROZEN_INIT pointed at best_model.pt, which this trainer
OVERWRITES, and reference_model.pt did not exist -- so the next run would have
created its "immutable yardstick" by copying the very model it was measuring.
EVAL_REFERENCE is now a file the trainer never writes, and it seeds the pool.

--------------------------------------------------------------------------------
C5  AN EMPTY FROZEN POOL NO LONGER DISABLES THE RANDOM OPPONENT  (defect D3)
--------------------------------------------------------------------------------
MEASURED (v4_08): with an empty pool, _sample_opponent returned "self" 100.0% of
the time over 20,000 draws -- so a fresh run with no reference checkpoint was
pure self-play until the first snapshot at epoch 100, with 0% random opponent
instead of the configured 15%.

--------------------------------------------------------------------------------
HYPOTHESIS AND DECISION RULE FOR THIS RUN
--------------------------------------------------------------------------------
HYPOTHESIS: the plateau at +1.9 to +2.4 pts/hand is a signal-to-noise failure --
a stochastic equilibrium set by (learning rate x gradient noise) / batch size --
and not a capability, representation or opponent-diversity limit.

PREDICTION: with ~55x more episodes per optimizer step, Eval/HandDiffVsHeuristic
should clear +3.0 within 300 epochs of resuming from the epoch-2400 checkpoint.

DECISION RULE, evaluated at epoch 2700 (300 epochs, ~6 evals at EVAL_EVERY=50):
  * hand_diff vs heuristic >= +3.0  -> CONFIRMED. Keep going; the fix works.
  * +2.4 to +3.0                    -> partial. Raise TARGET_GAMES again (B_simple
                                       may be at the top of the measured range) and
                                       run another 300.
  * <= +2.4 (i.e. within CI of the  -> REFUTED. The batch was not the binding
    +2.37 baseline) after 300 ep       constraint. Do NOT keep scaling the batch;
                                       move to the representation/objective items
                                       in AUDIT_V4_FINDINGS.md section 6.
Each eval is 250 matches, CI ~ +-0.40, so judge the TREND over the 6 evals with a
bootstrap slope, not any single point. That is what v4_02_history.py does.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from eval import evaluate_matches
from memory import Episode, make_minibatch
from model import RecurrentMAPPOModel
from perturbed_heuristic import perturbed_heuristic_action
from vec_env import VectorizedBelot

# ----------------------------- Config -----------------------------
NUM_ENVS           = 64
TARGET_GAMES       = 2048    # C1: 4x the rollout (was 512) -> ~7,000 episodes
PPO_ITERS          = 4
# C1: MINIBATCH_EPISODES is episodes per OPTIMIZER STEP; None = the whole rollout.
# MICROBATCH_EPISODES is only the forward/backward chunk, i.e. a memory knob.
MINIBATCH_EPISODES  = None
MICROBATCH_EPISODES = 128
CLIP_EPSILON       = 0.2
GAMMA              = 0.999
LAM                = 0.95
HIDDEN             = 512
VALUE_COEF         = 0.5
MAX_GRAD_NORM      = 0.5     # v4_01: joint clip measured INACTIVE (0.43 < 0.5), left alone
TARGET_KL          = 0.02
EPOCHS             = 10000
SEED               = 0

# C2: EMA of the weights. DISABLED BY DEFAULT -- v4_09 measured it and it did not
# help: EMA - final iterate = -0.047 +- 0.191 pts/hand over 3,000 paired deals, and
# on matches to 101 start/final/EMA were +2.365/+2.308/+2.414, all within CI. The
# noise-ball hypothesis it was meant to exploit is NOT supported.
# Caveat kept for whoever revisits it: with decay 0.98 over only 30 epochs, 0.98^30
# = 54.5% of that EMA was still the initial checkpoint, so the test rules out a
# large effect, not a small one. Set to 0.99 to re-enable if you run it long enough
# for the initialisation to wash out.
EMA_DECAY          = 0.0

LR_START           = 3e-4
LR_END             = 1e-4
LR_ANNEAL_EPOCHS   = 3000
# C1b: LINEAR SCALING RULE -- required, not optional tuning. C1 replaces 56 noisy
# optimizer steps per epoch with ~4 clean ones, and a pilot at unchanged LR
# measured approx_kl = 0.0000 per step: the batch got clean but the policy stopped
# moving, which would produce a flat result for a trivial reason. Optimal LR scales
# as B/(B+B_simple); going 128 -> ~7000 episodes/step with B_simple ~ 2e4 justifies
# ~40x. 8x is the conservative setting. WATCH Diag/ApproxKL: it should sit around
# 0.005-0.02. If it pins at the TARGET_KL guard and iters_completed drops below 4,
# lower this; if it stays near 0.000, raise it.
LR_BATCH_SCALE     = 8.0
# ---------------------------------------------------------------------------
# C6  RE-HEAT
# ---------------------------------------------------------------------------
# The entropy coefficient reached its 0.005 floor around epoch 2000 and the agent
# has trained at minimum exploration ever since. Measured consequences (v4_03):
# pi(sampled action) has median 1.000 in bidding and 0.989 in card play -- the
# policy is very nearly deterministic, so the distribution of hands it trains card
# play on is narrow and self-selected. A plateau observed at minimum temperature is
# partly a plateau OF THE SCHEDULE, which is cheap to test and does not need a
# restart. Re-heat and hold: no anneal back down inside this run, so the effect is
# a single clean variable.
ENTROPY_START      = 0.025
ENTROPY_END        = 0.025
ENTROPY_ANNEAL_EPOCHS = 1

# ---------------------------------------------------------------------------
# C7  A REAL LEAGUE
# ---------------------------------------------------------------------------
# What the shipped config actually produced (v4_08 + v4_00): FROZEN_INIT pointed at
# best_model.pt, which the trainer overwrites, so the "frozen 15%" opponent was a
# BYTE-IDENTICAL copy of the live model (SHA db65371052c1f0fe for both files). The
# effective mix was therefore 85% self-play + 15% uniform random -- and uniform
# random is saturated at this strength, so it teaches almost nothing. That is
# essentially pure self-play, and the cycling it predicts was measured directly:
# vs-reference rose +0.073 [+0.042, +0.112] per 100 epochs while absolute strength
# stayed flat at +0.021 [-0.082, +0.117].
#
# The fix has three parts:
#   1. Snapshots are taken 3x more often and the pool holds 8 instead of 3, so it
#      spans ~400 epochs of history instead of ~300 of near-identical clones.
#   2. Half of all games are played against the pool, and pool members are chosen
#      UNIFORMLY, so old (weaker, behaviourally different) members keep getting
#      played rather than being crowded out by the newest near-clone.
#   3. The pool is seeded from an immutable file the trainer never writes.
#
# DELIBERATELY NOT DONE: the greedy heuristic is NOT added to the training pool,
# even though it would be a strong and cheap sparring partner. It is the absolute
# yardstick, and putting a yardstick in the training pool is exactly defect D-B --
# it would make Eval/HandDiffVsHeuristic exploitable and destroy the only cheap
# absolute measurement this project has. PIMC (pimc.py) is held out for the same
# reason and is the yardstick above the model.
# ---------------------------------------------------------------------------
# C9  A NON-DESCENDANT SCRIPTED OPPONENT IN THE POOL   (the v7 change)
# ---------------------------------------------------------------------------
# MEASURED (v4_13, epoch-2848 weights, 128 chunks per regime, episode-equalised):
#
#   regime      |g|^2/|G|^2   pure-noise   deviation   disjoint cosine      B_simple
#   self             123.21          128        3.7%   +0.00029+-0.00107     437,395
#   random            92.67          128       27.6%   +0.00265+-0.00091      48,248
#   heuristic         77.64          128       39.3%   +0.00430+-0.00111      29,660
#
# `self` is indistinguishable from pure noise on BOTH statistics. `heuristic` has
# a clearly resolvable gradient with a B_simple 15x smaller. So the policy is
# stationary under the SELF-PLAY objective specifically, not globally -- a real
# improvement direction exists, and a training mix that is ~85% self-play cannot
# see it. That is the single mechanism explaining v6 section 6.1: every previous
# intervention perturbed the fixed point, the policy tracked it, and the self-play
# objective pulled it back.
#
# So a genuinely non-descendant opponent goes into the pool. It is the PERTURBED
# heuristic, never the exact one -- pimc.py delegates BIDDING to _heuristic_action,
# so training against the exact heuristic would leak bidding exploitation straight
# into the PIMC yardstick. See perturbed_heuristic.py.
#
# Note B_simple = 29,660 even in the good regime, versus ~7,000 episodes per
# optimizer step under C1 -- still ~4x short. Keep the large batch; do NOT revert
# to 128.
OPP_SELF           = 0.40
OPP_RANDOM         = 0.05
OPP_FROZEN         = 0.30
OPP_SCRIPT         = 0.25    # perturbed heuristic -- the non-descendant opponent
# C4: an immutable file this trainer NEVER writes. Create it once, by hand:
#     cp checkpoints/best_model.pt checkpoints/reference_model.pt
EVAL_REFERENCE     = "checkpoints/reference_model.pt"
FROZEN_INIT        = EVAL_REFERENCE
FROZEN_POOL_MAX    = 8       # C7: was 3
SNAPSHOT_EVERY     = 33      # C7: was 100 -- pool spans ~264 epochs of real history

EVAL_EVERY         = 50
EVAL_MATCHES       = 250
# C8: PIMC yardstick. Search runs at every card, so a hand costs ~40x a heuristic
# hand -- keep the match count small and the interval long.
PIMC_EVAL_EVERY    = 200
PIMC_EVAL_MATCHES  = 40
PIMC_D             = 16
SELECTION_METRIC   = "hand_diff_vs_heuristic_v4"   # C3
CHECKPOINT_DIR     = "checkpoints"


def zero_state(device, n=1):
    return (torch.zeros(1, n, HIDDEN, device=device),
            torch.zeros(1, n, HIDDEN, device=device))


def explained_variance(values, returns):
    return float(1.0 - (returns - values).var() / (returns.var() + 1e-8))


def anneal(start, end, epoch, horizon):
    f = min(max(epoch / float(horizon), 0.0), 1.0)
    return start + f * (end - start)


def flush_in_flight(model, vec, device):
    """Play every mid-hand env to the end of its hand WITHOUT storing, so no
    stored Episode straddles a rollout boundary (AUDIT_HANDOFF 4.2)."""
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
                torch.from_numpy(local).to(device), torch.from_numpy(glob).to(device),
                (h, c), torch.from_numpy(masks).to(device), is_sequence=False)
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
    """C5 / D3: the random opponent must not depend on the frozen pool. The old
    form short-circuited to 'self' whenever the pool was empty, measured at
    100.0% self over 20,000 draws."""
    r = random.random()
    if r < OPP_SELF:
        return "self", None
    if r < OPP_SELF + OPP_RANDOM:
        return "random", None
    if r < OPP_SELF + OPP_RANDOM + OPP_SCRIPT:
        return "script", None                      # C9: perturbed heuristic
    if frozen_pool:
        return "frozen", random.randrange(len(frozen_pool))
    return "self", None


def collect_rollout(model, vec, target_games, device, frozen_pool=None,
                    target_episodes=None):
    """C10: EQUALISE EPISODES, NOT GAMES.

    A self-play game stores 4 episodes (all seats learn); a mixed game stores only
    the 2 even seats. So raising the mixed share mechanically cuts the training
    data per rollout -- v5 section 5.2 measured the league arm at 5,730
    episodes/step against C1's 6,956 at identical game counts, an 18% deficit that
    confounded that comparison. v7 raises the mixed share further (to 60%), which
    would make it worse.

    Passing `target_episodes` stops on the episode count instead, so arms are
    comparable on the quantity that actually feeds the optimizer. `target_games`
    then acts purely as a safety cap.
    """
    frozen_pool = frozen_pool or []
    N = vec.num_envs
    flush_steps = flush_in_flight(model, vec, device)

    hidden     = {(e, a): zero_state(device) for e in range(N) for a in range(4)}
    hidden_opp = {(e, a): zero_state(device) for e in range(N) for a in range(4)}
    active_ep  = {(e, a): Episode()          for e in range(N) for a in range(4)}
    reward_acc = {(e, a): 0.0                for e in range(N) for a in range(4)}

    opp_kind, opp_idx = [None] * N, [None] * N
    for e in range(N):
        opp_kind[e], opp_idx[e] = _sample_opponent(frozen_pool)

    def is_learning(e, a):
        return opp_kind[e] == "self" or a % 2 == 0

    completed, games_done = [], 0
    n_by_kind = {"self": 0, "random": 0, "frozen": 0, "script": 0}

    def budget_reached():
        return (len(completed) >= target_episodes if target_episodes
                else games_done >= target_games)

    while not budget_reached() and games_done < target_games:
        agents, local, glob, masks = vec.observe_active()
        drive = ["cur" if is_learning(e, agents[e]) else opp_kind[e] for e in range(N)]
        cur_rows = [e for e in range(N) if drive[e] == "cur"]

        actions_np = np.zeros(N, dtype=np.int64)
        logprobs_np = np.zeros(N, dtype=np.float32)
        values_np = np.zeros(N, dtype=np.float32)

        if cur_rows:
            idx = torch.as_tensor(cur_rows, device=device)
            h_b = torch.cat([hidden[(e, agents[e])][0] for e in cur_rows], dim=1)
            c_b = torch.cat([hidden[(e, agents[e])][1] for e in cur_rows], dim=1)
            with torch.no_grad():
                dist, value, (nh, nc) = model(
                    torch.from_numpy(local).to(device).index_select(0, idx),
                    torch.from_numpy(glob).to(device).index_select(0, idx),
                    (h_b, c_b),
                    torch.from_numpy(masks).to(device).index_select(0, idx),
                    is_sequence=False)
                acts = dist.sample()
                lps = dist.log_prob(acts)
            a_np, l_np = acts.cpu().numpy(), lps.cpu().numpy()
            v_np = value.squeeze(-1).cpu().numpy()
            for j, e in enumerate(cur_rows):
                actions_np[e], logprobs_np[e], values_np[e] = a_np[j], l_np[j], v_np[j]
                hidden[(e, agents[e])] = (nh[:, j:j + 1, :].contiguous(),
                                          nc[:, j:j + 1, :].contiguous())

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
                    is_sequence=False)
                acts = dist.sample().cpu().numpy()
            for j, e in enumerate(rows):
                actions_np[e] = acts[j]
                hidden_opp[(e, agents[e])] = (nh[:, j:j + 1, :].contiguous(),
                                              nc[:, j:j + 1, :].contiguous())

        for e in range(N):
            if drive[e] == "random":
                actions_np[e] = int(np.random.choice(np.flatnonzero(masks[e])))
            elif drive[e] == "script":                      # C9
                actions_np[e] = perturbed_heuristic_action(vec.envs[e])

        for e in range(N):
            a = agents[e]
            key = (e, a)
            if drive[e] == "cur" and is_learning(e, a):
                active_ep[key].credit_pending_reward(reward_acc[key])
                reward_acc[key] = 0.0
                active_ep[key].add(local[e].copy(), glob[e].copy(), masks[e].copy(),
                                   int(actions_np[e]), float(logprobs_np[e]),
                                   float(values_np[e]))
            step_rewards, done, info = vec.step_env(e, int(actions_np[e]))
            for i in range(4):
                reward_acc[(e, i)] += step_rewards[i]

            if done:
                for i in range(4):
                    k = (e, i)
                    active_ep[k].credit_pending_reward(reward_acc[k])
                    if is_learning(e, i) and len(active_ep[k]) > 0:
                        completed.append(active_ep[k])
                games_done += 1
                n_by_kind[opp_kind[e]] += 1
                vec.finish_and_reset(e, info)
                opp_kind[e], opp_idx[e] = _sample_opponent(frozen_pool)
                for i in range(4):
                    active_ep[(e, i)] = Episode()
                    reward_acc[(e, i)] = 0.0
                    hidden[(e, i)] = zero_state(device)
                    hidden_opp[(e, i)] = zero_state(device)
                if budget_reached() or games_done >= target_games:
                    break

    return completed, {"flush_steps": flush_steps, "games_by_opponent": n_by_kind}


def update(model, optimizer, episodes, device, entropy_coef=ENTROPY_START):
    for ep in episodes:
        ep.returns, ep.advantages = ep.compute_gae(GAMMA, LAM)

    all_vals = torch.cat([torch.tensor(ep.values, dtype=torch.float32) for ep in episodes])
    ev = explained_variance(all_vals, torch.cat([ep.returns for ep in episodes]))

    all_adv = torch.cat([ep.advantages for ep in episodes])
    mean, std = all_adv.mean(), all_adv.std()
    for ep in episodes:
        ep.advantages = (ep.advantages - mean) / (std + 1e-8)

    step_size = MINIBATCH_EPISODES or len(episodes)      # C1
    actor_acc = critic_acc = entropy_acc = kl_acc = clip_acc = 0.0
    ent_bid_acc = ent_play_acc = 0.0
    bid_steps = play_steps = steps = iters_completed = 0
    stop = False

    for _ in range(PPO_ITERS):
        if stop:
            break
        random.shuffle(episodes)
        iter_kl, iter_steps = 0.0, 0

        for start in range(0, len(episodes), step_size):
            mb = episodes[start:start + step_size]
            mb_valid = float(sum(len(ep) for ep in mb))
            if mb_valid == 0:
                continue
            optimizer.zero_grad()
            a_s = c_s = e_s = kl_s = cf_s = 0.0
            eb_s = ep_s = nb_s = npl_s = 0.0

            # C1: accumulate over micro-chunks. Every term is normalised by the
            # WHOLE step's timestep count, so this is EXACTLY equivalent to one
            # backward pass over all `step_size` episodes -- only memory differs.
            for ms in range(0, len(mb), MICROBATCH_EPISODES):
                chunk = mb[ms:ms + MICROBATCH_EPISODES]
                (b_obs, b_gobs, b_masks, b_actions, b_old_logprobs,
                 b_adv, b_ret, pad_mask) = make_minibatch(chunk)
                to = lambda x: x.to(device)
                b_obs, b_gobs, b_masks = to(b_obs), to(b_gobs), to(b_masks)
                b_actions, b_old_logprobs = to(b_actions), to(b_old_logprobs)
                b_adv, b_ret, pad_mask = to(b_adv), to(b_ret), to(pad_mask)

                B = b_obs.size(0)
                h0 = torch.zeros(1, B, HIDDEN, device=device)
                c0 = torch.zeros(1, B, HIDDEN, device=device)
                dist, values, _ = model(b_obs, b_gobs, (h0, c0), b_masks, is_sequence=True)
                values = values.squeeze(-1)

                logratio = dist.log_prob(b_actions) - b_old_logprobs
                ratio = torch.exp(logratio)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_adv
                entropies = dist.entropy()

                actor_loss = -(torch.min(surr1, surr2) * pad_mask).sum() / mb_valid
                critic_loss = (F.mse_loss(values, b_ret, reduction='none')
                               * pad_mask).sum() / mb_valid
                entropy_loss = (entropies * pad_mask).sum() / mb_valid
                (actor_loss + VALUE_COEF * critic_loss
                 - entropy_coef * entropy_loss).backward()

                with torch.no_grad():
                    a_s += float(actor_loss); c_s += float(critic_loss)
                    e_s += float(entropy_loss)
                    kl_s += float((((ratio - 1) - logratio) * pad_mask).sum() / mb_valid)
                    cf_s += float(((torch.abs(ratio - 1.0) > CLIP_EPSILON).float()
                                   * pad_mask).sum() / mb_valid)
                    bid_mask = (b_masks[..., 32:].sum(-1) > 0).float() * pad_mask
                    play_mask = pad_mask - bid_mask
                    eb_s += float((entropies * bid_mask).sum()); nb_s += float(bid_mask.sum())
                    ep_s += float((entropies * play_mask).sum()); npl_s += float(play_mask.sum())

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            if nb_s > 0:
                ent_bid_acc += eb_s / nb_s; bid_steps += 1
            if npl_s > 0:
                ent_play_acc += ep_s / npl_s; play_steps += 1
            actor_acc += a_s; critic_acc += c_s; entropy_acc += e_s
            kl_acc += kl_s; clip_acc += cf_s
            steps += 1
            iter_kl += kl_s; iter_steps += 1

        iters_completed += 1
        if TARGET_KL is not None and iter_steps > 0 and (iter_kl / iter_steps) > 1.5 * TARGET_KL:
            stop = True

    if steps == 0:
        return {}
    return {"actor": actor_acc / steps, "critic": critic_acc / steps,
            "entropy": entropy_acc / steps,
            "entropy_bidding": ent_bid_acc / max(bid_steps, 1),
            "entropy_playing": ent_play_acc / max(play_steps, 1),
            "approx_kl": kl_acc / steps, "clip_frac": clip_acc / steps,
            "explained_variance": ev, "iters_completed": iters_completed,
            "grad_steps": steps, "episodes": len(episodes),
            "episodes_per_step": step_size}


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
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter("runs/belot_ppo_v4")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_START)
    vec = VectorizedBelot(NUM_ENVS)

    start_epoch, best_metric = 0, -float("inf")
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.pt")
    if os.path.exists(latest_ckpt):
        ck = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"] + 1
        if ck.get("selection_metric") == SELECTION_METRIC:
            best_metric = ck.get("best_metric", -float("inf"))
        else:
            print(f"  selection metric changed ({ck.get('selection_metric')} -> "
                  f"{SELECTION_METRIC}); resetting best_metric")
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh training run.")

    # C2: EMA tracker
    ema = snapshot(model, device) if EMA_DECAY else None

    frozen_pool = []
    if FROZEN_INIT and os.path.exists(FROZEN_INIT):
        frozen_pool.append(load_frozen(FROZEN_INIT, device))
        print(f"Frozen pool seeded from {FROZEN_INIT}")
    # C4: loaded as its OWN object from a path this trainer never writes.
    eval_reference = (load_frozen(EVAL_REFERENCE, device)
                      if EVAL_REFERENCE and os.path.exists(EVAL_REFERENCE) else None)
    if eval_reference is None:
        print(f"NOTE: {EVAL_REFERENCE} missing -- vs-reference eval disabled. Create it "
              f"once with:\n      cp checkpoints/best_model.pt {EVAL_REFERENCE}")

    for epoch in range(start_epoch, EPOCHS):
        lr = anneal(LR_START, LR_END, epoch, LR_ANNEAL_EPOCHS) * LR_BATCH_SCALE  # C1b
        for g in optimizer.param_groups:
            g["lr"] = lr
        ent_coef = anneal(ENTROPY_START, ENTROPY_END, epoch, ENTROPY_ANNEAL_EPOCHS)

        episodes, roll_info = collect_rollout(model, vec, TARGET_GAMES, device, frozen_pool)
        metrics = update(model, optimizer, episodes, device, entropy_coef=ent_coef)
        if not metrics:
            continue

        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA_DECAY).add_(pm, alpha=1 - EMA_DECAY)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)

        for k, tag in (("actor", "Loss/Actor"), ("critic", "Loss/Critic"),
                       ("entropy", "Loss/Entropy"), ("entropy_bidding", "Entropy/Bidding"),
                       ("entropy_playing", "Entropy/Playing"), ("approx_kl", "Diag/ApproxKL"),
                       ("clip_frac", "Diag/ClipFraction"),
                       ("explained_variance", "Diag/ExplainedVariance"),
                       ("iters_completed", "Diag/PPOItersCompleted"),
                       ("grad_steps", "Diag/GradSteps"),
                       ("episodes", "Diag/EpisodesCollected"),
                       ("episodes_per_step", "Diag/EpisodesPerOptimStep")):
            writer.add_scalar(tag, metrics[k], epoch)
        writer.add_scalar("Diag/FlushSteps", roll_info["flush_steps"], epoch)
        writer.add_scalar("Sched/LR", lr, epoch)
        writer.add_scalar("Sched/EntropyCoef", ent_coef, epoch)
        for k, v in roll_info["games_by_opponent"].items():
            writer.add_scalar(f"Rollout/GamesVs_{k}", v, epoch)

        if SNAPSHOT_EVERY and epoch > 0 and epoch % SNAPSHOT_EVERY == 0:
            frozen_pool.append(snapshot(model, device))
            if len(frozen_pool) > FROZEN_POOL_MAX:
                frozen_pool.pop(1 if len(frozen_pool) > 1 else 0)
            print(f"Epoch {epoch}: frozen pool size {len(frozen_pool)}")

        if epoch % EVAL_EVERY == 0:
            r = evaluate_matches(model, num_matches=EVAL_MATCHES, device=device,
                                 opponent="heuristic")
            writer.add_scalar("Eval/MatchWinVsHeuristic", r["match_win_rate"], epoch)
            writer.add_scalar("Eval/HandDiffVsHeuristic", r["avg_hand_diff"], epoch)
            line = (f"Epoch {epoch} | vs heuristic: match% {r['match_win_rate']:.3f} "
                    f"handdiff {r['avg_hand_diff']:+.2f} +-{r['hand_diff_ci95']:.2f}")
            metric = r["avg_hand_diff"]                      # C3

            if ema is not None:
                r_e = evaluate_matches(ema, num_matches=EVAL_MATCHES, device=device,
                                       opponent="heuristic")
                writer.add_scalar("Eval/HandDiffVsHeuristicEMA", r_e["avg_hand_diff"], epoch)
                line += f" | EMA handdiff {r_e['avg_hand_diff']:+.2f}"
                metric = max(metric, r_e["avg_hand_diff"])

            # C8: the yardstick ABOVE the model. Held out of the training pool on
            # purpose, so unlike vs-reference it cannot be exploited. It is ~40x
            # slower per hand than the heuristic (search at every card), hence far
            # fewer matches and a longer interval -- treat any single reading as
            # +-1.5 and judge it only as a trend.
            if PIMC_EVAL_EVERY and epoch % PIMC_EVAL_EVERY == 0:
                r_p = evaluate_matches(model, num_matches=PIMC_EVAL_MATCHES,
                                       device=device, opponent="pimc", pimc_D=PIMC_D)
                writer.add_scalar("Eval/HandDiffVsPIMC", r_p["avg_hand_diff"], epoch)
                writer.add_scalar("Eval/MatchWinVsPIMC", r_p["match_win_rate"], epoch)
                line += f" | vs PIMC {r_p['avg_hand_diff']:+.2f}"
            if eval_reference is not None:
                r_ref = evaluate_matches(model, num_matches=EVAL_MATCHES, device=device,
                                         opponent="model", frozen=eval_reference)
                writer.add_scalar("Eval/HandDiffVsReference", r_ref["avg_hand_diff"], epoch)
                # C3/D2: REPORTED, never selected on -- the reference is in the pool.
                line += f" | vs ref {r_ref['avg_hand_diff']:+.2f}"
            line += (f" | EV {metrics['explained_variance']:.2f} "
                     f"| eps/step {metrics['episodes_per_step']} "
                     f"| KL {metrics['approx_kl']:.4f}")
            print(line, flush=True)

            ck = {"epoch": epoch, "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "best_metric": best_metric, "selection_metric": SELECTION_METRIC}
            if ema is not None:
                ck["ema_state_dict"] = ema.state_dict()
            torch.save(ck, latest_ckpt)
            if metric > best_metric:
                best_metric = metric
                ck["best_metric"] = best_metric
                torch.save(ck, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    writer.close()


if __name__ == "__main__":
    train()
