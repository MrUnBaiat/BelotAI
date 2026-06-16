"""
Test suite for the vectorized Belot training architecture.

Layout:
  1. Engine correctness        - pure BelotEnv rules (unchanged, architecture-agnostic)
  2. Observation builder        - observation.build_observation (the new standalone port)
  3. Memory / GAE               - memory.Episode + make_minibatch (replaces AgentBuffer)
  4. Model forward / CTDE       - RecurrentMAPPOModel (unchanged)
  5. Vectorized rollout         - VectorizedBelot + collect_rollout + the three guards
  6. Update diagnostics & eval  - train.update metrics + eval.evaluate

The old AEC-wrapper / MultiAgentMemory / done-severing tests are gone: the training
path no longer routes through PettingZoo's agent_iter, and each Episode is now a
single complete trajectory, so cross-episode "done bleed" is structurally impossible
rather than something to guard at runtime.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from env import BelotEnv
from model import RecurrentMAPPOModel
from observation import build_observation
from vec_env import VectorizedBelot
from memory import Episode, make_minibatch
from train import collect_rollout, update
from eval import evaluate

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def random_legal_action(belot):
    legal = np.flatnonzero(belot.get_legal_actions())
    return int(np.random.choice(legal))


def play_to_rich_state(belot, max_steps=400):
    """Drive a raw BelotEnv with random legal actions until it sits in a rich
    PLAYING state: at least one completed trick (graveyard + last_trick populated)
    and a partially-filled current_trick. Resets if a game happens to end first."""
    for _ in range(max_steps):
        if belot.done:
            belot.reset()
        if (belot.phase == "PLAYING" and len(belot.graveyard) > 0
                and len(belot.current_trick) > 0 and not belot.done):
            return
        belot.step(random_legal_action(belot))
    raise RuntimeError("Failed to reach a rich PLAYING state")


# ========================================================================== #
# 1. ENGINE CORRECTNESS  (pure BelotEnv rules - unchanged)
# ========================================================================== #
def test_scoring_and_edge_cases():
    """Trick-evaluation hierarchy, the Pasledu last-trick bonus, and Bolt allocation."""
    engine = BelotEnv()
    engine.trump = 0
    engine.declaring_team = 0
    engine.defending_team = 1

    # Trump Jack beats trump 9 and non-trump Aces
    engine.current_trick = [(0, 15), (1, 23), (2, 2), (3, 4)]
    winner, points = engine._evaluate_trick()
    assert winner == 3, "Trump Jack did not beat trump 9 / non-trump Aces."
    assert points == 56, f"Expected 56 points, got {points}."

    # Pasledu (+10) on the final trick
    engine.tricks_played = 7
    engine.current_trick = [(0, 15), (1, 14), (2, 13), (3, 12)]
    engine.current_player = 3
    engine.raw_points_by_team = [0, 0]
    winner, points = engine._evaluate_trick()
    engine.raw_points_by_team[winner % 2] += points + 10
    assert engine.raw_points_by_team[0] == 30, "Pasledu failed to award the +10 bonus."

    # Bolt: declarer fails to exceed 80
    engine.raw_points_by_team = [80, 82]
    engine.bolts_by_team = [0, 0]
    engine.tricks_won_by_team = [4, 4]
    final_rewards = engine._calculate_final_rewards()
    assert final_rewards[0] == 0, "Bolt: declaring team should get 0."
    assert final_rewards[1] == 16, "Bolt: defending team should get 16."
    assert engine.bolts_by_team[0] == 1, "Bolt counter did not increment."


def test_action_masking_integrity():
    """Follow-suit, forced-ruff, and the overruff rule via get_legal_actions()."""
    engine = BelotEnv()
    engine.phase = "PLAYING"
    engine.trump = 0
    engine.declarer = 0
    engine.current_player = 1
    engine.declarer_has_played_trump = True

    # Must follow suit when able
    engine.current_trick = [(0, 11)]
    engine.hands[1] = [15, 0, 4]
    mask = engine.get_legal_actions()
    assert mask[15] and not mask[0] and not mask[4], "Follow-suit not enforced."

    # Must ruff when void in led suit
    engine.current_trick = [(0, 11)]
    engine.hands[1] = [23, 0]
    mask = engine.get_legal_actions()
    assert not mask[23] and mask[0], "Forced ruff not enforced."

    # Must overruff if able
    engine.current_trick = [(0, 11), (2, 2)]
    engine.current_player = 3
    engine.hands[3] = [0, 4, 23]
    mask = engine.get_legal_actions()
    assert not mask[23] and not mask[0] and mask[4], "Overruff rule not enforced."


def test_single_trick_dense_rewards():
    """A resolved trick emits the normalized zero-sum dense reward to both teams."""
    env = BelotEnv()
    env.reset()
    env.phase = "PLAYING"
    env.trump = 0
    env.current_player = 0
    env.tricks_played = 0
    env.declarer = 0
    env.hands[0] = [4]
    env.hands[1] = [2]
    env.hands[2] = [15]
    env.hands[3] = [11]

    env.step(4); env.step(2); env.step(15)
    _, step_rewards, done, _ = env.step(11)

    win = 55.0 / 162.0
    assert np.isclose(step_rewards[0], win) and np.isclose(step_rewards[2], win)
    assert np.isclose(step_rewards[1], -win) and np.isclose(step_rewards[3], -win)


def test_calculate_final_rewards():
    """End-of-hand game-point math: clean wins, ties, bolts, third-bolt penalty, capot."""
    env = BelotEnv()

    def check(tricks, raw, dec_team, bolts, exp_pts, exp_bolts, name):
        env.tricks_won_by_team = tricks
        env.raw_points_by_team = raw
        env.declaring_team = dec_team
        env.defending_team = 1 - dec_team
        env.bolts_by_team = bolts.copy()
        pts = env._calculate_final_rewards()
        assert pts == exp_pts, f"[{name}] expected {exp_pts}, got {pts}"
        assert env.bolts_by_team == exp_bolts, f"[{name}] bolts {exp_bolts}, got {env.bolts_by_team}"

    check([4, 4], [86, 76], 0, [0, 0], [8, 8, 8, 8], [0, 0], "Simple 86-76")
    check([4, 4], [81, 81], 0, [0, 0], [8, 8, 8, 8], [0, 0], "Equal 81-81")
    check([4, 4], [70, 92], 0, [0, 0], [0, 16, 0, 16], [1, 0], "Standard Bolt")
    check([4, 4], [70, 92], 0, [2, 0], [-10, 16, -10, 16], [0, 0], "3rd Bolt Penalty")
    check([8, 0], [162, 0], 0, [0, 0], [16, -10, 16, -10], [0, 0], "Capot")


# ========================================================================== #
# 2. OBSERVATION BUILDER  (observation.build_observation)
# ========================================================================== #
def test_state_encoding():
    """build_observation returns the right shapes/dtypes and (in a neutral match
    context) keeps every feature in [0, 1]."""
    belot = BelotEnv()
    abs_id = belot.current_player
    obs, g_obs, mask = build_observation(belot, abs_id, [0, 0])

    assert obs.shape == (513,) and g_obs.shape == (332,) and mask.shape == (38,)
    assert obs.dtype == np.float32 and g_obs.dtype == np.float32 and mask.dtype == np.int8
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert np.all(g_obs >= 0.0) and np.all(g_obs <= 1.0)


def test_local_state_truth_mapping():
    """Every slice of the 513-dim local vector matches the engine source of truth."""
    np.random.seed(1)
    belot = BelotEnv()
    play_to_rich_state(belot)
    abs_id = belot.current_player
    obs, _, _ = build_observation(belot, abs_id, [0, 0])
    engine = belot
    idx = 0

    # 1. Private Hand
    exp = np.zeros(32, np.float32); exp[engine.hands[abs_id]] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 32], exp, err_msg="Private Hand"); idx += 32

    # 2. Face-Up Card
    exp = np.zeros(32, np.float32)
    if engine.phase == "BIDDING" and engine.face_up_card is not None:
        exp[engine.face_up_card] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 32], exp, err_msg="Face-Up"); idx += 32

    # 3. Current Trump
    exp = np.zeros(5, np.float32)
    exp[0 if engine.trump is None else 1 + engine.trump] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 5], exp, err_msg="Trump"); idx += 5

    # 4. Relative Declarer
    exp = np.zeros(5, np.float32)
    exp[0 if engine.declarer is None else 1 + (engine.declarer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 5], exp, err_msg="Declarer"); idx += 5

    # 5. Phase
    exp = np.zeros(3, np.float32)
    if engine.phase == "BIDDING":
        exp[0 if engine.bidding_round == 1 else 1] = 1.0
    else:
        exp[2] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 3], exp, err_msg="Phase"); idx += 3

    # 6. Current Trick
    for rel in [1, 2, 3]:
        exp = np.zeros(36, np.float32); abs_p = (abs_id + rel) % 4
        for seq, (p, c) in enumerate(engine.current_trick):
            if p == abs_p:
                exp[c] = 1.0; exp[32 + seq] = 1.0
        np.testing.assert_array_equal(obs[idx:idx + 36], exp, err_msg=f"Trick rel{rel}"); idx += 36

    # 7. Game Stats (validated tightly in the global test)
    idx += 6

    # 8. Relative Dealer
    exp = np.zeros(4, np.float32); exp[(engine.dealer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 4], exp, err_msg="Dealer"); idx += 4

    # 9. Last Trick
    for rel in [0, 1, 2, 3]:
        exp = np.zeros(36, np.float32); abs_p = (abs_id + rel) % 4
        for seq, (p, c) in enumerate(engine.last_trick):
            if p == abs_p:
                exp[c] = 1.0; exp[32 + seq] = 1.0
        np.testing.assert_array_equal(obs[idx:idx + 36], exp, err_msg=f"LastTrick rel{rel}"); idx += 36

    # 10. Belief Matrix - verify the hard constraints
    belief = obs[idx:idx + 96].reshape(3, 32)
    others = [(abs_id + 1) % 4, (abs_id + 2) % 4, (abs_id + 3) % 4]
    for i, p in enumerate(others):
        known = np.where(engine.known_cards[p])[0]
        if len(known):
            assert np.all(belief[i][known] == 1.0), f"Known cards not 1.0 for {p}"
        impossible = np.where(engine.impossible_cards[p])[0]
        if len(impossible):
            assert np.all(belief[i][impossible] == 0.0), f"Impossible cards not 0.0 for {p}"
        assert np.all(belief[i][engine.hands[abs_id]] == 0.0), "Opponent believed to hold our card"
    idx += 96

    # 11. Trick Number
    exp = np.zeros(8, np.float32); exp[min(engine.tricks_played, 7)] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 8], exp, err_msg="Trick Number"); idx += 8

    # 12. Legal Action Mask feature
    exp = engine.get_legal_actions().astype(np.float32)
    np.testing.assert_array_equal(obs[idx:idx + 38], exp, err_msg="Mask feature"); idx += 38

    # 13. Graveyard
    exp = np.zeros(32, np.float32); exp[engine.graveyard] = 1.0
    np.testing.assert_array_equal(obs[idx:idx + 32], exp, err_msg="Graveyard"); idx += 32

    assert idx == 513, f"Local index desynced: {idx}"


def test_global_state_truth_mapping():
    """Every slice of the 332-dim global vector matches the engine source of truth,
    including the externally-supplied match score."""
    np.random.seed(2)
    belot = BelotEnv()
    play_to_rich_state(belot)
    abs_id = belot.current_player
    ms = [40, 30]                          # exercise non-zero match-score features
    _, g, _ = build_observation(belot, abs_id, ms)
    engine = belot
    team_us, team_them = abs_id % 2, 1 - abs_id % 2
    gi = 0

    # 1. Relative Hands
    for rel in [0, 1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        exp = np.zeros(32, np.float32); exp[engine.hands[abs_p]] = 1.0
        np.testing.assert_array_equal(g[gi:gi + 32], exp, err_msg=f"Hand rel{rel}"); gi += 32

    # 2. Graveyard
    exp = np.zeros(32, np.float32); exp[engine.graveyard] = 1.0
    np.testing.assert_array_equal(g[gi:gi + 32], exp, err_msg="Graveyard"); gi += 32

    # 3. Face-Up
    exp = np.zeros(32, np.float32)
    if engine.phase == "BIDDING" and engine.face_up_card is not None:
        exp[engine.face_up_card] = 1.0
    np.testing.assert_array_equal(g[gi:gi + 32], exp, err_msg="Face-Up"); gi += 32

    # 4. Trump
    exp = np.zeros(5, np.float32); exp[0 if engine.trump is None else 1 + engine.trump] = 1.0
    np.testing.assert_array_equal(g[gi:gi + 5], exp, err_msg="Trump"); gi += 5

    # 5. Relative Declarer
    exp = np.zeros(5, np.float32)
    exp[0 if engine.declarer is None else 1 + (engine.declarer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(g[gi:gi + 5], exp, err_msg="Declarer"); gi += 5

    # 6. Relative Dealer
    exp = np.zeros(4, np.float32); exp[(engine.dealer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(g[gi:gi + 4], exp, err_msg="Dealer"); gi += 4

    # 7. Phase
    exp = np.zeros(3, np.float32)
    if engine.phase == "BIDDING":
        exp[0 if engine.bidding_round == 1 else 1] = 1.0
    else:
        exp[2] = 1.0
    np.testing.assert_array_equal(g[gi:gi + 3], exp, err_msg="Phase"); gi += 3

    # 8. Relative Current Trick
    for rel in [1, 2, 3]:
        exp = np.zeros(36, np.float32); abs_p = (abs_id + rel) % 4
        for seq, (p, c) in enumerate(engine.current_trick):
            if p == abs_p:
                exp[c] = 1.0; exp[32 + seq] = 1.0
        np.testing.assert_array_equal(g[gi:gi + 36], exp, err_msg=f"Trick rel{rel}"); gi += 36

    # 9. Game Stats (uses the supplied match score)
    exp = np.array([
        ms[team_us] / 101.0, ms[team_them] / 101.0,
        engine.raw_points_by_team[team_us] / 162.0, engine.raw_points_by_team[team_them] / 162.0,
        engine.bolts_by_team[team_us] / 2.0, engine.bolts_by_team[team_them] / 2.0,
    ], dtype=np.float32)
    np.testing.assert_allclose(g[gi:gi + 6], exp, err_msg="Game Stats"); gi += 6

    # 10. Trick Number
    exp = np.zeros(8, np.float32); exp[min(engine.tricks_played, 7)] = 1.0
    np.testing.assert_array_equal(g[gi:gi + 8], exp, err_msg="Trick Number"); gi += 8

    # 11. Declarer Has Played Trump
    np.testing.assert_array_equal(
        g[gi:gi + 1], np.array([float(engine.declarer_has_played_trump)], np.float32),
        err_msg="Declarer Played Trump"); gi += 1

    assert gi == 332, f"Global index desynced: {gi}"


def test_global_to_local_deduction():
    """Shared factual blocks must be byte-identical between the local and global
    vectors from the acting agent's perspective (catches index drift between them)."""
    np.random.seed(3)
    belot = BelotEnv()
    play_to_rich_state(belot)
    abs_id = belot.current_player
    local, glob, _ = build_observation(belot, abs_id, [0, 0])

    pairs = {
        "Private Hand":   (slice(0, 32),    slice(0, 32)),
        "Face-Up":        (slice(32, 64),   slice(160, 192)),
        "Trump":          (slice(64, 69),   slice(192, 197)),
        "Declarer":       (slice(69, 74),   slice(197, 202)),
        "Phase":          (slice(74, 77),   slice(206, 209)),
        "Current Trick":  (slice(77, 185),  slice(209, 317)),
        "Game Stats":     (slice(185, 191), slice(317, 323)),
        "Dealer":         (slice(191, 195), slice(202, 206)),
        "Trick Number":   (slice(435, 443), slice(323, 331)),
        "Graveyard":      (slice(481, 513), slice(128, 160)),
    }
    for name, (ls, gs) in pairs.items():
        np.testing.assert_array_equal(local[ls], glob[gs], err_msg=f"Perspective mismatch: {name}")


# ========================================================================== #
# 3. MEMORY / GAE  (memory.Episode + make_minibatch)
# ========================================================================== #
def test_episode_gae_complete_trajectory():
    """Each Episode is one COMPLETE trajectory: GAE bootstraps 0.0 at the end and
    advantage flows fully back through the sequence (no intra-buffer done severing,
    which the old AgentBuffer needed and which is now impossible by construction)."""
    gamma, lam = 0.99, 0.95
    rewards, values = [1.0, 1.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]

    ep = Episode()
    for r, v in zip(rewards, values):
        ep.add(np.zeros(513, np.float32), np.zeros(332, np.float32),
               np.ones(38, np.float32), 0, -0.5, v)
        ep.rewards[-1] = r
    returns, adv = ep.compute_gae(gamma, lam)

    # Reference GAE: single complete episode, bootstrap 0.0.
    ref, gae, vv = [0.0] * 4, 0.0, values + [0.0]
    for t in reversed(range(4)):
        delta = rewards[t] + gamma * vv[t + 1] - vv[t]
        gae = delta + gamma * lam * gae
        ref[t] = gae

    np.testing.assert_allclose(adv.numpy(), ref, rtol=1e-5)
    np.testing.assert_allclose(returns.numpy(), np.array(ref) + np.array(values), rtol=1e-5)
    # Advantage at t=1 strictly exceeds its own reward because future rewards flow back.
    assert adv[1].item() > rewards[1], "Advantage failed to propagate through the trajectory."


def test_make_minibatch_padding():
    """Variable-length episodes pad into rectangular (B,T,...) tensors; pad_mask marks
    valid steps and padded action-mask rows are all-ones (so they never NaN log-probs)."""
    def make_ep(length):
        ep = Episode()
        for _ in range(length):
            ep.add(np.random.randn(513).astype(np.float32),
                   np.random.randn(332).astype(np.float32),
                   np.ones(38, np.float32), 0, -0.5, 0.5)
            ep.rewards[-1] = 1.0
        ep.returns, ep.advantages = ep.compute_gae()
        return ep

    eps = [make_ep(2), make_ep(3)]
    b_obs, b_gobs, b_masks, b_actions, b_logp, b_adv, b_ret, pad_mask = make_minibatch(eps)

    assert b_obs.shape == (2, 3, 513)
    assert b_gobs.shape == (2, 3, 332)
    assert b_masks.shape == (2, 3, 38)
    assert b_actions.shape == (2, 3) and b_adv.shape == (2, 3) and b_ret.shape == (2, 3)
    assert torch.equal(pad_mask, torch.tensor([[1., 1., 0.], [1., 1., 1.]]))
    assert torch.all(b_masks[0, 2] == 1.0), "Padded action-mask step must be all-ones."


# ========================================================================== #
# 4. MODEL FORWARD / CTDE  (RecurrentMAPPOModel - unchanged)
# ========================================================================== #
def test_lstm_and_ctde_rollout():
    """Single-step rollout forward: actor over 38 actions, scalar critic, LSTM advances."""
    model = RecurrentMAPPOModel()
    local = torch.randn(1, 513); glob = torch.randn(1, 332); mask = torch.ones(1, 38)
    hc = (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
    dist, value, new_hc = model(local, glob, hc, mask, is_sequence=False)
    assert dist.logits.shape == (1, 38)
    assert value.shape == (1, 1)
    assert new_hc[0].shape == (1, 1, 512)
    assert not torch.equal(hc[0], new_hc[0]), "LSTM hidden state did not update."


def test_lstm_and_ctde_training():
    """Batched sequence forward keeps (Batch, Seq, Feature) structure for actor & critic."""
    model = RecurrentMAPPOModel()
    B, T = 4, 8
    local = torch.randn(B, T, 513); glob = torch.randn(B, T, 332); mask = torch.ones(B, T, 38)
    hc = (torch.zeros(1, B, 512), torch.zeros(1, B, 512))
    dist, values, _ = model(local, glob, hc, mask, is_sequence=True)
    assert dist.logits.shape == (B, T, 38)
    assert values.shape == (B, T, 1)


def test_gradient_flow_padding():
    """pad_mask must sever the gradient graph for padded steps in BOTH heads."""
    model = RecurrentMAPPOModel()
    B, T = 2, 3
    local = torch.randn(B, T, 513, requires_grad=True)
    glob = torch.randn(B, T, 332, requires_grad=True)
    mask = torch.ones(B, T, 38)
    actions = torch.zeros((B, T), dtype=torch.long)
    hc = (torch.zeros(1, B, 512), torch.zeros(1, B, 512))
    pad_mask = torch.tensor([[1., 1., 1.], [1., 0., 0.]])

    dist, values, _ = model(local, glob, hc, mask, is_sequence=True)
    values = values.squeeze(-1)
    critic_loss = (F.mse_loss(values, torch.ones_like(values), reduction='none') * pad_mask).sum()
    actor_loss = -(dist.log_prob(actions) * pad_mask).sum()
    (critic_loss + actor_loss).backward()

    assert torch.any(local.grad[0, 2] != 0.0), "Valid step got no actor gradient."
    assert torch.all(local.grad[1, 1] == 0.0) and torch.all(local.grad[1, 2] == 0.0), "Actor grad leak into padding."
    assert torch.any(glob.grad[0, 2] != 0.0), "Valid step got no critic gradient."
    assert torch.all(glob.grad[1, 1] == 0.0) and torch.all(glob.grad[1, 2] == 0.0), "Critic grad leak into padding."


# ========================================================================== #
# 5. VECTORIZED ROLLOUT  (VectorizedBelot + collect_rollout + the three guards)
# ========================================================================== #
def test_vectorized_match_lifecycle():
    """finish_and_reset accumulates game points, and wipes match score + bolts the
    instant a side reaches 101 (the cross-episode lifecycle the wrapper used to own)."""
    vec = VectorizedBelot(1)

    # Below threshold -> scores and bolts persist
    vec.match_scores[0] = [95, 80]; vec.envs[0].bolts_by_team = [1, 0]
    vec.finish_and_reset(0, {"game_points": [0, 0, 0, 0]})
    assert vec.match_scores[0] == [95, 80]
    assert vec.envs[0].bolts_by_team == [1, 0]

    # Cross 101 -> everything wipes
    vec.match_scores[0] = [95, 80]; vec.envs[0].bolts_by_team = [1, 2]
    vec.finish_and_reset(0, {"game_points": [10, 0, 10, 0]})
    assert vec.match_scores[0] == [0, 0]
    assert vec.envs[0].bolts_by_team == [0, 0]

    # Exactly 101 -> wipes
    vec.match_scores[0] = [80, 90]; vec.envs[0].bolts_by_team = [0, 1]
    vec.finish_and_reset(0, {"game_points": [0, 11, 0, 11]})
    assert vec.match_scores[0] == [0, 0]
    assert vec.envs[0].bolts_by_team == [0, 0]


def test_rollout_true_up_credit_assignment():
    """The contract collect_rollout relies on: over a full hand, the reward credited
    to each seat (broadcast dense rewards + terminal true-up) sums exactly to that
    team's strategic target (game_point_diff / 16), and partners are symmetric."""
    np.random.seed(0)
    belot = BelotEnv()
    seat_totals = [0.0, 0.0, 0.0, 0.0]
    game_points = None
    while not belot.done:
        _, step_rewards, done, info = belot.step(random_legal_action(belot))
        for i in range(4):
            seat_totals[i] += step_rewards[i]
        if done:
            game_points = info["game_points"]

    t0 = (game_points[0] - game_points[1]) / 16.0
    t1 = (game_points[1] - game_points[0]) / 16.0
    np.testing.assert_almost_equal(seat_totals[0], t0, decimal=4)
    np.testing.assert_almost_equal(seat_totals[2], t0, decimal=4)
    np.testing.assert_almost_equal(seat_totals[1], t1, decimal=4)
    np.testing.assert_almost_equal(seat_totals[3], t1, decimal=4)


def test_collect_rollout_integrity():
    """End-to-end guard check on the real rollout:
       - discard-in-flight  -> exactly target_games*4 COMPLETE episodes
       - buffer contiguity  -> episodes group cleanly per game (seats 0..3)
       - terminal true-up   -> each game is zero-sum with partner symmetry."""
    torch.manual_seed(0); np.random.seed(0)
    model = RecurrentMAPPOModel()
    vec = VectorizedBelot(4)
    games = 8
    eps = collect_rollout(model, vec, games, CPU)

    assert len(eps) == games * 4, "Did not collect exactly target_games*4 episodes."
    for ep in eps:
        assert len(ep) >= 8
        assert len(ep.obs) == len(ep.global_obs) == len(ep.masks) == len(ep)
        assert len(ep.rewards) == len(ep.values) == len(ep.actions) == len(ep)
        assert ep.obs[0].shape == (513,) and ep.global_obs[0].shape == (332,) and ep.masks[0].shape == (38,)
        assert all(np.isfinite(r) for r in ep.rewards)

    for g in range(games):
        s = [sum(eps[g * 4 + k].rewards) for k in range(4)]
        assert abs(sum(s)) < 1e-3, f"Game {g} not zero-sum: {s}"
        np.testing.assert_almost_equal(s[0], s[2], decimal=4)   # partner symmetry
        np.testing.assert_almost_equal(s[1], s[3], decimal=4)
        np.testing.assert_almost_equal(s[0], -s[1], decimal=4)  # team zero-sum


def test_hidden_state_routing_no_leakage():
    """The safety property that makes (env, seat)-keyed gather/scatter sound: a batched
    forward over N rows must equal N independent single-row forwards. If any row leaked
    into another, hidden-state routing would silently corrupt timelines."""
    torch.manual_seed(0)
    model = RecurrentMAPPOModel(); model.eval()
    N = 5
    local = torch.randn(N, 513); glob = torch.randn(N, 332); mask = torch.ones(N, 38)
    h = torch.randn(1, N, 512); c = torch.randn(1, N, 512)

    with torch.no_grad():
        dist_b, val_b, (nh_b, nc_b) = model(local, glob, (h, c), mask, is_sequence=False)
        for e in range(N):
            dist_e, val_e, (nh_e, nc_e) = model(
                local[e:e + 1], glob[e:e + 1],
                (h[:, e:e + 1].contiguous(), c[:, e:e + 1].contiguous()),
                mask[e:e + 1], is_sequence=False)
            assert torch.allclose(dist_b.logits[e], dist_e.logits[0], atol=1e-5)
            assert torch.allclose(val_b[e], val_e[0], atol=1e-5)
            assert torch.allclose(nh_b[:, e], nh_e[:, 0], atol=1e-5)
            assert torch.allclose(nc_b[:, e], nc_e[:, 0], atol=1e-5)


# ========================================================================== #
# 6. UPDATE DIAGNOSTICS & EVAL
# ========================================================================== #
def test_update_returns_finite_diagnostics():
    """update() runs a real PPO step and returns finite, in-range health metrics
    (also confirms entropy/value terms are wired into the loss pipeline)."""
    torch.manual_seed(0); np.random.seed(0)
    model = RecurrentMAPPOModel()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    vec = VectorizedBelot(4)
    eps = collect_rollout(model, vec, 6, CPU)
    m = update(model, opt, eps, CPU)

    for k in ["actor", "critic", "entropy", "approx_kl", "clip_frac", "explained_variance"]:
        assert k in m and np.isfinite(m[k]), f"metric {k} missing or non-finite"
    assert m["approx_kl"] >= 0.0
    assert 0.0 <= m["clip_frac"] <= 1.0
    assert m["entropy"] >= 0.0


def test_eval_harness():
    """evaluate() returns a valid win-rate and finite point differential against both
    a random opponent and a frozen reference model."""
    torch.manual_seed(0); np.random.seed(0)
    model = RecurrentMAPPOModel()

    res = evaluate(model, num_games=10, device=CPU)
    assert 0.0 <= res["win_rate"] <= 1.0 and np.isfinite(res["avg_point_diff"])

    res_frozen = evaluate(model, num_games=6, device=CPU, frozen=copy.deepcopy(model))
    assert 0.0 <= res_frozen["win_rate"] <= 1.0 and np.isfinite(res_frozen["avg_point_diff"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))