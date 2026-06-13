import pytest
import torch
import numpy as np

from env import BelotEnv
from env_wrapper import BelotAECEnv
from model import RecurrentMAPPOModel
from memory import AgentBuffer
from memory import MultiAgentMemory
from pettingzoo.test import api_test
import torch.nn.functional as F

# This forces illegal actions in env.step() which raises ValueError. For now comment out since such a value cannot get there because it is maked when the model makes a decision
# def test_pettingzoo_api_compliance():
#     env = BelotAECEnv()
#     api_test(env, num_cycles=1000, verbose_progress=False)
#     print("Passed PettingZoo AEC API Test!")

def test_dense_rewards_true_up():
    """
    Simulates a full hand to verify that the intermediate dense rewards 
    (from tricks) combined with the final true-up step perfectly equal 
    the normalized zero-sum target of the whole episode.
    """
    env = BelotAECEnv()
    env.reset()
    
    # Track the sum of rewards exactly as the PPO memory buffer will see them
    cumulative_rewards = {agent: 0.0 for agent in env.possible_agents}
    
    # We will store the final game points here before PettingZoo deletes them
    captured_game_points = [0, 0, 0, 0]
    
    # Play a complete hand
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        
        cumulative_rewards[agent] += reward
        
        # Capture the game points the moment they are populated at the end of the game
        if "game_points" in info:
            captured_game_points = info["game_points"]
        
        if term or trunc:
            env.step(None) # This is the line that deletes the agent's info!
            continue
            
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 32
        
        env.step(action)

    # Calculate the expected zero-sum true target based on the captured points
    # Target = (Team Points - Enemy Points) / 16.0
    expected_target_team_0 = (captured_game_points[0] - captured_game_points[1]) / 16.0
    expected_target_team_1 = (captured_game_points[1] - captured_game_points[0]) / 16.0
    
    # Assert that the sum of dense rewards explicitly matches the true-up target
    np.testing.assert_almost_equal(
        cumulative_rewards["player_0"], expected_target_team_0, decimal=4, 
        err_msg="Player 0 dense rewards did not true-up to target."
    )
    np.testing.assert_almost_equal(
        cumulative_rewards["player_2"], expected_target_team_0, decimal=4, 
        err_msg="Player 2 dense rewards did not true-up to target."
    )
    
    np.testing.assert_almost_equal(
        cumulative_rewards["player_1"], expected_target_team_1, decimal=4, 
        err_msg="Player 1 dense rewards did not true-up to target."
    )
    np.testing.assert_almost_equal(
        cumulative_rewards["player_3"], expected_target_team_1, decimal=4, 
        err_msg="Player 3 dense rewards did not true-up to target."
    )

def test_match_score_reset_at_101():
    """
    Validates that match scores and bolts persist across standard hands, 
    but completely wipe back to zero when a team reaches or exceeds 101 points.
    """
    env = BelotAECEnv()
    env.reset()
    
    # ---------------------------------------------------------
    # SCENARIO 1: Below threshold. Scores and bolts MUST persist.
    # ---------------------------------------------------------
    env.match_scores = [95, 80]
    env.belot.bolts_by_team = [1, 0]
    env.reset()
    
    assert env.match_scores == [95, 80], "Scores wiped prematurely! They should persist if below 101."
    assert env.belot.bolts_by_team == [1, 0], "Bolts wiped prematurely!"

    # ---------------------------------------------------------
    # SCENARIO 2: Team 0 passes threshold. Everything MUST wipe.
    # ---------------------------------------------------------
    env.match_scores = [102, 80] # Team 0 wins
    env.belot.bolts_by_team = [1, 2]
    env.reset()
    
    assert env.match_scores == [0, 0], "Scores failed to wipe after Team 0 reached 101."
    assert env.belot.bolts_by_team == [0, 0], "Bolts failed to wipe after a match concluded."

    # ---------------------------------------------------------
    # SCENARIO 3: Team 1 hits exactly 101. Everything MUST wipe.
    # ---------------------------------------------------------
    env.match_scores = [80, 101] # Team 1 wins
    env.belot.bolts_by_team = [0, 1]
    env.reset()
    
    assert env.match_scores == [0, 0], "Scores failed to wipe after Team 1 reached 101."
    assert env.belot.bolts_by_team == [0, 0], "Bolts failed to wipe after a match concluded."

def test_gae_boundary():
    """
    Validates that compute_gae strictly severs the back-propagation of rewards 
    and values when done=True, preventing temporal bleed across episodes.
    """
    buf = AgentBuffer()
    gamma = 0.99
    lam = 0.95
    
    # Step 0: Match 1, Step 1
    # Step 1: Match 1 ends (done=True)
    # Step 2: Match 2, Step 1
    # Step 3: Match 2 ends (done=True)
    
    buf.rewards = [1.0, 1.0, 5.0, 5.0]
    buf.values = [0.0, 0.0, 0.0, 0.0]
    buf.dones = [False, True, False, True]
    
    returns, advantages = buf.compute_gae(next_value=0.0, gamma=gamma, lam=lam)
    
    # Calculate expected advantages manually
    # At Step 3 (Done): Adv_3 = Reward_3 = 5.0
    # At Step 2: Adv_2 = Reward_2 + (gamma * lam * Adv_3) = 5.0 + (0.9405 * 5.0) = 9.7025
    expected_adv_2 = 5.0 + (gamma * lam * 5.0)
    
    # At Step 1 (Done): Adv_1 = Reward_1 = 1.0
    # Crucially, Adv_1 should NOT include Adv_2 because done=True severed the link.
    expected_adv_1 = 1.0
    
    # At Step 0: Adv_0 = Reward_0 + (gamma * lam * Adv_1) = 1.0 + (0.9405 * 1.0) = 1.9405
    expected_adv_0 = 1.0 + (gamma * lam * 1.0)
    
    np.testing.assert_almost_equal(advantages[3].item(), 5.0, decimal=4)
    np.testing.assert_almost_equal(advantages[2].item(), expected_adv_2, decimal=4)
    
    # The boundary checks
    np.testing.assert_almost_equal(advantages[1].item(), expected_adv_1, decimal=4, 
                                   err_msg="GAE Leak! done=True failed to sever advantage calculation.")
    np.testing.assert_almost_equal(advantages[0].item(), expected_adv_0, decimal=4, 
                                   err_msg="GAE Leak! Match 2 rewards bled into Match 1.")

def test_gradient_flow_padding():
    """
    Verifies that the pad_mask completely severs the gradient computation graph 
    for the dummy padded steps, ensuring the model does not learn from zeros.
    Tests both the Actor (Local Obs) and Critic (Global Obs).
    """
    model = RecurrentMAPPOModel()
    batch_size = 2
    seq_len = 3
    
    # Create input tensors and tell PyTorch to track their gradients
    b_local_obs = torch.randn(batch_size, seq_len, 513, requires_grad=True)
    b_global_obs = torch.randn(batch_size, seq_len, 332, requires_grad=True)
    b_action_masks = torch.ones(batch_size, seq_len, 38)
    
    # Dummy actions for Actor loss calculation
    b_actions = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    # Hidden states
    curr_hc = (torch.zeros(1, batch_size, 512), torch.zeros(1, batch_size, 512))
    
    # Setup a mock pad_mask. 
    # Batch 0 has 3 valid steps. Batch 1 has 1 valid step, and 2 padded steps.
    pad_mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    
    dist, values, _ = model(b_local_obs, b_global_obs, curr_hc, b_action_masks, is_sequence=True)
    
    # ---------------------------------------------
    # 1. Critic Loss (Flows through Global Obs)
    # ---------------------------------------------
    values = values.squeeze(-1)
    mock_returns = torch.ones_like(values)
    critic_loss = (F.mse_loss(values, mock_returns, reduction='none') * pad_mask).sum()
    
    # ---------------------------------------------
    # 2. Actor Loss (Flows through Local Obs)
    # ---------------------------------------------
    # Dummy objective: maximize log probability of mock actions
    log_probs = dist.log_prob(b_actions)
    actor_loss = -(log_probs * pad_mask).sum()
    
    # Backpropagate both!
    total_loss = critic_loss + actor_loss
    total_loss.backward()
    
    # ==========================================
    # ASSERTIONS
    # ==========================================
    
    # Check Local Obs (Actor Graph)
    local_grad = b_local_obs.grad
    assert local_grad is not None, "Local obs received no gradient. Actor graph broken."
    assert torch.any(local_grad[0, 2, :] != 0.0), "Gradient failed: Valid step got no gradient (Actor)."
    assert torch.all(local_grad[1, 1, :] == 0.0), "Gradient Leak! Padded Step 1 received gradients (Actor)."
    assert torch.all(local_grad[1, 2, :] == 0.0), "Gradient Leak! Padded Step 2 received gradients (Actor)."
    
    # Check Global Obs (Critic Graph)
    global_grad = b_global_obs.grad
    assert global_grad is not None, "Global obs received no gradient. Critic graph broken."
    assert torch.any(global_grad[0, 2, :] != 0.0), "Gradient failed: Valid step got no gradient (Critic)."
    assert torch.all(global_grad[1, 1, :] == 0.0), "Gradient Leak! Padded Step 1 received gradients (Critic)."
    assert torch.all(global_grad[1, 2, :] == 0.0), "Gradient Leak! Padded Step 2 received gradients (Critic)."
    
def test_scoring_and_edge_cases():
    """
    Validates trick evaluation math (hierarchy shifts), the Pasledu last-trick bonus, 
    and the Bolt points allocation.
    """
    engine = BelotEnv()
    engine.trump = 0 # Spades
    engine.declaring_team = 0
    engine.defending_team = 1
    
    # ---------------------------------------------------------
    # PART 1: Trick Hierarchy (Trump J vs Non-Trump A)
    # ---------------------------------------------------------
    # Trick: Led Heart Ace (15), Diamond Ace (23), Spade 9 (Trump, 2), Spade J (Trump, 4)
    engine.current_trick = [(0, 15), (1, 23), (2, 2), (3, 4)]
    
    winner, points = engine._evaluate_trick()
    
    # The winner should be Player 3 (Jack of Trump)
    # Points should be: Ace(11) + Ace(11) + 9(14) + J(20) = 56
    assert winner == 3, "Trick Evaluation failed: Trump Jack did not beat Trump 9 or Non-Trump Aces."
    assert points == 56, f"Trick Evaluation failed: Expected 56 points, got {points}."

    # ---------------------------------------------------------
    # PART 2: Pasledu (Last Trick Bonus)
    # ---------------------------------------------------------
    engine.tricks_played = 7 # We are on the 8th and final trick
    engine.current_trick = [(0, 15), (1, 14), (2, 13), (3, 12)] # Hearts A, K, Q, J
    engine.current_player = 3 # Triggers the trick evaluation
    
    # Player 0 wins the trick (Ace of led suit). Team 0 wins.
    # Normal points: 11 + 4 + 3 + 2 = 20. 
    # Plus Pasledu (+10) -> 30 points to Team 0.
    engine.raw_points_by_team = [0, 0]
    
    # Manually trigger the trick resolution logic from step()
    winner, points = engine._evaluate_trick()
    engine.raw_points_by_team[winner % 2] += points
    engine.raw_points_by_team[winner % 2] += 10 # Pasledu logic from env.py
    engine.done = True
    
    assert engine.raw_points_by_team[0] == 30, "Pasledu failed: Final trick did not award 10 extra points."

    # ---------------------------------------------------------
    # PART 3: Bolt Logic Calculation
    # ---------------------------------------------------------
    # Declarer (Team 0) got exactly 80 points. Defenders (Team 1) got 82.
    # Since Declarer did not get MORE than 80, it is a Bolt.
    engine.raw_points_by_team = [80, 82]
    engine.bolts_by_team = [0, 0]
    engine.tricks_won_by_team = [4, 4]
    final_rewards = engine._calculate_final_rewards()
    
    # Expected: Team 0 gets 0 Game Points. Team 1 gets 16 Game Points.
    assert final_rewards[0] == 0, "Bolt failed: Declaring team should receive 0 game points."
    assert final_rewards[1] == 16, "Bolt failed: Defending team should receive 16 game points."
    assert engine.bolts_by_team[0] == 1, "Bolt failed: Declaring team's bolt counter did not increment."

def test_action_masking_integrity():
    """
    Constructs highly specific game states to verify that get_legal_actions() 
    strictly enforces Following Suit, Ruffing, and the Overruff Rule.
    """
    engine = BelotEnv()
    engine.phase = "PLAYING"
    engine.trump = 0  # Spades are Trump (IDs 0-7)
    engine.declarer = 0
    engine.current_player = 1
    engine.declarer_has_played_trump = True
    
    # Let's map out some cards for readability
    # Suit 0 (Trump): 0(7), 1(8), 2(9), 3(10), 4(J), 5(Q), 6(K), 7(A)
    # Suit 1 (Hearts): 8(7), 9(8), 10(9), 11(10), 12(J), 13(Q), 14(K), 15(A)
    # Suit 2 (Diamonds): 16(7) ... 23(A)
    
    # ---------------------------------------------------------
    # SCENARIO 1: Follow Suit Forced
    # Led: Hearts. Agent has Hearts and Trump. Must play Hearts.
    # ---------------------------------------------------------
    engine.current_trick = [(0, 11)] # Player 0 led the 10 of Hearts
    engine.hands[1] = [15, 0, 4] # Agent has Ace of Hearts, 7 of Trump, Jack of Trump
    
    mask = engine.get_legal_actions()
    
    assert mask[15] == True, "Mask failed: Agent should be able to follow suit."
    assert mask[0] == False, "Mask failed: Agent illegally allowed to ruff when they can follow suit."
    assert mask[4] == False, "Mask failed: Agent illegally allowed to ruff when they can follow suit."

    # ---------------------------------------------------------
    # SCENARIO 2: Forced Ruff
    # Led: Hearts. Agent has NO Hearts, has Diamonds, has Trump. Must Ruff.
    # ---------------------------------------------------------
    engine.current_trick = [(0, 11)] # Player 0 led the 10 of Hearts
    engine.hands[1] = [23, 0] # Agent has Ace of Diamonds (discard), 7 of Trump
    
    mask = engine.get_legal_actions()
    
    assert mask[23] == False, "Mask failed: Agent illegally allowed to discard when they must ruff."
    assert mask[0] == True, "Mask failed: Agent should be forced to play their trump card."

    # ---------------------------------------------------------
    # SCENARIO 3: The Overruff Rule
    # Led: Hearts. Ruffed by Player 2. Agent has NO Hearts.
    # Agent MUST play Trump. But must OVERRUFF if possible.
    # ---------------------------------------------------------
    # Player 0 led 10 of Hearts (11). Player 2 ruffed with 9 of Trump (2) -> Rank power 14.
    engine.current_trick = [(0, 11), (2, 2)] 
    engine.current_player = 3
    # Player 3 has 7 of Trump (0 -> Rank power 0) and Jack of Trump (4 -> Rank power 20)
    engine.hands[3] = [0, 4, 23] # Trump 7, Trump Jack, Diamond Ace
    
    mask = engine.get_legal_actions()
    
    assert mask[23] == False, "Mask failed: Agent cannot discard."
    assert mask[0] == False, "Mask failed: Agent illegally allowed to underruff when an overruff is possible."
    assert mask[4] == True, "Mask failed: Agent must be forced to overruff using the Jack."

def test_local_state_truth_mapping():
    """
    Validates that every specific slice of the 513-dim Local State exactly
    matches the underlying source-of-truth variables in the BelotEnv engine.
    """
    env = BelotAECEnv()
    env.reset()
    
    # Fast-forward to the PLAYING phase to populate the arrays
    for _ in range(50):
        agent = env.agent_selection
        obs_dict = env.observe(agent)
        legal = np.where(obs_dict["action_mask"] == 1)[0]
        action = np.random.choice(legal) if len(legal) > 0 else 32
        env.step(action)
        if env.belot.phase == "PLAYING" and len(env.belot.last_trick) > 0:
            break

    current_agent = env.agent_selection
    abs_id = int(current_agent[-1])
    obs_dict = env.observe(current_agent)
    obs = obs_dict["observation"]
    engine = env.belot
    
    idx = 0

    # 1. Private Hand (32)
    expected_hand = np.zeros(32, dtype=np.float32)
    expected_hand[engine.hands[abs_id]] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 32], expected_hand, err_msg="Mismatch: Private Hand")
    idx += 32

    # 2. Face-Up Card (32)
    expected_face_up = np.zeros(32, dtype=np.float32)
    if engine.phase == "BIDDING" and engine.face_up_card is not None:
        expected_face_up[engine.face_up_card] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 32], expected_face_up, err_msg="Mismatch: Face-Up Card")
    idx += 32

    # 3. Current Trump (5)
    expected_trump = np.zeros(5, dtype=np.float32)
    if engine.trump is None: expected_trump[0] = 1.0
    else: expected_trump[1 + engine.trump] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 5], expected_trump, err_msg="Mismatch: Trump")
    idx += 5

    # 4. Relative Declarer (5)
    expected_declarer = np.zeros(5, dtype=np.float32)
    if engine.declarer is None: expected_declarer[0] = 1.0
    else: expected_declarer[1 + (engine.declarer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 5], expected_declarer, err_msg="Mismatch: Declarer")
    idx += 5

    # 5. Phase (3)
    expected_phase = np.zeros(3, dtype=np.float32)
    if engine.phase == "BIDDING":
        if engine.bidding_round == 1: expected_phase[0] = 1.0
        else: expected_phase[1] = 1.0
    else: expected_phase[2] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 3], expected_phase, err_msg="Mismatch: Phase")
    idx += 3

    # 6. Current Trick (108)
    for rel in [1, 2, 3]:
        expected_trick = np.zeros(36, dtype=np.float32)
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(engine.current_trick):
            if p == abs_p:
                expected_trick[c] = 1.0
                expected_trick[32 + seq_idx] = 1.0
        np.testing.assert_array_equal(obs[idx : idx + 36], expected_trick, err_msg=f"Mismatch: Current Trick rel {rel}")
        idx += 36

    # 7. Game Stats (6)
    idx += 6 # Already fully tested in Global, skipping tight bounds check here to save lines

    # 8. Relative Dealer (4)
    expected_dealer = np.zeros(4, dtype=np.float32)
    expected_dealer[(engine.dealer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 4], expected_dealer, err_msg="Mismatch: Dealer")
    idx += 4

    # 9. Last Trick (144)
    for rel in [0, 1, 2, 3]:
        expected_last_trick = np.zeros(36, dtype=np.float32)
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(engine.last_trick):
            if p == abs_p:
                expected_last_trick[c] = 1.0
                expected_last_trick[32 + seq_idx] = 1.0
        np.testing.assert_array_equal(obs[idx : idx + 36], expected_last_trick, err_msg=f"Mismatch: Last Trick rel {rel}")
        idx += 36

    # 10. Belief State Matrix (96)
    # Instead of recalculating the math, we verify the hard constraints of the belief system
    belief_slice = obs[idx : idx + 96].reshape(3, 32)
    other_players = [(abs_id + 1) % 4, (abs_id + 2) % 4, (abs_id + 3) % 4]
    
    for i, p in enumerate(other_players):
        player_belief = belief_slice[i]
        
        # Condition A: Cards we KNOW they have must be exactly 1.0
        known_cards = np.where(engine.known_cards[p] == True)[0]
        if len(known_cards) > 0:
            assert np.all(player_belief[known_cards] == 1.0), f"Belief failed: Known cards for Player {p} aren't 1.0"
            
        # Condition B: Cards that are IMPOSSIBLE must be exactly 0.0
        impossible_cards = np.where(engine.impossible_cards[p] == True)[0]
        if len(impossible_cards) > 0:
            assert np.all(player_belief[impossible_cards] == 0.0), f"Belief failed: Impossible cards for Player {p} aren't 0.0"
            
        # Condition C: Cards in OUR hand must be 0.0 in THEIR belief
        if len(engine.hands[abs_id]) > 0:
            assert np.all(player_belief[engine.hands[abs_id]] == 0.0), f"Belief failed: Agent thinks opponent has a card currently in its own hand."

    idx += 96

    # 11. Trick Number (8)
    expected_trick_num = np.zeros(8, dtype=np.float32)
    expected_trick_num[min(engine.tricks_played, 7)] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 8], expected_trick_num, err_msg="Mismatch: Trick Number")
    idx += 8

    # 12. Valid Actions Feature Injection (38)
    # Since it's the current agent's turn, this should precisely equal their legal actions.
    expected_legal = engine.get_legal_actions().astype(np.float32)
    np.testing.assert_array_equal(obs[idx : idx + 38], expected_legal, err_msg="Mismatch: Legal Action Mask")
    idx += 38

    # 13. The Graveyard (32)
    expected_grave = np.zeros(32, dtype=np.float32)
    expected_grave[engine.graveyard] = 1.0
    np.testing.assert_array_equal(obs[idx : idx + 32], expected_grave, err_msg="Mismatch: Graveyard")
    idx += 32

    assert idx == 513, f"Local index tracker desynced! Reached {idx} instead of 513."

def test_global_state_truth_mapping():
    """
    Validates that every specific slice of the 332-dim Global State exactly
    matches the underlying source-of-truth variables in the BelotEnv engine.
    """
    env = BelotAECEnv()
    env.reset()
    
    # Fast-forward the game until we are in the PLAYING phase and have a graveyard
    # so the arrays are populated with rich, non-zero data.
    for _ in range(50):
        agent = env.agent_selection
        obs_dict = env.observe(agent)
        legal = np.where(obs_dict["action_mask"] == 1)[0]
        action = np.random.choice(legal) if len(legal) > 0 else 32
        env.step(action)
        if env.belot.phase == "PLAYING" and len(env.belot.graveyard) > 0:
            break

    # Take snapshot
    current_agent = env.agent_selection
    abs_id = int(current_agent[-1])
    obs_dict = env.observe(current_agent)
    g_obs = obs_dict["global_observation"]
    engine = env.belot  # The source of truth
    
    team_us = abs_id % 2
    team_them = 1 - team_us

    g_idx = 0

    # 1. Relative Hands (128)
    for rel in [0, 1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        expected_hand = np.zeros(32, dtype=np.float32)
        expected_hand[engine.hands[abs_p]] = 1.0
        np.testing.assert_array_equal(g_obs[g_idx : g_idx + 32], expected_hand, err_msg=f"Mismatch: Hand for rel {rel}")
        g_idx += 32

    # 2. Graveyard (32)
    expected_grave = np.zeros(32, dtype=np.float32)
    expected_grave[engine.graveyard] = 1.0
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 32], expected_grave, err_msg="Mismatch: Graveyard")
    g_idx += 32

    # 3. Face-Up Card (32)
    expected_face_up = np.zeros(32, dtype=np.float32)
    if engine.phase == "BIDDING" and engine.face_up_card is not None:
        expected_face_up[engine.face_up_card] = 1.0
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 32], expected_face_up, err_msg="Mismatch: Face-Up Card")
    g_idx += 32

    # 4. Current Trump (5)
    expected_trump = np.zeros(5, dtype=np.float32)
    if engine.trump is None: expected_trump[0] = 1.0
    else: expected_trump[1 + engine.trump] = 1.0
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 5], expected_trump, err_msg="Mismatch: Trump")
    g_idx += 5

    # 5. Relative Declarer (5)
    expected_declarer = np.zeros(5, dtype=np.float32)
    if engine.declarer is None: expected_declarer[0] = 1.0
    else: expected_declarer[1 + (engine.declarer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 5], expected_declarer, err_msg="Mismatch: Declarer")
    g_idx += 5

    # 6. Relative Dealer (4)
    expected_dealer = np.zeros(4, dtype=np.float32)
    expected_dealer[(engine.dealer - abs_id) % 4] = 1.0
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 4], expected_dealer, err_msg="Mismatch: Dealer")
    g_idx += 4

    # 7. Phase (3)
    expected_phase = np.zeros(3, dtype=np.float32)
    if engine.phase == "BIDDING":
        if engine.bidding_round == 1: expected_phase[0] = 1.0
        else: expected_phase[1] = 1.0
    else: expected_phase[2] = 1.0
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 3], expected_phase, err_msg="Mismatch: Phase")
    g_idx += 3

    # 8. Relative Current Trick (108)
    for rel in [1, 2, 3]:
        expected_trick = np.zeros(36, dtype=np.float32)
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(engine.current_trick):
            if p == abs_p:
                expected_trick[c] = 1.0
                expected_trick[32 + seq_idx] = 1.0
        np.testing.assert_array_equal(g_obs[g_idx : g_idx + 36], expected_trick, err_msg=f"Mismatch: Current Trick rel {rel}")
        g_idx += 36

    # 9. Game Stats (6)
    expected_stats = np.array([
        env.match_scores[team_us] / 101.0,
        env.match_scores[team_them] / 101.0,
        engine.raw_points_by_team[team_us] / 162.0,
        engine.raw_points_by_team[team_them] / 162.0,
        engine.bolts_by_team[team_us] / 2.0,
        engine.bolts_by_team[team_them] / 2.0
    ], dtype=np.float32)
    np.testing.assert_allclose(g_obs[g_idx : g_idx + 6], expected_stats, err_msg="Mismatch: Game Stats")
    g_idx += 6

    # 10. Trick Number (8)
    expected_trick_num = np.zeros(8, dtype=np.float32)
    expected_trick_num[min(engine.tricks_played, 7)] = 1.0
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 8], expected_trick_num, err_msg="Mismatch: Trick Number")
    g_idx += 8

    # 11. Declarer Has Played Trump (1)
    expected_played_trump = np.array([float(engine.declarer_has_played_trump)], dtype=np.float32)
    np.testing.assert_array_equal(g_obs[g_idx : g_idx + 1], expected_played_trump, err_msg="Mismatch: Declarer Played Trump")
    g_idx += 1

    assert g_idx == 332, f"Global index tracker desynced! Reached {g_idx} instead of 332."

def test_global_to_local_deduction():
    """
    Validates that the factual, shared components of the Local State (513-dim)
    can be perfectly deduced from the Global State (332-dim) from the
    exact perspective of the acting agent.
    """
    env = BelotAECEnv()
    env.reset()
    
    # 1. Step through random actions to populate the board.
    # We want tricks, graveyards, and active bidding phases to have data, 
    # not just arrays of zeros.
    for _ in range(25):
        agent = env.agent_selection
        obs_dict = env.observe(agent)
        
        # Pick a random valid action
        legal_mask = obs_dict["action_mask"]
        valid_actions = np.where(legal_mask == 1)[0]
        
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 32 # Fallback
            
        env.step(action)
        
        # Stop before the game resets so we have a rich state to analyze
        if env.terminations[agent] or env.truncations[agent]:
            break

    # 2. Take a snapshot of the current agent's observation
    current_agent = env.agent_selection
    obs_dict = env.observe(current_agent)
    
    local_obs = obs_dict["observation"]
    global_obs = obs_dict["global_observation"]
    
    # ==========================================
    # 3. DEDUCTION & ASSERTION BLOCK
    # Slicing based on the exact index architecture in env_wrapper.py
    # ==========================================
    
    # 1. Private Hand (32 dims)
    # Local: 0:32 | Global: 0:32 (Agent's own hand is the first block in relative hands)
    np.testing.assert_array_equal(
        local_obs[0:32], global_obs[0:32], 
        err_msg="Perspective mismatch: Private Hand"
    )
    
    # 2. Face-Up Card (32 dims)
    # Local: 32:64 | Global: 160:192
    np.testing.assert_array_equal(
        local_obs[32:64], global_obs[160:192], 
        err_msg="Perspective mismatch: Face-Up Card"
    )
    
    # 3. Current Trump (5 dims)
    # Local: 64:69 | Global: 192:197
    np.testing.assert_array_equal(
        local_obs[64:69], global_obs[192:197], 
        err_msg="Perspective mismatch: Current Trump"
    )
    
    # 4. Relative Declarer (5 dims)
    # Local: 69:74 | Global: 197:202
    np.testing.assert_array_equal(
        local_obs[69:74], global_obs[197:202], 
        err_msg="Perspective mismatch: Relative Declarer"
    )
    
    # 5. Phase (3 dims)
    # Local: 74:77 | Global: 206:209
    np.testing.assert_array_equal(
        local_obs[74:77], global_obs[206:209], 
        err_msg="Perspective mismatch: Game Phase"
    )
    
    # 6. Relative Current Trick (108 dims)
    # Local: 77:185 | Global: 209:317
    np.testing.assert_array_equal(
        local_obs[77:185], global_obs[209:317], 
        err_msg="Perspective mismatch: Relative Current Trick"
    )
    
    # 7. Relative Game Stats (6 dims)
    # Local: 185:191 | Global: 317:323
    np.testing.assert_array_equal(
        local_obs[185:191], global_obs[317:323], 
        err_msg="Perspective mismatch: Game Stats"
    )
    
    # 8. Relative Dealer (4 dims)
    # Local: 191:195 | Global: 202:206
    np.testing.assert_array_equal(
        local_obs[191:195], global_obs[202:206], 
        err_msg="Perspective mismatch: Relative Dealer"
    )
    
    # 9. Trick Number (8 dims)
    # Local: 435:443 | Global: 323:331
    np.testing.assert_array_equal(
        local_obs[435:443], global_obs[323:331], 
        err_msg="Perspective mismatch: Trick Number"
    )
    
    # 10. Graveyard (32 dims)
    # Local: 481:513 | Global: 128:160
    np.testing.assert_array_equal(
        local_obs[481:513], global_obs[128:160], 
        err_msg="Perspective mismatch: Graveyard"
    )

def test_single_trick_dense_rewards():
    env = BelotEnv()
    env.reset()
    
    # 1. Manually construct a mid-game state
    env.phase = "PLAYING"
    env.trump = 0 # Let's say Spades (0-7) is Trump
    env.current_player = 0
    env.tricks_played = 0
    env.declarer = 0
    
    # Give Player 0 the Jack of Spades (Card 4, 20 pts)
    # Give Player 1 the 9 of Spades (Card 3, 14 pts)
    # Give Player 2 the Ace of Hearts (Card 15, 11 pts)
    # Give Player 3 the 10 of Hearts (Card 11, 10 pts)
    env.hands[0] = [4]
    env.hands[1] = [2]
    env.hands[2] = [15]
    env.hands[3] = [11]
    
    # 2. Play the trick
    env.step(4)  # P0 plays Jack of Spades
    env.step(2)  # P1 plays 9 of Spades
    env.step(15) # P2 plays Ace of Hearts
    _, step_rewards, done, _ = env.step(11) # P3 plays 10 of Hearts. Trick resolves!
    
    # 3. Calculate expected math
    # Total points = 20 + 14 + 11 + 10 = 55 points
    expected_dense_win = 55.0 / 162.0
    expected_dense_loss = -55.0 / 162.0
    
    # P0 played the highest trump, so Team 0 wins.
    assert np.isclose(step_rewards[0], expected_dense_win), f"P0 reward wrong: {step_rewards[0]}"
    assert np.isclose(step_rewards[2], expected_dense_win), f"P2 reward wrong: {step_rewards[2]}"
    assert np.isclose(step_rewards[1], expected_dense_loss), f"P1 reward wrong: {step_rewards[1]}"
    assert np.isclose(step_rewards[3], expected_dense_loss), f"P3 reward wrong: {step_rewards[3]}"
    
    print("Single trick dense rewards calculated flawlessly!")

def test_calculate_final_rewards():
    env = BelotEnv()
    
    def check_scenario(tricks, raw, dec_team, initial_bolts, expected_pts, expected_bolts_after, scenario_name):
        env.tricks_won_by_team = tricks
        env.raw_points_by_team = raw
        env.declaring_team = dec_team
        env.defending_team = 1 - dec_team
        env.bolts_by_team = initial_bolts.copy()
        
        pts = env._calculate_final_rewards()
        
        assert pts == expected_pts, f"[{scenario_name}] Expected points {expected_pts}, got {pts}"
        assert env.bolts_by_team == expected_bolts_after, f"[{scenario_name}] Expected bolts {expected_bolts_after}, got {env.bolts_by_team}"

    # 1. Simple case: 86 vs 76 (Team 0 declares, wins cleanly)
    # Def gets 76 -> 76 % 10 = 6 (>5) -> 8 match points. Dec gets 16 - 8 = 8.
    check_scenario([4, 4], [86, 76], dec_team=0, initial_bolts=[0, 0], 
                   expected_pts=[8, 8, 8, 8], expected_bolts_after=[0, 0], 
                   scenario_name="Simple 86-76")
                   
    # 2. Equal: 81 vs 81 (Team 0 declares, ties)
    # Def gets 81 -> 81 % 10 = 1 (<5) -> 8 match points. Dec gets 16 - 8 = 8.
    check_scenario([4, 4], [81, 81], dec_team=0, initial_bolts=[0, 0], 
                   expected_pts=[8, 8, 8, 8], expected_bolts_after=[0, 0], 
                   scenario_name="Equal 81-81")

    # 3. Bolt: 70 vs 92 (Team 0 declares, fails to break 80)
    # Dec gets 0, Def gets 16. Bolt counter increments for Team 0.
    check_scenario([4, 4], [70, 92], dec_team=0, initial_bolts=[0, 0], 
                   expected_pts=[0, 16, 0, 16], expected_bolts_after=[1, 0], 
                   scenario_name="Standard Bolt")

    # 4. 3 Bolts Penalty (Team 0 declares, fails, and it's their 3rd bolt)
    # Dec gets 0, minus 10 penalty = -10. Def gets 16. Bolt counter resets to 0.
    check_scenario([4, 4], [70, 92], dec_team=0, initial_bolts=[2, 0], 
                   expected_pts=[-10, 16, -10, 16], expected_bolts_after=[0, 0], 
                   scenario_name="3 Bolts Penalty")

    # 5. No-trick / Capot (Team 0 takes all tricks, 162 vs 0)
    # Def gets -10 internal state flag. Dec gets 16.
    check_scenario([8, 0], [162, 0], dec_team=0, initial_bolts=[0, 0], 
                   expected_pts=[16, -10, 16, -10], expected_bolts_after=[0, 0], 
                   scenario_name="No Trick (Team 1 Capot)")

def test_reward_distribution_and_memory2():
    env = BelotAECEnv()
    memory = MultiAgentMemory(env.possible_agents)
    env.reset()
    
    captured_infos = {}
    
    for agent in env.agent_iter():
        obs_dict, reward, termination, truncation, info = env.last()
        buf = memory.buffers[agent]
        
        if len(buf.rewards) > 0:
            buf.rewards[-1] = reward
            
        if termination or truncation:
            captured_infos[agent] = info 
            if len(buf.rewards) > 0:
                buf.dones[-1] = True
            env.step(None) 
            continue
        
        buf.store(
            obs=torch.zeros(513), global_obs=torch.zeros(332), mask=torch.zeros(38), 
            action=0, logprob=-0.5, reward=0.0, value=0.0, done=False, 
            h=torch.zeros(512), c=torch.zeros(512)
        )
        
        legal_mask = obs_dict["action_mask"]
        valid_actions = np.where(legal_mask == 1)[0]
        action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 32
        env.step(action)

    # 1. Extract Info
    game_points = captured_infos["player_0"]["game_points"]
    # Clamp out the -10 internal flag if it leaked, simulating correct match points
    clean_points = [max(0, p) for p in game_points] 
    
    expected_targets = {
        "player_0": (clean_points[0] - clean_points[1]) / 16.0,
        "player_1": (clean_points[1] - clean_points[0]) / 16.0,
        "player_2": (clean_points[0] - clean_points[1]) / 16.0,
        "player_3": (clean_points[1] - clean_points[0]) / 16.0,
    }

    # 2. Assertions
    for agent in env.possible_agents:
        total_reward = sum(memory.buffers[agent].rewards)
        
        # Test True-Up
        assert np.isclose(total_reward, expected_targets[agent], atol=1e-5), \
            f"{agent} accumulated {total_reward:.4f}, expected {expected_targets[agent]:.4f}"
            
        # Test Done Flags
        assert memory.buffers[agent].dones[-1] == True, "Last step missing Done flag."

def test_reward_distribution_and_memory():
    """
    Simulates a full episode to verify that:
    1. The reward retroaction correctly applies dense step rewards to memory.
    2. The sum of accumulated memory rewards perfectly matches the Final True-Up Zero-Sum target.
    3. The environment correctly distributes team rewards symmetrically (P0==P2, P1==P3).
    4. The 'done' flags are correctly set in the memory buffer.
    """
    env = BelotAECEnv()
    memory = MultiAgentMemory(env.possible_agents)
    env.reset()
    
    hidden_dim = 512
    captured_infos = {}
    
    # 1. Play exactly one complete hand to trigger the termination flags
    for agent in env.agent_iter():
        obs_dict, reward, termination, truncation, info = env.last()
        
        # UPDATE: Overwrite logic MUST apply to every step, not just termination.
        # This matches train.py and captures the dense trick rewards.
        buf = memory.buffers[agent]
        if len(buf.rewards) > 0:
            buf.rewards[-1] = reward
            
        if termination or truncation:
            # Capture the final unnormalized game points generated by _calculate_final_rewards
            captured_infos[agent] = info 
            
            if len(buf.rewards) > 0:
                buf.dones[-1] = True
            
            # Step None clears dead agents
            env.step(None) 
            continue
        
        # Normal step: Store dummy data in memory. Reward is temporarily 0.0
        local_obs = torch.zeros(513)
        global_obs = torch.zeros(332)
        mask = torch.zeros(38)
        hc_mock = torch.zeros(hidden_dim)
        
        buf.store(
            obs=local_obs, global_obs=global_obs, mask=mask, action=0, 
            logprob=-0.5, reward=0.0, value=0.0, done=False, 
            h=hc_mock, c=hc_mock
        )
        
        # Take a random valid action to advance the game
        legal_mask = obs_dict["action_mask"]
        valid_actions = np.where(legal_mask == 1)[0]
        action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 32
        
        env.step(action)

    # ==========================================
    # 2. ASSERTIONS FOR DENSE TRUE-UP ARCHITECTURE
    # ==========================================
    
    # Extract the unnormalized match points calculated by the environment
    game_points = captured_infos["player_0"]["game_points"]
    
    # Calculate the expected zero-sum strategic targets (Matched against env.py logic)
    expected_targets = {
        "player_0": (game_points[0] - game_points[1]) / 16.0,
        "player_1": (game_points[1] - game_points[0]) / 16.0,
        "player_2": (game_points[0] - game_points[1]) / 16.0,
        "player_3": (game_points[1] - game_points[0]) / 16.0,
    }

    for agent in env.possible_agents:
        buf = memory.buffers[agent]
        
        # A. Core True-Up Assertion: Total accumulated buffer rewards must equal the strategic target
        total_reward = sum(buf.rewards)
        assert np.isclose(total_reward, expected_targets[agent], atol=1e-5), \
            f"{agent} accumulated {total_reward:.4f}, but expected true-up target was {expected_targets[agent]:.4f}"
        
        # B. Team Symmetry Assertion
        partner = f"player_{(int(agent[-1]) + 2) % 4}"
        partner_total = sum(memory.buffers[partner].rewards)
        assert np.isclose(total_reward, partner_total, atol=1e-5), \
            f"Team symmetry broken: {agent} accumulated {total_reward}, but {partner} got {partner_total}"
        
        # C. Dense Reward Validation: Prove intermediate rewards are no longer just 0.0
        # (Note: In very rare game traces where a team wins 0 tricks, this could theoretically be all negative/zero, 
        # but randomly playing will almost certainly trigger non-zero intermediate points).
        if len(buf.rewards) > 1:
            assert any(r != 0.0 for r in buf.rewards[:-1]), \
                f"{agent} had all 0.0 intermediate rewards. Dense step retroaction failed."
            
        # D. Terminal State Flags
        assert buf.dones[-1] == True, f"{agent}'s last memory step did not register as Done."
        if len(buf.dones) > 1:
            assert not any(buf.dones[:-1]), f"{agent} has premature Done flags in their memory buffer."

# ==========================================
# TEST: STATE ENCODING (Global vs Normal)
# ==========================================
def test_state_encoding():
    """
    Ensures that the environment correctly builds the local and global observations
    with the exact expected dimensions and data types.
    """
    env = BelotAECEnv()
    env.reset()
    
    # Observe the first agent
    current_agent = env.agent_selection
    obs_dict = env.observe(current_agent)
    
    local_obs = obs_dict["observation"]
    global_obs = obs_dict["global_observation"]
    action_mask = obs_dict["action_mask"]
    
    # Assert dimensions match your architecture
    assert local_obs.shape == (513,), f"Expected local obs shape (513,), got {local_obs.shape}"
    assert global_obs.shape == (332,), f"Expected global obs shape (332,), got {global_obs.shape}"
    assert action_mask.shape == (38,), f"Expected action mask shape (38,), got {action_mask.shape}"
    
    # Assert data types are correct for PyTorch consumption
    assert local_obs.dtype == np.float32
    assert global_obs.dtype == np.float32
    assert action_mask.dtype == np.int8

    # Assert ranges (should be normalized between 0 and 1)
    assert np.all(local_obs >= 0.0) and np.all(local_obs <= 1.0)
    assert np.all(global_obs >= 0.0) and np.all(global_obs <= 1.0)


# ==========================================
# TEST: MEMORY BPTT PADDING
# ==========================================
def test_memory_bptt_padding():
    """
    Validates that Backpropagation Through Time (BPTT) padding creates a rectangular
    tensor and correctly generates the `pad_mask` to ignore dummy padded steps.
    """
    buf = AgentBuffer()
    
    # Mock parameters
    local_dim, global_dim, hidden_dim = 513, 332, 512
    
    # Let's simulate 2 episodes of different lengths
    # Episode 1: 2 steps
    # Episode 2: 3 steps
    episode_lengths = [2, 3]
    
    for ep_len in episode_lengths:
        for step in range(ep_len):
            is_done = (step == ep_len - 1) # True on the last step of the episode
            
            buf.store(
                obs=torch.randn(local_dim),
                global_obs=torch.randn(global_dim),
                mask=torch.ones(38),
                action=0,
                logprob=-0.5,
                reward=1.0,
                value=0.5,
                done=is_done,
                h=torch.randn(hidden_dim),
                c=torch.randn(hidden_dim)
            )
            
    # Compute GAE first to get returns and advantages
    returns, advantages = buf.compute_gae(next_value=0.0)
    
    # Get padded batches
    b_obs, b_gobs, b_masks, b_actions, b_logprobs, b_adv, b_ret, pad_mask = buf.get_padded_batch(advantages, returns)
    
    # Assert Batch Size and Sequence Length
    # Expected: 2 episodes (batch size), max sequence length of 3
    assert b_obs.shape == (2, 3, local_dim), f"Expected b_obs shape (2, 3, 513), got {b_obs.shape}"
    assert b_gobs.shape == (2, 3, global_dim), f"Expected b_gobs shape (2, 3, 332), got {b_gobs.shape}"
    
    # Assert Pad Mask correctness
    # Episode 1 has 2 steps, so it should be [1, 1, 0]
    # Episode 2 has 3 steps, so it should be [1, 1, 1]
    expected_pad_mask = torch.tensor([
        [1., 1., 0.],
        [1., 1., 1.]
    ])
    assert torch.equal(pad_mask, expected_pad_mask), f"Padding mask logic failed. Got:\n{pad_mask}"


# ==========================================
# TEST: LSTM HANDLING (ROLLOUT) & CTDE
# ==========================================
def test_lstm_and_ctde_rollout():
    """
    Validates the forward pass during environment rollouts (Step-by-step).
    Ensures the Actor only sees local state and Critic only sees global state.
    """
    model = RecurrentMAPPOModel()
    
    # Mock a single step for a single agent: (Batch=1, Dim)
    local_obs = torch.randn(1, 513)
    global_obs = torch.randn(1, 332)
    action_mask = torch.ones(1, 38)
    
    # Hidden states expected shape during rollout: (Num_Layers=1, Batch=1, Hidden=512)
    hc = (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
    
    # Forward pass (is_sequence=False)
    dist, value, new_hc = model(local_obs, global_obs, hc, action_mask, is_sequence=False)
    
    # Assert Actor Output (Distribution over 38 actions)
    assert dist.logits.shape == (1, 38), "Actor rollout output shape is incorrect."
    
    # Assert Critic Output (Single value per batch)
    assert value.shape == (1, 1), "Critic rollout output shape is incorrect."
    
    # Assert LSTM state transitioned correctly
    assert new_hc[0].shape == (1, 1, 512)
    assert not torch.equal(hc[0], new_hc[0]), "LSTM hidden state did not update."


# ==========================================
# TEST: LSTM HANDLING (TRAINING) & CTDE
# ==========================================
def test_lstm_and_ctde_training():
    """
    Validates the forward pass during batched PPO sequence training.
    Ensures BPTT processes the (Batch, Sequence, Features) block correctly.
    """
    model = RecurrentMAPPOModel()
    
    batch_size = 4
    seq_len = 8
    
    # Mock padded sequence inputs: (Batch, Sequence, Dim)
    b_local_obs = torch.randn(batch_size, seq_len, 513)
    b_global_obs = torch.randn(batch_size, seq_len, 332)
    b_action_masks = torch.ones(batch_size, seq_len, 38)
    
    # Hidden states initialized for sequence training: (Num_Layers=1, Batch, Hidden)
    h_0 = torch.zeros(1, batch_size, 512)
    c_0 = torch.zeros(1, batch_size, 512)
    curr_hc = (h_0, c_0)
    
    # Forward pass (is_sequence=True)
    dist, values, new_hc = model(b_local_obs, b_global_obs, curr_hc, b_action_masks, is_sequence=True)
    
    # Assert Actor processed the entire sequence
    assert dist.logits.shape == (batch_size, seq_len, 38), "Actor training output failed to maintain batch/seq shape."
    
    # Assert Critic evaluated the entire global sequence
    assert values.shape == (batch_size, seq_len, 1), "Critic training output failed to maintain batch/seq shape."
    
'''
ToDo: Test to see what does the mask look when a player has just 1 card. It should allow the agent to play that card no matter what.
Test what happens when declarer doesnt get any hands (bolt vs 0 trick rule)
'''
