"""
Single source of truth for observation construction.

This is a faithful, side-effect-free port of `BelotAECEnv.observe()`. It takes a
raw `BelotEnv` plus the per-env running match score and returns the three arrays
the policy needs. Keeping it standalone lets the vectorized driver build a batch
of observations without spinning up the full PettingZoo AEC machinery per env.
"""

import numpy as np


def build_observation(belot, abs_id, match_scores):
    """
    Build the local (513,), egocentric global (332,) observations and the
    legal-action mask (38,) for player `abs_id` in `belot`.

    `match_scores` is this env's running [team0, team1] match score.
    """
    team_us = abs_id % 2
    team_them = 1 - team_us

    # ====================== LOCAL OBSERVATION (513) ======================
    obs = np.zeros(513, dtype=np.float32)
    idx = 0

    # 1. Private Hand (32)
    for card in belot.hands[abs_id]:
        obs[idx + card] = 1.0
    idx += 32

    # 2. Face-Up Card (32)
    if belot.face_up_card is not None and belot.phase == "BIDDING":
        obs[idx + belot.face_up_card] = 1.0
    idx += 32

    # 3. Current Trump (5)
    if belot.trump is None:
        obs[idx] = 1.0
    else:
        obs[idx + 1 + belot.trump] = 1.0
    idx += 5

    # 4. Relative Declarer (5)
    if belot.declarer is None:
        obs[idx] = 1.0
    else:
        obs[idx + 1 + (belot.declarer - abs_id) % 4] = 1.0
    idx += 5

    # 5. Phase (3)
    if belot.phase == "BIDDING":
        obs[idx] = 1.0 if belot.bidding_round == 1 else 0.0
        if belot.bidding_round != 1:
            obs[idx + 1] = 1.0
    else:
        obs[idx + 2] = 1.0
    idx += 3

    # 6. Current Trick (108): Left(1), Partner(2), Right(3)
    for rel in [1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(belot.current_trick):
            if p == abs_p:
                obs[idx + c] = 1.0
                obs[idx + 32 + seq_idx] = 1.0
        idx += 36

    # 7. Game Stats (6)
    obs[idx]     = match_scores[team_us] / 101.0
    obs[idx + 1] = match_scores[team_them] / 101.0
    obs[idx + 2] = belot.raw_points_by_team[team_us] / 162.0
    obs[idx + 3] = belot.raw_points_by_team[team_them] / 162.0
    obs[idx + 4] = belot.bolts_by_team[team_us] / 2.0
    obs[idx + 5] = belot.bolts_by_team[team_them] / 2.0
    idx += 6

    # 8. Relative Dealer (4)
    obs[idx + (belot.dealer - abs_id) % 4] = 1.0
    idx += 4

    # 9. Last Trick (144): Me(0), Left(1), Partner(2), Right(3)
    for rel in [0, 1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(belot.last_trick):
            if p == abs_p:
                obs[idx + c] = 1.0
                obs[idx + 32 + seq_idx] = 1.0
        idx += 36

    # 10. Belief State Matrix (96)
    unseen = np.ones(32, dtype=bool)
    unseen[belot.hands[abs_id]] = False
    unseen[belot.graveyard] = False
    for p, c in belot.current_trick:
        unseen[c] = False
    if belot.phase == "BIDDING" and belot.face_up_card is not None:
        unseen[belot.face_up_card] = False

    other_players = [(abs_id + 1) % 4, (abs_id + 2) % 4, (abs_id + 3) % 4]

    # FIX #1: a card KNOWN to sit in some specific hand is not a candidate for
    # ANY unresolved slot -- the original code only excluded it from the known
    # holder's own row, leaving ~1.0 of phantom mass on the other two rows.
    known_any = (belot.known_cards[other_players[0]]
                 | belot.known_cards[other_players[1]]
                 | belot.known_cards[other_players[2]])

    W = np.zeros((3, 32), dtype=np.float32)
    for i, p in enumerate(other_players):
        valid = unseen & ~belot.impossible_cards[p] & ~known_any
        W[i, valid] = 1.0

    remaining = np.array(
        [len(belot.hands[p]) - belot.known_cards[p].sum() for p in other_players],
        dtype=np.float32,
    )

    # FIX #2: iterative proportional fitting instead of one row pass + clip.
    # Row sums must equal each opponent's unresolved hand-slot count; column
    # sums must not exceed 1 (during PLAYING they converge to exactly 1, during
    # BIDDING the slack is the face-down talon). The original single pass left
    # columns off by up to ~15% and clip() saturated non-certain entries at 1.0.
    P = W.copy()
    for _ in range(6):
        rs = P.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        P = P * (remaining[:, np.newaxis] / rs)
        cs = P.sum(axis=0, keepdims=True)
        P = P * np.where(cs > 1.0, 1.0 / np.maximum(cs, 1e-8), 1.0)
    P = np.clip(P, 0.0, 1.0)

    for i, p in enumerate(other_players):
        belief = np.zeros(32, dtype=np.float32)
        belief[belot.known_cards[p]] = 1.0
        unresolved = ~belot.known_cards[p]
        belief[unresolved] = P[i, unresolved]
        obs[idx:idx + 32] = belief
        idx += 32

    # 11. Trick Number (8)
    obs[idx + min(belot.tricks_played, 7)] = 1.0
    idx += 8

    # Legal-action mask (real for the active player, zeros otherwise)
    if belot.current_player == abs_id and not belot.done:
        legal_mask = belot.get_legal_actions().astype(np.int8)
    else:
        legal_mask = np.zeros(38, dtype=np.int8)

    # 12. Valid Actions feature (38)
    obs[idx:idx + 38] = legal_mask.astype(np.float32)
    idx += 38

    # 13. Graveyard (32)
    for c in belot.graveyard:
        obs[idx + c] = 1.0
    idx += 32

    assert idx == 513, f"Local obs built {idx} features, expected 513"

    # ====================== GLOBAL OBSERVATION (332) ======================
    g_obs = np.zeros(332, dtype=np.float32)
    g = 0

    # 1. Relative Hands (128): Me, Left, Partner, Right
    for rel in [0, 1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for card in belot.hands[abs_p]:
            g_obs[g + card] = 1.0
        g += 32

    # 2. Graveyard (32)
    for c in belot.graveyard:
        g_obs[g + c] = 1.0
    g += 32

    # 3. Face-Up Card (32)
    if belot.phase == "BIDDING" and belot.face_up_card is not None:
        g_obs[g + belot.face_up_card] = 1.0
    g += 32

    # 4. Current Trump (5)
    if belot.trump is None:
        g_obs[g] = 1.0
    else:
        g_obs[g + 1 + belot.trump] = 1.0
    g += 5

    # 5. Relative Declarer (5)
    if belot.declarer is None:
        g_obs[g] = 1.0
    else:
        g_obs[g + 1 + (belot.declarer - abs_id) % 4] = 1.0
    g += 5

    # 6. Relative Dealer (4)
    g_obs[g + (belot.dealer - abs_id) % 4] = 1.0
    g += 4

    # 7. Phase (3)
    if belot.phase == "BIDDING":
        if belot.bidding_round != 1:
            g_obs[g + 1] = 1.0
        else:
            g_obs[g] = 1.0
    else:
        g_obs[g + 2] = 1.0
    g += 3

    # 8. Relative Current Trick (108): Left, Partner, Right
    for rel in [1, 2, 3]:
        abs_p = (abs_id + rel) % 4
        for seq_idx, (p, c) in enumerate(belot.current_trick):
            if p == abs_p:
                g_obs[g + c] = 1.0
                g_obs[g + 32 + seq_idx] = 1.0
        g += 36

    # 9. Relative Game Stats (6)
    g_obs[g]     = match_scores[team_us] / 101.0
    g_obs[g + 1] = match_scores[team_them] / 101.0
    g_obs[g + 2] = belot.raw_points_by_team[team_us] / 162.0
    g_obs[g + 3] = belot.raw_points_by_team[team_them] / 162.0
    g_obs[g + 4] = belot.bolts_by_team[team_us] / 2.0
    g_obs[g + 5] = belot.bolts_by_team[team_them] / 2.0
    g += 6

    # 10. Trick Number (8)
    g_obs[g + min(belot.tricks_played, 7)] = 1.0
    g += 8

    # 11. Declarer Has Played Trump (1)
    g_obs[g] = float(belot.declarer_has_played_trump)
    g += 1

    assert g == 332, f"Global obs built {g} features, expected 332"

    return obs, g_obs, legal_mask