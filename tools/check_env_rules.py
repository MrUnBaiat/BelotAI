"""
AUDIT 4 -- Game-rule invariants over mass random playouts.

Checks per game:
  I1: legal mask never empty before done
  I2: independently re-derived trick winner == env's winner (independent power tables)
  I3: raw_points_by_team sums to exactly 162 (152 card points + 10 pasledu)
  I4: hands empty & graveyard == 32 at done
  I5: game_points branch consistency (standard: sum 16; bolt: 0/16; zero-trick: -10 cases)
  I6: independent re-implementation of get_legal_actions() matches the env's mask
Also reports behavioural stats relevant to "irrational-looking" play:
  - how often a non-declarer is forbidden from leading trump (trump-lock rule)
  - how often a player is FORCED to ruff while their partner is already winning
    (this variant has no "partner is master" exemption)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from belot.env import BelotEnv

NT_POWER = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 3: 6, 7: 7}
T_POWER  = {0: 0, 1: 1, 5: 2, 6: 3, 3: 4, 7: 5, 2: 6, 4: 7}


def ref_winner(trick, trump):
    led = trick[0][1] // 8
    best_p, key = None, (-1, -1)
    for p, c in trick:
        s = c // 8
        if s == trump:
            k = (1, T_POWER[c % 8])
        elif s == led:
            k = (0, NT_POWER[c % 8])
        else:
            k = (-1, -1)
        if k > key:
            key, best_p = k, p
    return best_p


def ref_legal(env):
    """Independent re-implementation of the variant's stated rules."""
    legal = np.zeros(38, dtype=bool)
    if env.done:
        return legal
    if env.phase == "BIDDING":
        if not (env.bidding_round == 2 and env.current_player == env.dealer):
            legal[32] = True
        if env.bidding_round == 1:
            legal[33] = True
        else:
            for s in range(4):
                if s != env.face_up_suit:
                    legal[34 + s] = True
        return legal

    hand = env.hands[env.current_player]
    trump = env.trump
    if len(env.current_trick) == 0:
        all_trump = all(c // 8 == trump for c in hand)
        may_lead_trump = (env.declarer_has_played_trump
                          or env.current_player == env.declarer or all_trump)
        for c in hand:
            if c // 8 == trump and not may_lead_trump:
                continue
            legal[c] = True
        return legal

    led = env.current_trick[0][1] // 8
    in_led = [c for c in hand if c // 8 == led]
    trumps = [c for c in hand if c // 8 == trump]
    trick_trumps = [c for _, c in env.current_trick if c // 8 == trump]
    hi = max((T_POWER[c % 8] for c in trick_trumps), default=-1)
    over = [c for c in trumps if T_POWER[c % 8] > hi]

    if in_led:
        if led == trump and over:
            for c in over: legal[c] = True
        else:
            for c in in_led: legal[c] = True
    elif trumps:
        for c in (over if over else trumps):
            legal[c] = True
    else:
        for c in hand:
            legal[c] = True
    return legal


def main():
    np.random.seed(11)
    n_games = 3000
    fails = {k: 0 for k in ["I1", "I2", "I3", "I4", "I5", "I6"]}
    stats = dict(bolts=0, zero_trick=0, lead_locked=0, lead_states=0,
                 forced_ruff_vs_partner=0, ruff_states=0, declarer_wins=0,
                 hands=0, bid2=0)

    for _ in range(n_games):
        env = BelotEnv()
        tricks_seen = 0
        info = {}
        while not env.done:
            m_env = env.get_legal_actions()
            if not m_env.any():
                fails["I1"] += 1; break
            m_ref = ref_legal(env)
            if not np.array_equal(m_env, m_ref):
                fails["I6"] += 1
                print("LEGALITY MISMATCH", env.phase, env.current_trick,
                      sorted(env.hands[env.current_player]), np.flatnonzero(m_env),
                      np.flatnonzero(m_ref))
            # behavioural stats
            if env.phase == "PLAYING":
                p = env.current_player
                if len(env.current_trick) == 0:
                    stats["lead_states"] += 1
                    has_tr = any(c // 8 == env.trump for c in env.hands[p])
                    if has_tr and not any(m_env[c] for c in env.hands[p] if c // 8 == env.trump):
                        stats["lead_locked"] += 1
                else:
                    led = env.current_trick[0][1] // 8
                    if (not any(c // 8 == led for c in env.hands[p])
                            and any(c // 8 == env.trump for c in env.hands[p])):
                        stats["ruff_states"] += 1
                        if ref_winner(env.current_trick, env.trump) % 2 == p % 2:
                            stats["forced_ruff_vs_partner"] += 1
            legal = np.flatnonzero(m_env)
            _, _, done, info = env.step(int(np.random.choice(legal)))
            if env.tricks_played > tricks_seen:
                tricks_seen = env.tricks_played
                if ref_winner(env.last_trick, env.trump) != env.current_player:
                    fails["I2"] += 1
        # terminal checks
        if sum(env.raw_points_by_team) != 162: fails["I3"] += 1
        if any(env.hands) or len(env.graveyard) != 32: fails["I4"] += 1
        gp = info.get("game_points", [0, 0, 0, 0])
        s = gp[0] + gp[1]
        if s not in (16, 6, -4):        # standard / one-side zero-trick / +3rd-bolt overlap
            fails["I5"] += 1
        stats["hands"] += 1
        if min(env.tricks_won_by_team) == 0: stats["zero_trick"] += 1
        dec = env.declaring_team
        if env.raw_points_by_team[dec] <= 80: stats["bolts"] += 1
        if gp[dec] > gp[1 - dec]: stats["declarer_wins"] += 1

    print("Invariant failures over", n_games, "random games:", fails)
    print("\nBehavioural stats (uniform-random play):")
    print(f"  declarer bolts (raw<=80):            {stats['bolts']/stats['hands']:.1%}")
    print(f"  declarer team wins the hand:         {stats['declarer_wins']/stats['hands']:.1%}")
    print(f"  zero-trick hands:                    {stats['zero_trick']/stats['hands']:.1%}")
    print(f"  lead states with trump-lead LOCKED:  {stats['lead_locked']/stats['lead_states']:.1%}")
    print(f"  forced to ruff while PARTNER winning:{stats['forced_ruff_vs_partner']/max(stats['ruff_states'],1):.1%} "
          f"of void-with-trump states")
    ok = all(v == 0 for v in fails.values())
    print("\nRESULT:", "PASS -- env rules internally consistent" if ok
          else "FAIL -- see counters above")


if __name__ == "__main__":
    main()
