"""
AUDIT 5 -- Calibrating "what win rate should a competent policy get vs random?"

A ~40-line greedy heuristic (no search, no card counting, no partner modelling):
  bidding : accept/declare only with real trump strength, else pass
  playing : if you can take the trick, take it with the cheapest sufficient card;
            otherwise dump the lowest-value card; mildly prefers not to waste
            trumps.

Team 0 = heuristic, Team 1 = uniform random legal, dealer rotates.
This bounds the eval scale: the trained agent should land at or above this.
Also prints random-vs-random as a sanity 50% check.
"""
import sys
import numpy as np

sys.path.insert(0, '.')  # run from the project root
from env import BelotEnv

NT_POWER = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 3: 6, 7: 7}
T_POWER  = {0: 0, 1: 1, 5: 2, 6: 3, 3: 4, 7: 5, 2: 6, 4: 7}
T_PTS    = {0: 0, 1: 0, 2: 14, 3: 10, 4: 20, 5: 3, 6: 4, 7: 11}
NT_PTS   = {0: 0, 1: 0, 2: 0, 3: 10, 4: 2, 5: 3, 6: 4, 7: 11}


def power(c, trump):
    return (1, T_POWER[c % 8]) if c // 8 == trump else (0, NT_POWER[c % 8])


def pts(c, trump):
    return T_PTS[c % 8] if c // 8 == trump else NT_PTS[c % 8]


def suit_strength(hand, s, extra=None):
    cards = [c for c in hand if c // 8 == s] + ([extra] if extra is not None and extra // 8 == s else [])
    return len(cards), sum(T_PTS[c % 8] for c in cards)


def heuristic_action(env):
    legal = np.flatnonzero(env.get_legal_actions())
    hand = env.hands[env.current_player]

    if env.phase == "BIDDING":
        if 33 in legal:  # round 1: accepting also wins you the face-up card
            n, p = suit_strength(hand, env.face_up_suit, extra=env.face_up_card)
            return 33 if (n >= 3 and p >= 20) or n >= 4 else 32
        suits = [a - 34 for a in legal if a >= 34]
        best = max(suits, key=lambda s: suit_strength(hand, s))
        n, p = suit_strength(hand, best)
        if 32 in legal and not (n >= 3 and p >= 20):
            return 32
        return 34 + best

    cards = [c for c in legal]
    trick = env.current_trick
    if not trick:  # leading: highest non-trump power, keep trumps unless strong
        def lead_key(c):
            is_t = c // 8 == env.trump
            return (power(c, env.trump)[1] - (3 if is_t and T_POWER[c % 8] < 6 else 0))
        return max(cards, key=lead_key)

    cur_best = max((power(c, env.trump) + (0,) for _, c in trick))[:2] if trick else (-1, -1)
    cur_winner = max(trick, key=lambda pc: power(pc[1], env.trump))[0]
    partner_winning = (cur_winner % 2) == (env.current_player % 2)
    winners = [c for c in cards if power(c, env.trump) > cur_best]

    if partner_winning and len(trick) >= 2:
        # feed points if safe-ish, else dump cheapest
        return max(cards, key=lambda c: (pts(c, env.trump), -power(c, env.trump)[1])) \
            if len(trick) == 3 else min(cards, key=lambda c: (pts(c, env.trump), power(c, env.trump)[1]))
    if winners:
        if len(trick) == 3:  # last to speak: cheapest guaranteed winner
            return min(winners, key=lambda c: (power(c, env.trump)[1],))
        return max(winners, key=lambda c: power(c, env.trump)[1])  # play strength early
    return min(cards, key=lambda c: (pts(c, env.trump), power(c, env.trump)[1]))


def play_match(n_games, team0_policy, seed):
    rng = np.random.default_rng(seed)
    wins, diffs = 0, []
    for g in range(n_games):
        env = BelotEnv()
        env.dealer = g % 4          # rotate dealer to remove positional bias
        env.reset()
        info = {}
        while not env.done:
            if env.current_player % 2 == 0 and team0_policy is not None:
                a = team0_policy(env)
            else:
                a = int(rng.choice(np.flatnonzero(env.get_legal_actions())))
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        wins += gp[0] > gp[1]
        diffs.append(gp[0] - gp[1])
    return wins / n_games, float(np.mean(diffs))


if __name__ == "__main__":
    np.random.seed(0)
    w, d = play_match(3000, None, seed=1)
    print(f"random  vs random : win={w:.1%}  avg point diff={d:+.2f}   (sanity ~50%)")
    w, d = play_match(3000, heuristic_action, seed=2)
    print(f"greedy heuristic vs random : win={w:.1%}  avg point diff={d:+.2f}")
    print("\nInterpretation: this is the bar a *shallow* hand-written policy sets.")
    print("A trained model at ~60% vs random is far below even this baseline.")
