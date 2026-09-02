"""
The reward identity -- the invariant every learned value in this project rests on.

Rewards are dense (a share of each trick's card points) but the objective is the
GAME-point difference, and the two do not agree hand by hand: 162 raw points map
non-linearly onto 16 game points, and a declaring team on 80 or fewer scores zero.
`env.py` reconciles them with a terminal true-up that corrects each seat's dense
rewards so the episode sums to the real outcome.

    sum over an episode of a seat's rewards  ==  (gp_us - gp_them) / 16

If that identity ever breaks, every advantage, every return and every value target
is quietly measuring something other than winning -- and nothing downstream would
tell you, because the losses would still go down. So it is asserted directly rather
than assumed, at ~1e-16, over whole hands driven by several different policies.

Usage:
    python tools/check_reward_identity.py [N_HANDS]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from belot.env import BelotEnv
from belot.heuristic import _heuristic_action, _random_action

TOL = 1e-9


def check(action_fn, label, n, seed0):
    """Play `n` hands and return the worst deviation from the identity."""
    st = np.random.get_state()
    worst, worst_hand = 0.0, -1
    for g in range(n):
        env = BelotEnv()
        env.dealer = g % 4
        np.random.seed(seed0 + g)
        env.reset()
        env.bolts_by_team = [0, 0]
        acc = np.zeros(4)
        info = {}
        while not env.done:
            _, r, _, info = env.step(int(action_fn(env)))
            acc += np.asarray(r, dtype=float)
        gp = info["game_points"]
        for s in range(4):
            target = (gp[s % 2] - gp[1 - s % 2]) / 16.0
            d = abs(acc[s] - target)
            if d > worst:
                worst, worst_hand = d, g
    np.random.set_state(st)
    ok = worst < TOL
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<22s} worst |sum r - target| = "
          f"{worst:.3e}" + ("" if ok else f"   (hand {worst_hand})"))
    return ok


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"\nREWARD IDENTITY -- {n} hands per policy, all four seats\n")
    ok = True
    ok &= check(_heuristic_action, "greedy heuristic", n, 880_000)
    ok &= check(_random_action, "uniform random", n, 910_000)

    # A mixed table is the case that actually matters: the trainer stores episodes
    # from games where only some seats are learning, and the true-up has to hold
    # for every seat regardless of who drove it.
    def mixed(env):
        return (_heuristic_action(env) if env.current_player % 2 == 0
                else _random_action(env))

    ok &= check(mixed, "mixed table", n, 940_000)

    print("\n" + ("ALL PASS -- the identity holds exactly" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
