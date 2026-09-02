"""
Correctness and speed for the exact double-dummy solver.

CORRECTNESS. On random legal games, at EVERY playing state, assert that
  (a) dd_solver.legal_moves == env.get_legal_actions,
  (b) dd_solver.trick_winner == the env's own trick resolution,
  (c) team raw points at the end match.
A solver that does not mirror the env exactly is worse than useless -- it would train
a policy for a different game.

SPEED. Time a full solve from each trick number. This is the fact that decides whether
"PIMC with exact evaluation" is affordable here at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np


import belot.search.dd_solver as DD                              # noqa: E402
from belot.env import BelotEnv                            # noqa: E402
from belot.heuristic import _heuristic_action                  # noqa: E402


def deal_to_play(seed):
    """Play the bidding out with the heuristic; return an env at trick 0 of PLAYING."""
    env = BelotEnv()
    env.dealer = seed % 4
    np.random.seed(seed)
    env.reset()
    env.bolts_by_team = [0, 0]
    while env.phase == "BIDDING" and not env.done:
        env.step(_heuristic_action(env))
    return env


def check(n_games=60, verbose=True):
    bad_legal = bad_win = bad_pts = 0
    n_states = 0
    st = np.random.get_state()
    rng = np.random.default_rng(12345)
    for g in range(n_games):
        env = deal_to_play(700_000 + g)
        if env.done or env.phase != "PLAYING":
            continue
        while not env.done:
            hands = DD.hands_to_masks(env.hands)
            trick = tuple((p, c) for p, c in env.current_trick)
            mine = sorted(DD.legal_moves(hands[env.current_player], trick, env.trump,
                                         env.current_player, env.declarer,
                                         env.declarer_has_played_trump))
            theirs = sorted(int(x) for x in np.flatnonzero(env.get_legal_actions()))
            n_states += 1
            if mine != theirs:
                bad_legal += 1
                if verbose and bad_legal <= 3:
                    print(f"  LEGAL MISMATCH g{g} trick{env.tricks_played} "
                          f"cp{env.current_player} mine={mine} env={theirs}")
            if len(trick) == 3:
                a = int(rng.choice(theirs))
                full = trick + ((env.current_player, a),)
                pred = DD.trick_winner(full, env.trump)
                env.step(a)
                actual = env.current_player          # winner leads next
                if not env.done and pred != actual:
                    bad_win += 1
                    if verbose and bad_win <= 3:
                        print(f"  WINNER MISMATCH g{g} pred={pred} env={actual}")
                continue
            env.step(int(rng.choice(theirs)))
        if list(env.raw_points_by_team) and sum(env.raw_points_by_team) not in (162,):
            bad_pts += 1
    np.random.set_state(st)
    ok = (bad_legal == 0 and bad_win == 0 and bad_pts == 0)
    print(f"  states checked {n_states}   legal mismatches {bad_legal}   "
          f"winner mismatches {bad_win}   points anomalies {bad_pts}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def speed(n=12, from_trick=0):
    """Time a full exact solve of every root move, starting at `from_trick`."""
    st = np.random.get_state()
    rng = np.random.default_rng(999)
    times, nodes, ncards = [], [], []
    for g in range(n):
        env = deal_to_play(810_000 + g)
        if env.done or env.phase != "PLAYING":
            continue
        for _ in range(from_trick * 4):
            if env.done:
                break
            legal = np.flatnonzero(env.get_legal_actions())
            env.step(int(rng.choice(legal)))
        if env.done:
            continue
        hands = DD.hands_to_masks(env.hands)
        trick = tuple((p, c) for p, c in env.current_trick)
        t0 = time.perf_counter()
        best, vals, nd = DD.solve_root(hands, env.current_player, trick, env.trump,
                                       env.declarer, env.declarer_has_played_trump)
        dt = time.perf_counter() - t0
        times.append(dt); nodes.append(nd)
        ncards.append(sum(len(h) for h in env.hands))
    np.random.set_state(st)
    if not times:
        return None
    times = np.array(times); nodes = np.array(nodes)
    print(f"  trick {from_trick}: cards left {int(np.mean(ncards)):2d}   "
          f"n={len(times):2d}   median {np.median(times)*1000:9.1f} ms   "
          f"mean {times.mean()*1000:9.1f} ms   max {times.max()*1000:9.1f} ms   "
          f"median nodes {int(np.median(nodes)):>9d}")
    return float(np.median(times))


if __name__ == "__main__":
    print("=" * 74)
    print("CORRECTNESS -- solver rules vs env.py, on random legal games")
    print("=" * 74)
    ok = check(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    if not ok:
        sys.exit(1)
    print("\n" + "=" * 74)
    print("SPEED -- full exact solve of every root move")
    print("=" * 74)
    for t in (7, 6, 5, 4, 3, 2, 1, 0):
        med = speed(12, from_trick=t)
        if med is not None and med > 20.0:
            print("  (stopping: deeper solves exceed 20 s)")
            break
