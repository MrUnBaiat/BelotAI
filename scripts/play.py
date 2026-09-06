"""
Play one hand with the composite player and print it, card by card.

    python scripts/play.py --ckpt checkpoints/v8_exp/expd_latest.pt

A demo and a debugging aid: it shows the auction, every trick, who won it and for
how many points, and -- for each of the composite's own decisions -- whether the
network or the search chose the card. Seats 0 and 2 are the composite; seats 1 and
3 are the greedy heuristic, so the two sides are visibly different players.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from belot.env import BelotEnv
from belot.heuristic import _heuristic_action
from belot.search.composite import DEFAULT_D, build_player

DEFAULT_CKPT = os.path.join("checkpoints", "v8_exp", "expd_latest.pt")
SUITS = "shdc"          # spades hearts diamonds clubs -- ASCII, because a
                        # Windows console in cp1252 cannot encode the glyphs
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
SEATS = ["S0", "S1", "S2", "S3"]


def card(c):
    return f"{RANKS[c % 8]}{SUITS[c // 8]}"


def hand(cs):
    return " ".join(card(c) for c in sorted(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--deal", type=int, default=0, help="deal seed")
    ap.add_argument("--D", type=int, default=DEFAULT_D)
    ap.add_argument("--min-trick", type=int, default=3)
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        sys.exit(f"checkpoint not found: {a.ckpt}\n"
                 f"Weights are not distributed with the repo -- see the README.")

    state = np.random.get_state()
    act, reset, reseed = build_player(a.ckpt, D=a.D, min_trick=a.min_trick)
    reseed(a.deal)
    reset()

    env = BelotEnv()
    env.dealer = a.deal % 4
    np.random.seed(770_000 + a.deal)
    env.reset()
    env.bolts_by_team = [0, 0]

    print(f"\ndeal {a.deal}   dealer {SEATS[env.dealer]}   "
          f"face-up {card(env.face_up_card)}")
    print(f"composite = seats 0 and 2   |   greedy heuristic = seats 1 and 3\n")
    for s in range(4):
        who = "composite" if s % 2 == 0 else "heuristic"
        print(f"  {SEATS[s]} ({who:<9s}) {hand(env.hands[s])}")

    print("\n--- auction ---")
    info = {}
    while env.phase == "BIDDING" and not env.done:
        s = env.current_player
        mine = s % 2 == 0
        chosen = act(env) if mine else _heuristic_action(env)
        name = {32: "pass", 33: "accept"}.get(chosen,
                                              f"trump {SUITS[chosen - 34]}"
                                              if chosen >= 34 else str(chosen))
        print(f"  {SEATS[s]}  {name}")
        env.step(int(chosen))
    if env.trump is not None:
        print(f"  -> trump {SUITS[env.trump]}, declarer {SEATS[env.declarer]}")
    print("\n  after the talon:")
    for s in range(4):
        print(f"  {SEATS[s]} {hand(env.hands[s])}")

    print("\n--- play ---")
    trick_no = -1
    while not env.done:
        s = env.current_player
        if env.tricks_played != trick_no:
            trick_no = env.tricks_played
            src = "search" if trick_no >= a.min_trick else "network"
            print(f"\n  trick {trick_no}   (composite plays by {src})")
        n_legal = int(env.get_legal_actions().sum())
        mine = s % 2 == 0
        chosen = act(env) if mine else _heuristic_action(env)
        if mine:
            tag = ("forced" if n_legal == 1 else
                   "search" if env.tricks_played >= a.min_trick else "network")
        else:
            tag = "heuristic"
        print(f"    {SEATS[s]} plays {card(chosen):>4s}   [{tag}]")
        _, _, _, info = env.step(int(chosen))
        if env.tricks_played != trick_no and not env.done:
            print(f"    -> raw points now  team0 {env.raw_points_by_team[0]:>3d}"
                  f"   team1 {env.raw_points_by_team[1]:>3d}")

    gp = info["game_points"]
    print(f"\n--- result ---")
    print(f"  raw points   team0 {env.raw_points_by_team[0]:>3d}"
          f"   team1 {env.raw_points_by_team[1]:>3d}   (162 total)")
    print(f"  game points  team0 {gp[0]:>3d}   team1 {gp[1]:>3d}")
    print(f"  composite (team 0) {'wins' if gp[0] > gp[1] else 'loses' if gp[0] < gp[1] else 'ties'}"
          f" this hand by {gp[0] - gp[1]:+d}\n")
    np.random.set_state(state)


if __name__ == "__main__":
    main()
