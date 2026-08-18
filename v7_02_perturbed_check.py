"""
AUDIT v7 / EXP-C precondition -- is the perturbed heuristic actually different?

A perturbation too small to change behaviour would be the exact greedy heuristic
wearing a hat, and would re-create the yardstick contamination it exists to avoid
(pimc.py delegates bidding to _heuristic_action, so exploitation of heuristic
bidding transfers into the PIMC yardstick).

Conversely a perturbation that makes it far WEAKER is a bad sparring partner and
would mostly teach the model to punish bad bidding.

MEASURED HERE
  T1 legality        every action legal.
  T2 divergence      fraction of decisions where it differs from the exact
                     heuristic, split by phase. Bidding should differ materially;
                     card play should be identical by construction (0.0%).
  T3 strength        paired vs the exact heuristic on identical deals. Wanted:
                     clearly non-zero (so it is a different policy) but not a
                     collapse (so it is still a useful opponent).
  T4 declare rate    the perturbation is a looser bid threshold, so it must
                     declare noticeably more often -- a direct check that the
                     intended behavioural change actually happened.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
from env import BelotEnv
from eval import _heuristic_action
from perturbed_heuristic import perturbed_heuristic_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
BASE = 770_000


def duel(a0, a1, n):
    """a0 plays seats 0&2, a1 plays 1&3. Returns per-deal gp diff + declare rate."""
    out = np.empty(n)
    dec = 0
    for d in range(n):
        env = BelotEnv(); env.dealer = d % 4
        np.random.seed(BASE + d); env.reset(); env.bolts_by_team = [0, 0]
        info = {}
        while not env.done:
            a = (a0 if env.current_player % 2 == 0 else a1)(env)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        out[d] = gp[0] - gp[1]
        dec += int(env.declaring_team == 0)
    return out, dec / n


def main():
    bad_legal = 0
    diff_bid = tot_bid = diff_play = tot_play = 0

    for d in range(400):
        env = BelotEnv(); env.dealer = d % 4
        np.random.seed(BASE + d); env.reset(); env.bolts_by_team = [0, 0]
        while not env.done:
            a_ref = _heuristic_action(env)
            a_new = perturbed_heuristic_action(env)
            if not env.get_legal_actions()[a_new]:
                bad_legal += 1
            if env.phase == "BIDDING":
                tot_bid += 1; diff_bid += int(a_ref != a_new)
            else:
                tot_play += 1; diff_play += int(a_ref != a_new)
            env.step(a_new)

    print(f"T1 legality      : {bad_legal} illegal  -> "
          f"{'PASS' if bad_legal == 0 else 'FAIL'}")
    print(f"T2 divergence    : bidding {diff_bid}/{tot_bid} = "
          f"{diff_bid / max(tot_bid,1):.1%}   "
          f"card play {diff_play}/{tot_play} = {diff_play / max(tot_play,1):.1%}")
    print(f"   card play must be 0.0% by construction -> "
          f"{'PASS' if diff_play == 0 else 'FAIL'}")
    print(f"   bidding must be materially non-zero    -> "
          f"{'PASS' if diff_bid / max(tot_bid,1) > 0.05 else 'FAIL (too similar)'}")

    ref, ref_dec = duel(_heuristic_action, _heuristic_action, N)
    per, per_dec = duel(perturbed_heuristic_action, _heuristic_action, N)
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    d = per - ref
    print(f"\nT3 strength ({N} paired deals, as team 0 vs the EXACT heuristic)")
    print(f"   exact  vs exact  : {ref.mean():+.3f} +- {ci(ref):.3f}  (floor, expect ~0)")
    print(f"   pertd  vs exact  : {per.mean():+.3f} +- {ci(per):.3f}")
    print(f"   paired difference: {d.mean():+.3f} +- {ci(d):.3f}  "
          f"[{'SIGNIFICANT' if abs(d.mean()) > ci(d) else 'not significant'}]")
    print(f"\nT4 declare rate  : exact {ref_dec:.1%}  perturbed {per_dec:.1%}  "
          f"(looser threshold must declare MORE) -> "
          f"{'PASS' if per_dec > ref_dec + 0.02 else 'FAIL'}")

    print("\n" + "=" * 74)
    ok_diff = diff_bid / max(tot_bid, 1) > 0.05 and diff_play == 0
    ok_str = abs(d.mean()) < 3.0
    if ok_diff and ok_str and bad_legal == 0:
        print("USABLE as a training opponent: behaviourally distinct in bidding,")
        print("identical in card play, and not collapsed in strength.")
    else:
        print("NOT usable as specified -- adjust the perturbation before EXP-C.")


if __name__ == "__main__":
    main()
