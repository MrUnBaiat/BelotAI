"""
AUDIT v5 / EXP-1 -- correctness tests for pimc.py, BEFORE measuring its strength.

A search player that quietly plays illegal moves, samples inconsistent worlds, or
perturbs the global RNG would produce a strength number that looks fine and is
worthless. Four properties are checked by execution:

T1 LEGALITY. Every action PIMC returns is legal in the real position.

T2 DETERMINIZATION CONSISTENCY. Every sampled world must (a) give each opponent
   exactly their true hand size, (b) use exactly the unseen cards with no
   duplicates and none of the searcher's own or already-played cards, (c) place
   every `known_cards` card with its known holder. Constraint (d), respecting
   `impossible_cards`, is reported as a RATE rather than asserted, because the
   sampler has a documented relaxation fallback for jointly-unsatisfiable states.

T3 GLOBAL RNG PURITY. Running PIMC must not advance numpy's global RNG, otherwise
   re-seeding no longer reproduces identical deals and every paired experiment in
   this project silently breaks.

T4 GROUND-TRUTH REACHABILITY. The true opponent hands must themselves satisfy the
   constraints the sampler enforces -- a sanity check that the inference arrays in
   env.py are sound (0 contradictions, matching audit 4.4).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from belot.env import BelotEnv
from belot.heuristic import _heuristic_action
from belot.search.pimc import make_pimc, sample_determinization

D = 8
N_HANDS = 60


def main():
    pimc = make_pimc(D=D, seed=0)
    bad_legal = bad_size = bad_cards = bad_known = 0
    imp_ok = imp_tot = 0
    truth_viol = 0
    decisions = 0
    rng = np.random.default_rng(1)

    np.random.seed(4242)
    state_before = None

    for h in range(N_HANDS):
        env = BelotEnv()
        env.dealer = h % 4
        np.random.seed(5000 + h)
        env.reset()
        env.bolts_by_team = [0, 0]
        while not env.done:
            if env.phase == "PLAYING":
                me = env.current_player
                seen = set(env.hands[me]) | set(env.graveyard) | \
                    {c for _, c in env.current_trick}
                unseen = {c for c in range(32) if c not in seen}
                others = [(me + 1) % 4, (me + 2) % 4, (me + 3) % 4]

                # ---- T4: is the TRUE world consistent with the inferences? ----
                for q in others:
                    for c in env.hands[q]:
                        if env.impossible_cards[q, c]:
                            truth_viol += 1

                # ---- T2: sampled worlds ----
                for _ in range(4):
                    hands = sample_determinization(env, me, rng)
                    assert hands is not None
                    for q in others:
                        if len(hands[q]) != len(env.hands[q]):
                            bad_size += 1
                        for c in hands[q]:
                            imp_tot += 1
                            if not env.impossible_cards[q, c]:
                                imp_ok += 1
                        for c in np.flatnonzero(env.known_cards[q]):
                            if int(c) in unseen and int(c) not in hands[q]:
                                bad_known += 1
                    allc = [c for q in others for c in hands[q]]
                    if len(allc) != len(set(allc)) or set(allc) != unseen:
                        bad_cards += 1

                # ---- T1: legality ----
                a = pimc(env)
                if not env.get_legal_actions()[a]:
                    bad_legal += 1
                decisions += 1
            else:
                a = pimc(env)
                if not env.get_legal_actions()[a]:
                    bad_legal += 1
            env.step(a)

        if h == 0:
            # ---- T3: capture global RNG state, run PIMC, compare ----
            np.random.seed(999)
            state_before = np.random.get_state()[1][:8].copy()
            e2 = BelotEnv(); e2.dealer = 0
            np.random.seed(777); e2.reset()
            np.random.seed(999)
            for _ in range(30):
                if e2.done:
                    break
                e2.step(pimc(e2))
            state_after = np.random.get_state()[1][:8].copy()
            rng_pure = np.array_equal(state_before, state_after)

    print(f"decisions exercised: {decisions}  (D={D}, {N_HANDS} hands)")
    print(f"T1 legality          : {bad_legal} illegal actions            "
          f"-> {'PASS' if bad_legal == 0 else 'FAIL'}")
    print(f"T2 hand sizes        : {bad_size} wrong                       "
          f"-> {'PASS' if bad_size == 0 else 'FAIL'}")
    print(f"T2 card partition    : {bad_cards} malformed worlds           "
          f"-> {'PASS' if bad_cards == 0 else 'FAIL'}")
    print(f"T2 known_cards honoured: {bad_known} misplaced                "
          f"-> {'PASS' if bad_known == 0 else 'FAIL'}")
    print(f"T2 impossible_cards respected: {imp_ok}/{imp_tot} "
          f"({imp_ok / max(imp_tot,1):.4%}) -- relaxation fallback accounts for the rest")
    print(f"T3 global RNG untouched: {rng_pure}                           "
          f"-> {'PASS' if rng_pure else 'FAIL'}")
    print(f"T4 true world vs inferences: {truth_viol} contradictions      "
          f"-> {'PASS' if truth_viol == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
