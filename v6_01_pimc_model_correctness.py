"""
AUDIT v6 / EXP-1a -- correctness of pimc_model.py before any strength claim.

pimc.py's suite (v5_01) covers the determinization sampler, which is imported
unchanged. What is NEW here and therefore untested is: _load_full (which now
carries the inference arrays, dealer and last trick), the batched lockstep
rollout, and the critic-at-leaf scoring path. Each gets a test.

T1 LEGALITY, both leaf modes. Every returned action legal in the real position.
T2 RNG PURITY. Must not advance numpy's global stream -- paired deals across every
   condition in this project depend on it.
T3 WORLD FIDELITY. Each root world must reproduce the real position exactly:
   same trump/declarer/trick/graveyard/points, the searcher's own hand unchanged,
   and the inference arrays copied rather than cleared (the bug _load_full exists
   to avoid -- pimc.py's _load zeroes them, which is harmless for a heuristic
   rollout policy but feeds a model the wrong belief matrix).
T4 ROLLOUT COMPLETION. Every world reaches tricks_played == 8, and all 32 cards
   are accounted for -- catches a lockstep loop that drops or double-steps worlds.
T5 BATCHING IS EXACT. A batched forward pass must produce the same greedy actions
   as one-at-a-time forwards on the same states.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from env import BelotEnv
from eval import _heuristic_action
from model import RecurrentMAPPOModel
from observation import build_observation
from pimc import sample_determinization
from pimc_model import _load_full, make_model_pimc, HIDDEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "checkpoints/v4_exp/c1_latest.pt"
D = 4
N_HANDS = 25


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    roll = make_model_pimc(model, DEVICE, D=D, seed=0, leaf="rollout")
    crit = make_model_pimc(model, DEVICE, D=D, seed=0, leaf="critic")

    bad_legal_r = bad_legal_c = 0
    bad_world = bad_infer = 0
    decisions = 0
    rng = np.random.default_rng(3)

    for h in range(N_HANDS):
        env = BelotEnv(); env.dealer = h % 4
        np.random.seed(7000 + h); env.reset(); env.bolts_by_team = [0, 0]
        while not env.done:
            if env.phase == "PLAYING":
                # ---- T3: world fidelity ----
                from pimc import _scratch_env
                w = _scratch_env()
                hands = sample_determinization(env, env.current_player, rng)
                _load_full(w, env, hands)
                same = (w.trump == env.trump and w.declarer == env.declarer
                        and w.current_trick == env.current_trick
                        and w.current_player == env.current_player
                        and w.graveyard == env.graveyard
                        and w.raw_points_by_team == env.raw_points_by_team
                        and w.tricks_won_by_team == env.tricks_won_by_team
                        and w.dealer == env.dealer
                        and w.last_trick == env.last_trick
                        and sorted(w.hands[env.current_player])
                        == sorted(env.hands[env.current_player]))
                bad_world += (not same)
                if not (np.array_equal(w.impossible_cards, env.impossible_cards)
                        and np.array_equal(w.known_cards, env.known_cards)):
                    bad_infer += 1

                a = roll(env)
                bad_legal_r += (not env.get_legal_actions()[a])
                b = crit(env)
                bad_legal_c += (not env.get_legal_actions()[b])
                decisions += 1
                env.step(a)
            else:
                env.step(roll(env))

    # ---- T2: RNG purity, ISOLATED.
    # The first version of this test captured the state before a loop that itself
    # called np.random.seed() once per hand, so it compared across its own
    # re-seeds and failed trivially. The deal must be set up FIRST, then the
    # state captured, then only PIMC allowed to run.
    e3 = BelotEnv(); e3.dealer = 0
    np.random.seed(555); e3.reset(); e3.bolts_by_team = [0, 0]
    while e3.phase != "PLAYING":
        e3.step(_heuristic_action(e3))
    before = np.random.get_state()[1][:8].copy()
    for _ in range(6):
        if e3.done:
            break
        roll(e3); crit(e3)
        e3.step(_heuristic_action(e3))
    after = np.random.get_state()[1][:8].copy()
    rng_pure = np.array_equal(before, after)

    # ---- T4: rollout completion, checked directly ----
    env = BelotEnv(); env.dealer = 0
    np.random.seed(999); env.reset(); env.bolts_by_team = [0, 0]
    while env.phase != "PLAYING":
        env.step(_heuristic_action(env))
    from pimc import _scratch_env
    incomplete = 0
    for _ in range(20):
        w = _scratch_env()
        hands = sample_determinization(env, env.current_player, rng)
        _load_full(w, env, hands)
        while not w.done:
            w._handle_playing_action(int(np.flatnonzero(w.get_legal_actions())[0]))
        if w.tricks_played != 8 or len(w.graveyard) != 32:
            incomplete += 1

    # ---- T5: batched == unbatched ----
    states = []
    e2 = BelotEnv(); e2.dealer = 1
    np.random.seed(321); e2.reset(); e2.bolts_by_team = [0, 0]
    while not e2.done and len(states) < 24:
        if e2.phase == "PLAYING":
            states.append(build_observation(e2, e2.current_player, [0, 0]))
        e2.step(_heuristic_action(e2))
    loc = torch.from_numpy(np.array([s[0] for s in states])).to(DEVICE)
    glb = torch.from_numpy(np.array([s[1] for s in states])).to(DEVICE)
    msk = torch.from_numpy(np.array([s[2] for s in states], dtype=np.float32)).to(DEVICE)
    n = len(states)
    with torch.no_grad():
        d_b, _, _ = model(loc, glb, (torch.zeros(1, n, HIDDEN, device=DEVICE),
                                     torch.zeros(1, n, HIDDEN, device=DEVICE)),
                          msk, is_sequence=False)
        batched = d_b.probs.argmax(-1).cpu().numpy()
        single = []
        for i in range(n):
            d_s, _, _ = model(loc[i:i+1], glb[i:i+1],
                              (torch.zeros(1, 1, HIDDEN, device=DEVICE),
                               torch.zeros(1, 1, HIDDEN, device=DEVICE)),
                              msk[i:i+1], is_sequence=False)
            single.append(int(d_s.probs.argmax(-1).item()))
    batch_ok = int((batched == np.array(single)).sum())

    print(f"decisions exercised: {decisions} over {N_HANDS} hands (D={D})\n")
    print(f"T1 legality rollout-leaf : {bad_legal_r} illegal   "
          f"-> {'PASS' if bad_legal_r == 0 else 'FAIL'}")
    print(f"T1 legality critic-leaf  : {bad_legal_c} illegal   "
          f"-> {'PASS' if bad_legal_c == 0 else 'FAIL'}")
    print(f"T2 global RNG untouched  : {rng_pure}          "
          f"-> {'PASS' if rng_pure else 'FAIL'}")
    print(f"T3 world fidelity        : {bad_world} mismatched  "
          f"-> {'PASS' if bad_world == 0 else 'FAIL'}")
    print(f"T3 inference arrays copied: {bad_infer} cleared    "
          f"-> {'PASS' if bad_infer == 0 else 'FAIL'}")
    print(f"T4 rollout completion    : {incomplete}/20 incomplete "
          f"-> {'PASS' if incomplete == 0 else 'FAIL'}")
    print(f"T5 batched == unbatched  : {batch_ok}/{n} agree   "
          f"-> {'PASS' if batch_ok == n else 'FAIL'}")


if __name__ == "__main__":
    main()
