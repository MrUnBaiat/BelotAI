"""
E4 -- HOW MUCH ROOM IS THERE ABOVE THE SEARCH FAMILY?

Nothing in this project has ever measured the CEILING of the PIMC family. Phase 7 built
a player at +0.974 +- 0.175 over the agent and recommended more of the same; the obvious
question -- how much is left up there, and where does it live -- has never been asked.

THREE PLAYERS, identical except for WHICH HANDS THE SEARCH IS ALLOWED TO SEE. All three
use the same model base (it bids and plays tricks 0-2), the same exact solver, the same
deals, the same opponent, swap-paired, control exactly zero.

  SAMPLED   D=8 determinizations sampled the normal way          -- the Phase 7 player
  PARTNER   D=8 determinizations with the PARTNER'S TRUE HAND fixed, opponents sampled
  CHEAT     the TRUE deal, solved exactly, D=1                   -- double-dummy optimum

READING, FIXED BEFORE THE RUN.

  CHEAT - SAMPLED  = the total value of the information the search does not have. It is
                     an UPPER BOUND on every route that improves inference or reasons
                     about information sets -- IS-MCTS, Deep CFR, belief networks, the
                     lot. If it is small, the search family is near its ceiling and the
                     only remaining lever is compute. If it is large, there is a real
                     prize above PIMC and it is worth an expensive method to reach it.

  PARTNER - SAMPLED = an UPPER BOUND on partnership signalling. A signalling convention
                     can at very best communicate the partner's whole hand, and this
                     measures having it for free. Phase 7 recorded signalling as an
                     unmeasured blind spot bounding the DD family; this is the bound.

  The two together SPLIT the headroom into "know your partner" and "know the opponents".

WHY THE PARTNER CONSTRUCTION IS SOUND. It re-uses `pimc.sample_determinization`
unmodified, by temporarily marking the partner's true cards in `env.known_cards` -- the
same channel the env already uses for forced inferences. The sampler then forces exactly
those cards into the partner's hand and samples only the two opponents. `env` is
restored immediately afterwards.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import torch


import belot.search.dd_solver as DD                              # noqa: E402
import belot.evaluation.swap_eval as SW                              # noqa: E402
from belot.search.composite import model_backed                # noqa: E402
from belot.evaluation.swap_eval import ci
from belot.search.composite import load_model             # noqa: E402
from belot.search.pimc import sample_determinization             # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _solve_pick(env, hands, me):
    masks = DD.hands_to_masks(hands)
    trick = tuple((p, c) for p, c in env.current_trick)
    _, vals, _ = DD.solve_root(masks, me, trick, env.trump, env.declarer,
                               env.declarer_has_played_trump)
    return vals


def make_solver(mode, D=8, seed=0):
    """mode in {'sampled', 'partner', 'cheat'} -- identical except for the hands seen."""
    state = {"rng": np.random.default_rng(seed)}

    def act(env):
        legal = np.flatnonzero(env.get_legal_actions())
        if len(legal) == 1:
            return int(legal[0])
        me = env.current_player
        team = me % 2
        tot = np.zeros(len(legal))
        n = 0
        reps = 1 if mode == "cheat" else D
        for _ in range(reps):
            if mode == "cheat":
                hands = [list(h) for h in env.hands]
            elif mode == "partner":
                pa = (me + 2) % 4
                saved = env.known_cards[pa].copy()
                for c in env.hands[pa]:
                    env.known_cards[pa, c] = True
                hands = sample_determinization(env, me, state["rng"])
                env.known_cards[pa] = saved
            else:
                hands = sample_determinization(env, me, state["rng"])
            if hands is None:
                continue
            vals = _solve_pick(env, hands, me)
            for i, a in enumerate(legal):
                v = vals[int(a)]
                tot[i] += v if team == 0 else -v
            n += 1
        if n == 0:
            return -1
        return int(legal[int(np.argmax(tot))])

    def reseed(s):
        state["rng"] = np.random.default_rng(s)

    return act, reseed


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mt = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    st = np.random.get_state()
    net = load_model()
    Y = SW.net_policy(net, DEV)
    print(f"E4 HEADROOM: model base, exact solver from trick {mt}, D={D}")
    print(f"   n={n} swap-paired deals, opponent = the model\n")

    cores = {m: make_solver(m, D, 0) for m in ("sampled", "partner", "cheat")}

    c0, r0 = cores["sampled"]
    print("--- CONTROL: SAMPLED against itself (must be exactly zero) ---")
    ctl = SW.swap_edges(model_backed(net, c0, mt), model_backed(net, c0, mt), 30,
                        pimc_seed_fn=lambda d: r0(88_000 + d))
    ok = np.abs(ctl).max() == 0.0
    print(f"  {ctl.mean():+.3f}   max|edge| {np.abs(ctl).max():.2e}   "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  CONTROL FAILED -- nothing below counts.")
        np.random.set_state(st)
        return

    out = {}
    for m in ("sampled", "partner", "cheat"):
        core, re = cores[m]
        t0 = time.time()
        e = SW.swap_edges(model_backed(net, core, mt), Y, n,
                          pimc_seed_fn=lambda d, r=re: r(89_000 + d))
        out[m] = e
        sig = "SIGNIFICANT" if abs(e.mean()) > ci(e) else "ns"
        print(f"  {m:<10s} vs model  {e.mean():+.3f} +- {ci(e):.3f} [{sig}]"
              f"   changed {int((e != 0).sum())}/{n}   "
              f"{(time.time()-t0)/n*1000:.0f} ms/deal", flush=True)

    print("\n--- THE HEADROOM DECOMPOSITION (paired, same deals) ---")
    dp = out["partner"] - out["sampled"]
    dc = out["cheat"] - out["sampled"]
    dr = out["cheat"] - out["partner"]
    for lab, d in (("partner's hand (= signalling BOUND)", dp),
                   ("full perfect information (= TOTAL)", dc),
                   ("the two opponents' hands", dr)):
        sig = "SIGNIFICANT" if abs(d.mean()) > ci(d) else "ns"
        print(f"  {lab:<38s} {d.mean():+.3f} +- {ci(d):.3f} [{sig}]")
    if abs(dc.mean()) > 1e-9:
        print(f"\n  signalling is at most {100*dp.mean()/dc.mean():.0f}% of the total "
              f"information headroom")
    print(f"  for scale: 660 epochs of PPO = +0.089 +- 0.286;  "
          f"selective search over the agent = +0.974 +- 0.175")
    np.random.set_state(st)


if __name__ == "__main__":
    main()
