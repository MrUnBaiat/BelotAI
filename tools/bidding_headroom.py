"""
S12 -- WHAT IS BIDDING WORTH?  The headroom nobody has ever priced.

WHY NOW. Bidding is the one component of the player that has never been attacked. The
policy gradient is deadest exactly there (25th-percentile pi(a) = 0.9989), no search has
ever touched it, and P9-3 corrected the model's edge over the greedy heuristic DOWN from
+0.610 +- 0.401 to +0.272 +- 0.180 -- so the component is both smaller than the project
believed and, with card-play search now measured as saturated on four axes, a larger share
of whatever is left.

PRICE THE CEILING BEFORE BUILDING ANYTHING. The question is not "is a solver-based bidder
better" but "how much is ANY better bidder worth". So the top arm is a CHEAT bidder that
cannot be beaten by any real bidder given the same card play.

THE CHEAT BIDDER. At each of its own bidding decisions, for every legal bid: fork the env,
apply the bid, and roll the hand forward to completion with the normal policies in every
seat (model card play, model bidding for the other seats). Take the actual final
game-point difference. Choose the bid with the best outcome. This uses the true hidden
cards, so it is an upper bound on bidding skill given this card-play policy -- no bidder
that cannot see the deal can do better, and the only noise is zero because every policy in
the simulation is deterministic.

THREE ARMS, identical card play (the model plays every card in every arm), so the only
thing that varies is who bids for the even seats:

  MODEL    the current bidder                                      (baseline)
  HEUR     `_heuristic_action`, which is what `pimc.py` bids with   (the P9-3 reference)
  CHEAT    the perfect-information one-ply bidder above             (the ceiling)

READING RULE, FIXED BEFORE THE RUN.
  CHEAT - MODEL >= +1.0 significant -> bidding is a large untapped component and a
        solver-based bidder is worth building; it becomes the top route.
  +0.25 to +1.0 significant -> real but modest; rank it against the other live routes on
        size rather than on novelty.
  within +-0.25 -> the model's bidding is already near the achievable ceiling given this
        card play. Bidding is CLOSED and should be recorded as such.
  CHEAT below MODEL -> the lookahead is mis-specified (it optimises the wrong objective or
        the LSTM fork is wrong); treat as a bug, not a result.

Also re-measures MODEL - HEUR, which P9-3 put at +0.272 +- 0.180.

CAVEAT, STATED IN ADVANCE. Card play here is the model alone, not the model+search
composite. Bidding value could interact with card-play strength -- a better bidder may be
worth more when the cards are played better. This measures the isolated component.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import time

import numpy as np
import torch


import belot.evaluation.swap_eval as SW                              # noqa: E402
from belot.evaluation.swap_eval import ci
from belot.search.composite import load_model             # noqa: E402
from belot.heuristic import _heuristic_action                  # noqa: E402
from belot.observation import build_observation           # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 512


def _zero():
    return (torch.zeros(1, 1, HIDDEN, device=DEV),
            torch.zeros(1, 1, HIDDEN, device=DEV))


def make_player(net, bidder="model"):
    """Model card play always. `bidder` selects who chooses the bidding actions."""
    hc = {}

    def reset():
        for s in range(4):
            hc[s] = _zero()

    @torch.no_grad()
    def net_action(env, state):
        s = env.current_player
        local, glob, mask = build_observation(env, s, [0, 0])
        dist, _, state[s] = net(
            torch.from_numpy(local).unsqueeze(0).to(DEV),
            torch.from_numpy(glob).unsqueeze(0).to(DEV), state[s],
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(DEV),
            is_sequence=False)
        return int(dist.probs.argmax(-1).item())

    def rollout_to_end(env, state, first):
        """Apply `first`, then play the hand out with the model everywhere."""
        e = copy.deepcopy(env)
        st = {k: (v[0].clone(), v[1].clone()) for k, v in state.items()}
        info = {}
        a = first
        while True:
            _, _, done, info = e.step(int(a))
            if done or e.done:
                break
            a = net_action(e, st)
        gp = info.get("game_points")
        return gp

    @torch.no_grad()
    def fn(env):
        own = net_action(env, hc)          # advances the LSTM on EVERY decision
        if env.phase != "BIDDING":
            return own
        if bidder == "model":
            return own
        if bidder == "heur":
            return int(_heuristic_action(env))
        legal = np.flatnonzero(env.get_legal_actions())
        if len(legal) == 1:
            return int(legal[0])
        team = env.current_player % 2
        best_a, best_v = None, -1e18
        for a in legal:
            gp = rollout_to_end(env, hc, int(a))
            if gp is None:
                continue
            v = gp[team] - gp[1 - team]
            if v > best_v:
                best_v, best_a = v, int(a)
        return own if best_a is None else best_a

    reset()
    return fn, reset


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    st = np.random.get_state()
    net = load_model()
    print(f"S12 BIDDING HEADROOM: model card play throughout, only the bidder varies")
    print(f"    n={n} swap-paired deals, opponent = the model (model bidding)")
    print("")

    Y = make_player(net, "model")
    ctl = SW.swap_edges(make_player(net, "cheat"), make_player(net, "cheat"), 30)
    ok = np.abs(ctl).max() == 0.0
    print(f"  CONTROL {ctl.mean():+.3f}  max|edge| {np.abs(ctl).max():.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  CONTROL FAILED -- nothing below counts.")
        np.random.set_state(st)
        return

    out = {}
    for lab in ("cheat", "heur"):
        t0 = time.time()
        e = SW.swap_edges(make_player(net, lab), Y, n)
        out[lab] = e
        sig = "SIGNIFICANT" if abs(e.mean()) > ci(e) else "ns"
        print(f"  {lab.upper():<6s} bidder vs MODEL bidder   {e.mean():+.3f} +- "
              f"{ci(e):.3f} [{sig}]   changed {int((e != 0).sum())}/{n}   "
              f"{(time.time()-t0)/n*1000:.0f} ms/deal", flush=True)

    c = out["cheat"]
    lo, hi = c.mean() - ci(c), c.mean() + ci(c)
    sig_ = abs(c.mean()) > ci(c)
    print("")
    print(f"  THE CEILING ON BIDDING (cheat - model): {c.mean():+.3f} +- {ci(c):.3f}"
          f"   interval [{lo:+.3f}, {hi:+.3f}]")
    # branch checked against the docstring rule before printing (standing note)
    if c.mean() >= 1.0 and sig_:
        print("    -> LARGE untapped component; a solver-based bidder becomes the top route.")
    elif c.mean() >= 0.25 and sig_:
        print("    -> REAL BUT MODEST; rank on size against the other live routes.")
    elif hi <= 0.25 and lo >= -0.25:
        print("    -> CLOSED: the model's bidding is already near the achievable ceiling "
              "given this card play.")
    elif c.mean() < 0:
        print("    -> CHEAT BELOW MODEL: treat as a bug in the lookahead, not a result.")
    else:
        print(f"    -> UNDERPOWERED; interval spans the bar. n for +-0.25: "
              f"{int(len(c) * (ci(c) / 0.25) ** 2)}")
    h = out["heur"]
    print(f"\n  MODEL - HEUR bidder: {-h.mean():+.3f} +- {ci(h):.3f}"
          f"   (P9-3 measured +0.272 +- 0.180)")
    print(f"  the model captures "
          f"{100 * (-h.mean()) / max(c.mean() - h.mean(), 1e-9):.0f}% of the "
          f"heuristic-to-ceiling range")
    np.random.set_state(st)


if __name__ == "__main__":
    main()
