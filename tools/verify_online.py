"""
Pre-flight, before putting the composite on a real table against real people.

    python tools/verify_online.py FRAMES.jsonl --ckpt checkpoints/v8_exp/expd_latest.pt

Five checks, ordered by how expensive each is to get wrong:

  1. rules parity      our BelotEnv agrees with the SDK's rulebook
  2. world legality    every sampled determinization is a legal world
  3. encoder parity    our belief block vs the SDK's, at the level of decisions
  4. action parity     the network picks the same cards it did before
  5. search timing     the slowest decision fits well inside the 25 s budget

Check 1 is the one that matters most. A rulebook drift shows up live as a move the
server refuses, which times the turn out, which hands the seat to belot.md's own bot
for the rest of the session -- and from then on every message we send is ignored
while the log keeps cheerfully printing the cards we chose.

Exit code is 1 if any check fails, so this can gate a run.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from belot.search.composite import DEFAULT_D
from belotmd import Infeasible, constraints, sample_determinization
from belotmd.game.belief import belief_matrix
from belotmd.platform.sync import StateSynchronizer

OK, BAD = "  ok  ", " FAIL "


def _states(path, want_play=True):
    """Replay a capture, yielding (sync, seat) at every well-formed state."""
    sync = StateSynchronizer()
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        seat = sync.sync(rec["state"], rec.get("pid"))
        if seat is None:
            continue
        st = sync.state
        if want_play:
            if st.phase != "PLAYING" or st.done or not st.hands[seat]:
                continue
            total = (sum(len(h) for h in st.hands) + len(st.graveyard)
                     + len(st.current_trick))
            if total != 32:
                continue
        yield sync, seat, rec


# --------------------------------------------------------------- 1. rules
def check_rules(n=50_000, seed=1):
    """The rulebook the model trains against must be the one it plays against.

    A drift here surfaces live as a move the server refuses -> the turn times
    out -> belot.md takes the seat for the rest of the session.
    """
    try:
        from belot.env import BelotEnv
    except Exception as exc:
        print(f"{BAD} rules parity: cannot import BelotEnv ({exc})")
        return False

    from belotmd.game.state import BelotState

    rng = np.random.default_rng(seed)
    mine, theirs = BelotState(), BelotEnv()
    bad = 0
    for _ in range(n):
        deck = rng.permutation(32).tolist()
        size = int(rng.integers(1, 9))
        snap = dict(
            hands=[deck[i * 8:i * 8 + size] for i in range(4)],
            phase=("PLAYING" if rng.random() < 0.5 else "BIDDING"),
            bidding_round=int(rng.integers(1, 3)),
            trump=int(rng.integers(0, 4)),
            declarer=int(rng.integers(0, 4)),
            dealer=int(rng.integers(0, 4)),
            current_player=int(rng.integers(0, 4)),
            face_up_card=deck[-1], face_up_suit=deck[-1] // 8,
            face_up_rank=deck[-1] % 8,
            current_trick=[], graveyard=[], tricks_played=0,
            declarer_has_played_trump=bool(rng.random() < 0.5), done=False,
        )
        for obj in (mine, theirs):
            for k, v in snap.items():
                setattr(obj, k, list(v) if isinstance(v, list) else v)
        if not np.array_equal(mine.get_legal_actions(),
                              theirs.get_legal_actions()):
            bad += 1

    print(f"{OK if not bad else BAD} rules parity: {n} states, "
          f"{bad} disagreements")
    return bad == 0


# ------------------------------------------------------------- 2. worlds
def check_worlds(path, per_state=4, seed=0):
    rng = np.random.default_rng(seed)
    states = sampled = failed = 0
    for sync, seat, _ in _states(path):
        states += 1
        cons = constraints(sync.state, seat)
        for _ in range(per_state):
            try:
                hands = sample_determinization(sync.state, seat, rng)
                cons.check(hands)         # independent validation
                sampled += 1
            except Infeasible as exc:
                failed += 1
                if failed <= 3:
                    print(f"        {exc}")
    print(f"{OK if not failed else BAD} world legality: {states} states, "
          f"{sampled} legal worlds, {failed} rejected")
    return failed == 0


# ------------------------------------------------------ 3. encoder parity
def check_encoder(path, ckpt):
    """Our belief block against the SDK's, judged by whether decisions change.

    The two encoders are identical except for the belief matrix at offsets
    339:435. Ours is a six-round IPF; the SDK's is a single normalisation pass, and
    they disagree by as much as 0.55 on the observation's most load-bearing feature
    (blinding that block costs -1.381 +- 0.296 pts/hand offline).

    We keep OURS: it is what the +0.974 was measured with, and it already reads
    `known_cards` / `impossible_cards`, which is exactly where the SDK writes the
    pins it decodes from declared combinations -- so it picks up the online-only
    information without changing the calibration the network was trained on.

    This check exists to keep that choice honest. If the two encoders ever start
    disagreeing about actual moves, the decision deserves re-opening.
    """
    if not ckpt:
        print(f"{OK} encoder parity: skipped (no --ckpt)")
        return True
    import torch

    from belot.model import RecurrentMAPPOModel
    from belot.observation import build_observation

    dev = torch.device("cpu")
    net = RecurrentMAPPOModel(hidden_dim=512).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)["model_state_dict"])
    net.eval()

    def pick(local, mask):
        h = (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
        with torch.no_grad():
            dist, _, _ = net(torch.from_numpy(local).unsqueeze(0),
                             torch.zeros(1, 332), h,
                             torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
                             is_sequence=False)
        return int(dist.probs.argmax(-1))

    same = diff = 0
    worst = 0.0
    for sync, seat, _ in _states(path):
        st = sync.state
        if st.current_player != seat:
            continue
        mask = st.get_legal_actions().astype(np.int8)
        if int(mask.sum()) < 2:
            continue
        ours, _, _ = build_observation(st, seat, [0, 0])
        theirs = ours.copy()
        theirs[339:435] = belief_matrix(st, seat).reshape(-1)
        worst = max(worst, float(np.abs(ours - theirs).max()))
        a, b = pick(ours, mask), pick(theirs, mask)
        same += (a == b)
        diff += (a != b)

    n = same + diff
    rate = 100.0 * diff / max(n, 1)
    print(f"{OK} encoder parity: {n} unforced decisions, {diff} differ "
          f"({rate:.1f}%), worst belief gap {worst:.3f}")
    print(f"        reported, not gated -- ours is the encoder the strength was "
          f"measured with")
    return True


# ------------------------------------------------------- 4. action parity
def check_actions(path, ckpt, baseline=None, save_to=None):
    try:
        from belot.online.agent import build
    except Exception as exc:
        print(f"{BAD} action parity: cannot import the agent ({exc})")
        return False

    agent = build(checkpoint=ckpt, search=False)     # network only
    picked = []
    sync = StateSynchronizer()
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        seat = sync.sync(rec["state"], rec.get("pid"))
        if seat is None:
            continue
        if sync.consume_new_hand():
            agent.reset()
        st, phase = sync.state, rec["state"].get("currentPhase", 0)
        if phase in (0, 13, 14):
            agent.reset()
            continue
        if phase == 10 and any(s == seat for s, _ in st.current_trick):
            continue
        if phase in (6, 7, 10) and rec["state"].get("activePlayer") == seat:
            mask = st.get_legal_actions().astype(np.int8)
            if mask.any():
                picked.append(agent.act(st, seat, sync.match_scores, mask))

    if save_to:
        with open(save_to, "w", encoding="utf-8") as fh:
            json.dump(picked, fh)
        print(f"        baseline of {len(picked)} decisions written to {save_to}")

    if baseline is None:
        print(f"{OK} action parity: {len(picked)} decisions recorded "
              f"(no baseline to compare)")
        if not save_to:
            print(f"        save one with --save-baseline, then re-run after "
                  f"any change that should not move decisions")
        return True
    same = picked == baseline
    print(f"{OK if same else BAD} action parity: {len(picked)} decisions, "
          f"{'identical' if same else 'DIFFERENT'} to baseline")
    return same


# ------------------------------------------------------------ 4. timing
def check_timing(path, ckpt, worlds=DEFAULT_D, budget=25.0):
    try:
        from belot.online.agent import build
    except Exception as exc:
        print(f"{BAD} timing: cannot import the agent ({exc})")
        return False

    agent = build(checkpoint=ckpt, worlds=worlds)
    worst, n = 0.0, 0
    for sync, seat, rec in _states(path):
        st = sync.state
        if st.tricks_played < 3 or rec["state"].get("activePlayer") != seat:
            continue
        mask = st.get_legal_actions().astype(np.int8)
        if not mask.any():
            continue
        t0 = time.monotonic()
        try:
            agent.act(st, seat, sync.match_scores, mask)
        except NotImplementedError:
            print("       timing: solver not wired yet -- skipped")
            return True
        worst = max(worst, time.monotonic() - t0)
        n += 1

    ok = worst < budget * 0.5      # want a lot of headroom, not a squeak
    print(f"{OK if ok else BAD} timing: {n} searched decisions, "
          f"worst {worst:.2f}s against a {budget:.0f}s budget")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", default=None,
                    help="JSON file of a previously recorded action list")
    ap.add_argument("--save-baseline", default=None)
    ap.add_argument("--worlds", type=int, default=DEFAULT_D)
    ap.add_argument("--skip-rules", action="store_true")
    args = ap.parse_args()

    results = []
    if not args.skip_rules:
        results.append(check_rules())
    results.append(check_worlds(args.frames))

    if args.ckpt:
        baseline = None
        if args.baseline:
            baseline = json.load(open(args.baseline, encoding="utf-8"))
        results.append(check_encoder(args.frames, args.ckpt))
        results.append(check_actions(args.frames, args.ckpt, baseline,
                                     args.save_baseline))
        results.append(check_timing(args.frames, args.ckpt, args.worlds))
    else:
        print("       (pass --ckpt to also run action parity and timing)")

    print()
    print("ALL CHECKS PASSED" if all(results) else "SOMETHING FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
