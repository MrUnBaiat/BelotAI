"""
How the player is actually doing against humans.

    python tools/online_report.py                 # everything under sessions/
    python tools/online_report.py FRAMES.jsonl    # one recording

READ THE POWER LINE BEFORE THE RESULT. Offline, strength is measured on swap-paired
deals: each deal played twice with the seat pairs exchanged, so card luck cancels and
identical policies return exactly zero. **None of that is available online.** A deal
cannot be replayed, so the instrument here is the raw per-hand mean, whose standard
deviation is about 10.9 game points -- roughly twenty times the effect being looked
for. Hands needed:

    +-1.0 pts/hand     ~460
    +-0.5 pts/hand   ~1,830
    +-0.25 pts/hand  ~7,300

At the rate humans play, +-0.5 is several nights. The report prints the interval next
to those targets so it is obvious when a number is still noise, which for a long while
it will be.

WHAT IS COUNTED, AND WHAT IS NOT. Three things distort a naive read of the score
table, all of them observed in real captures:

  * a **bolted** team's cell is the marker `BT-n`, not a number. Its cumulative is
    unchanged -- it scored zero -- and reading the marker as a score is nonsense.
  * a **cancelled** deal (fewer than 14 points, or four sevens) appends a DUPLICATE
    cumulative row and scores nothing. Counting it is counting a hand that never
    happened.
  * once belot.md flags our seat as bot-controlled, the cards played at our seat are
    the platform bot's. Those hands measure the platform, not us, and are excluded.
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESSION_DIR = "sessions"

# The SDK's own frame parsing. Reused rather than reimplemented: `split_sessions`
# encodes a hard-won lesson about which key does NOT delimit a match, and a second
# implementation of it would be a second place to get that wrong.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "BelotMDPlayer", "tools"))
    from frame_inspector import bolt_marker, jparse, load, split_sessions
except ImportError:                                            # pragma: no cover
    sys.exit("could not import the SDK's frame_inspector; point PYTHONPATH at "
             "belotmd's tools/ directory")


def ci95(xs):
    """Half-width of the 95% interval on the mean."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def my_seat(recs):
    """Our seat, from the recorded player id against the players table."""
    for r in recs:
        pid = r.get("pid")
        if not pid:
            continue
        for p in r["state"].get("players", []) or []:
            if str(p.get("id")) == str(pid):
                return int(p.get("position"))
    return None


def takeover_index(recs):
    """Index of the first frame where belot.md flagged OUR seat as bot-controlled.

    Everything from there on is the platform bot playing our cards. Returns None if
    it never happened, which is the outcome to hope for.
    """
    seat = my_seat(recs)
    if seat is None:
        return None
    for i, r in enumerate(recs):
        players = r["state"].get("players", []) or []
        if seat < len(players) and players[seat].get("bot"):
            return i
    return None


def final_table(recs):
    """The last non-empty scoreTable in this span."""
    table = []
    for r in recs:
        raw = r["state"].get("scoreTable")
        if raw in (None, ""):
            continue
        t = jparse(raw, [])
        if isinstance(t, list) and t:
            table = t
    return table


def per_hand(table):
    """Cumulative rows -> one (delta_team0, delta_team1, bolted0, bolted1) per SCORED
    hand.

    A `BT-n` cell means that team was bolted: it scored zero, so its cumulative is
    carried forward unchanged. A row identical to its predecessor is a cancelled deal
    and is dropped rather than counted as a scoreless hand.
    """
    out = []
    prev = [0.0, 0.0]
    prev_row = None
    for row in table:
        if not isinstance(row, list) or len(row) < 2:
            continue
        if prev_row is not None and row == prev_row:
            prev_row = row
            continue                       # cancelled deal: duplicate row, no score
        cur, bolted = [], [False, False]
        for t in (0, 1):
            cell = row[t]
            if bolt_marker(cell) is not None:
                cur.append(prev[t])
                bolted[t] = True
            elif isinstance(cell, (int, float)) and not isinstance(cell, bool):
                cur.append(float(cell))
            else:
                cur.append(prev[t])
        out.append((cur[0] - prev[0], cur[1] - prev[1], bolted[0], bolted[1]))
        prev, prev_row = cur, row
    return out


def analyse(path):
    """One recording -> a list of per-hand results from OUR team's point of view."""
    recs = load(path)
    if not recs:
        return [], {"file": os.path.basename(path), "note": "empty"}

    seat = my_seat(recs)
    cut = takeover_index(recs)
    if cut is not None:
        recs = recs[:cut]              # drop the platform bot's play entirely

    hands = []
    matches = 0
    for span in split_sessions(recs):
        table = final_table(span)
        rows = per_hand(table)
        if not rows:
            continue
        matches += 1
        team = (seat % 2) if seat is not None else 0
        for d0, d1, b0, b1 in rows:
            ours, theirs = (d0, d1) if team == 0 else (d1, d0)
            bolt_us = b0 if team == 0 else b1
            bolt_them = b1 if team == 0 else b0
            hands.append({"diff": ours - theirs, "us": ours, "them": theirs,
                          "bolt_us": bolt_us, "bolt_them": bolt_them})

    meta = {"file": os.path.basename(path), "seat": seat, "matches": matches,
            "truncated_at_takeover": cut is not None}
    return hands, meta


def decision_mix():
    """The agent's own view of what it did, from the supervisor's session log."""
    p = os.path.join(SESSION_DIR, "log.jsonl")
    if not os.path.exists(p):
        return None
    keys = ("network", "searched", "fallback", "degraded", "infeasible",
            "solve_errors", "worlds")
    tot = {k: 0 for k in keys}
    sessions = seat_losses = 0
    worst = 0.0
    for line in open(p, encoding="utf-8"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        sessions += 1
        seat_losses += bool(row.get("seat_lost"))
        worst = max(worst, float(row.get("max_decision_s") or 0.0))
        for k in keys:
            tot[k] += int(row.get(k) or 0)
    tot.update(sessions=sessions, seat_losses=seat_losses, worst_decision_s=worst)
    return tot


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="*",
                    help=f"recordings; default is every {SESSION_DIR}/frames_*.jsonl")
    args = ap.parse_args()

    paths = args.frames or sorted(glob.glob(
        os.path.join(SESSION_DIR, "frames_*.jsonl")))
    if not paths:
        sys.exit(f"no recordings found (looked in {SESSION_DIR}/)")

    all_hands, metas = [], []
    for p in paths:
        hands, meta = analyse(p)
        meta["hands"] = len(hands)
        metas.append(meta)
        all_hands += hands

    print(f"\n{'=' * 72}\nONLINE PERFORMANCE vs HUMAN OPPONENTS\n{'=' * 72}")
    print(f"  {len(paths)} recording(s), "
          f"{sum(m.get('matches', 0) for m in metas)} match(es), "
          f"{len(all_hands)} scored hands")
    dropped = [m for m in metas if m.get("truncated_at_takeover")]
    if dropped:
        print(f"  {len(dropped)} recording(s) TRUNCATED at a seat takeover -- "
              f"everything after it was the platform bot, not us")

    if not all_hands:
        print("\n  no scored hands yet")
        return 0

    diffs = [h["diff"] for h in all_hands]
    n = len(diffs)
    mean = sum(diffs) / n
    half = ci95(diffs)
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)

    print(f"\n-- strength --")
    print(f"  pts/hand            {mean:+.3f} +- {half:.3f}   (n={n})")
    print(f"  hand win rate       {100 * wins / n:.1f}%   "
          f"(ties {100 * ties / n:.1f}%)")
    print(f"  bolted them         {100 * sum(h['bolt_them'] for h in all_hands) / n:.1f}%"
          f"    bolted ourselves  "
          f"{100 * sum(h['bolt_us'] for h in all_hands) / n:.1f}%")
    sig = abs(mean) > half if half == half else False
    print(f"  {'SIGNIFICANT' if sig else 'NOT significant'} at this sample size")

    print(f"\n-- how much more play is needed --")
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1))
    print(f"  per-hand sd {sd:.1f}  ->  hands for a given interval:")
    for target in (1.0, 0.5, 0.25):
        need = int((1.96 * sd / target) ** 2)
        state = "reached" if half <= target else f"{need - n:,} more"
        print(f"    +-{target:<5.2f} needs {need:>7,}   ({state})")
    print("  There is no swap-paired control online -- a deal cannot be replayed with")
    print("  the seats exchanged -- so this is the raw mean and it converges slowly.")

    mix = decision_mix()
    if mix:
        dec = mix["network"] + mix["searched"] + mix["fallback"]
        print(f"\n-- what the agent did ({mix['sessions']} logged sessions) --")
        if dec:
            print(f"  decisions           {dec:,}   "
                  f"network {100 * mix['network'] / dec:.1f}% / "
                  f"searched {100 * mix['searched'] / dec:.1f}% / "
                  f"fell back {100 * mix['fallback'] / dec:.1f}%")
        print(f"  worlds solved       {mix['worlds']:,}")
        print(f"  worst decision      {mix['worst_decision_s']:.2f}s "
              f"of a 25s budget")
        if mix["degraded"]:
            print(f"  degraded beliefs    {mix['degraded']:,} decisions fell back "
                  f"(joined mid-hand; not searchable)")
        if mix["seat_losses"]:
            print(f"  SEAT LOSSES         {mix['seat_losses']} -- check the log "
                  f"for a refused move or a slow turn just before each")
        if mix["solve_errors"]:
            print(f"  SOLVER FAULTS       {mix['solve_errors']} -- should be zero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
