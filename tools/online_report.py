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

AND THE ONE THAT ACTUALLY BIT. **Our seat is not fixed for a recording.** Before a match
starts the host may rotate players around the table to set up teams (close code 4005);
nobody is removed, only seat indices move. Reading our seat from the FIRST frame -- which
this tool used to do -- therefore reads a PRE-ROTATION seat, and since the team is
`seat % 2`, a rotation of odd parity silently inverts the sign of every hand in that
recording. It did: 7 of the first 109 hands were counted for the wrong team, moving the
headline from +0.174 to -1.294. The seat is now taken from frames where the match was
actually being played, per span, and each span's per-hand deltas are reconciled against
its own final cumulative row -- the arithmetic check that would have caught it.
"""

import argparse
import collections
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESSION_DIR = "sessions"

# A match is under way from the first bid onward. Frames below this phase are the
# lobby, where the host can still rotate seats -- so they are exactly the frames a
# seat must NOT be read from.
MATCH_PHASE = 6                                    # protocol.TRUMP_CHOOSE_1

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


def seat_in_frame(rec):
    """Our seat in ONE frame, resolved by player id rather than remembered.

    Resolving per frame is what makes this robust to a lobby seat rotation: the
    players table in the frame is the authority on where we are sitting at that
    moment, and it costs nothing to ask it every time.
    """
    pid = rec.get("pid")
    if not pid:
        return None
    for p in rec["state"].get("players", []) or []:
        if str(p.get("id")) == str(pid):
            return int(p.get("position"))
    return None


def my_seat(recs, playing_only=True):
    """The seat we actually PLAYED, as the modal seat over in-match frames.

    `playing_only=False` falls back to any frame, for spans that never started a
    match (a table that dissolved in the lobby) -- those carry no scored hands, so
    the answer only affects diagnostics.
    """
    for require_match in ((True, False) if playing_only else (False,)):
        counts = collections.Counter()
        for r in recs:
            if require_match and r["state"].get("currentPhase", 0) < MATCH_PHASE:
                continue
            seat = seat_in_frame(r)
            if seat is not None:
                counts[seat] += 1
        if counts:
            return counts.most_common(1)[0][0]
    return None


def takeover_index(recs):
    """Index of the first frame where belot.md flagged OUR seat as bot-controlled.

    Everything from there on is the platform bot playing our cards. Returns None if
    it never happened, which is the outcome to hope for.

    The seat is resolved per frame, so a lobby rotation cannot make this read some
    other player's `bot` flag.
    """
    for i, r in enumerate(recs):
        seat = seat_in_frame(r)
        if seat is None:
            continue
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


def reconcile(table, rows):
    """Summed per-hand deltas must land on the table's own final cumulative row.

    The check that would have caught the seat-parity bug, the bolt markers and the
    cancelled-deal duplicates alike: it compares this tool's arithmetic against a
    number the server itself published. A cell carrying a `BT-n` marker states no
    total, so it is skipped rather than guessed at.
    """
    problems = []
    last = next((r for r in reversed(table)
                 if isinstance(r, list) and len(r) >= 2), None)
    if last is None or not rows:
        return problems
    for t in (0, 1):
        cell = last[t]
        if bolt_marker(cell) is not None:
            continue
        if not isinstance(cell, (int, float)) or isinstance(cell, bool):
            continue
        total = sum(r[t] for r in rows)
        if abs(total - float(cell)) > 1e-9:
            problems.append(f"team {t}: hands sum to {total:g}, but the table's "
                            f"own final row says {cell}")
    return problems


def analyse(path):
    """One recording -> a list of per-hand results from OUR team's point of view."""
    recs = load(path)
    if not recs:
        return [], {"file": os.path.basename(path), "note": "empty"}

    cut = takeover_index(recs)
    if cut is not None:
        recs = recs[:cut]              # drop the platform bot's play entirely

    hands = []
    matches = 0
    seats, problems = set(), []
    for span in split_sessions(recs):
        table = final_table(span)
        rows = per_hand(table)
        if not rows:
            continue
        matches += 1
        # Per span, not per file: a recording can hold several matches, and the
        # host reseats between them.
        seat = my_seat(span)
        seats.add(seat)
        problems += reconcile(table, rows)
        team = (seat % 2) if seat is not None else 0
        for d0, d1, b0, b1 in rows:
            ours, theirs = (d0, d1) if team == 0 else (d1, d0)
            bolt_us = b0 if team == 0 else b1
            bolt_them = b1 if team == 0 else b0
            hands.append({"diff": ours - theirs, "us": ours, "them": theirs,
                          "bolt_us": bolt_us, "bolt_them": bolt_them})

    meta = {"file": os.path.basename(path), "seat": sorted(s for s in seats
                                                           if s is not None),
            "matches": matches, "truncated_at_takeover": cut is not None,
            "problems": problems}
    return hands, meta


def decision_mix():
    """The agent's own view of what it did, from the supervisor's session log.

    `hands_dealt` comes from the synchronizer's own hand counter. The older
    `hand_boundaries` field is deliberately not read: the SDK calls `agent.reset()`
    on every lobby and end-phase frame, not once per hand, so across the first 13
    live sessions it counted 326 against 119 hands actually dealt. Old rows are
    still summed for everything else.
    """
    p = os.path.join(SESSION_DIR, "log.jsonl")
    if not os.path.exists(p):
        return None
    keys = ("network", "searched", "fallback", "degraded", "infeasible",
            "solve_errors", "worlds", "hands_dealt", "tables", "seat_losses")
    tot = {k: 0 for k in keys}
    sessions = 0
    worst = 0.0
    stops = collections.Counter()
    for line in open(p, encoding="utf-8"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        sessions += 1
        worst = max(worst, float(row.get("max_decision_s") or 0.0))
        for k in keys:
            tot[k] += int(row.get(k) or 0)
        # Rows written before seat losses were counted carry a bool instead.
        if "seat_losses" not in row and row.get("seat_lost"):
            tot["seat_losses"] += 1
        if row.get("stop_reason"):
            stops[row["stop_reason"]] += 1
        if row.get("error"):
            stops[f"ERROR {row['error']}"] += 1
    tot.update(sessions=sessions, worst_decision_s=worst, stops=stops)
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
    bad = [(m, p) for m in metas for p in m.get("problems") or []]
    if bad:
        print(f"\n  SCORE TABLE DOES NOT RECONCILE -- treat every number below as "
              f"suspect:")
        for m, p in bad:
            print(f"    {m['file']}: {p}")

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
        if mix["hands_dealt"]:
            print(f"  hands dealt         {mix['hands_dealt']:,} "
                  f"at {mix['tables']:,} table(s)   "
                  f"({len(all_hands):,} of them scored)")
        if dec:
            print(f"  decisions           {dec:,}   "
                  f"network {100 * mix['network'] / dec:.1f}% / "
                  f"searched {100 * mix['searched'] / dec:.1f}% / "
                  f"fell back {100 * mix['fallback'] / dec:.1f}%")
        print(f"  worlds solved       {mix['worlds']:,}")
        print(f"  worst decision      {mix['worst_decision_s']:.2f}s "
              f"of a 25s budget")
        if mix["stops"]:
            print("  how stretches ended:")
            for why, n in mix["stops"].most_common():
                print(f"    {n:3d}x  {why}")
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
