"""
Reading belot.md recordings (`sessions/frames_*.jsonl`).

Copied from the belotmd SDK's `tools/frame_inspector.py` (commit 01e81fb). Those helpers
live outside the SDK's installable package, so this repository used to reach into a
neighbouring `BelotMDPlayer/tools` folder by name to import them. Keeping the few
functions it needs here removes that folder dependency; `tests/test_online_report.py`
holds them to the same behaviour.

Each recording line is one frame: `{"t": unix time, "pid": our player id, "state": the
server's state payload}`.
"""

import json


def load(path):
    """Every frame of a recording, skipping (and reporting) unreadable lines."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"  [skip line {n}: {e}]")
    return out


def jparse(v, default):
    """The server sends several fields as JSON *strings*; parse them if so."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return default
    return v if v is not None else default


def bolt_marker(cell):
    """'BT-3' / 'bt_3' / 'BT3' -> 3.  Anything else -> None."""
    if not isinstance(cell, str):
        return None
    s = cell.strip().upper().replace("_", "-")
    if not s.startswith("BT"):
        return None
    digits = "".join(c for c in s if c.isdigit())
    return int(digits) if digits else 0


def split_sessions(recs):
    """A recording file is opened in APPEND mode, so it can hold several matches. A
    boundary is a scoreTable that SHRINKS (a fresh match starts with an empty table and
    only ever grows) or a long wall-clock gap (the process was restarted).

    NOT gameStartTime. It looks like the obvious key and it is not stable within a
    single match -- observed live, it changed three times in one match, which split it
    into a lobby fragment, the match, and an end-of-match fragment."""
    sessions, cur = [], []
    prev_t = prev_rows = None
    for r in recs:
        st = r.get("state", {})
        t = r.get("t")
        rows = len(jparse(st.get("scoreTable", "[]"), []) or [])
        boundary = (
            (prev_t is not None and t is not None and t - prev_t > 600)
            or (prev_rows is not None and rows < prev_rows)
        )
        if boundary and cur:
            sessions.append(cur)
            cur = []
        cur.append(r)
        prev_t, prev_rows = t, rows
    if cur:
        sessions.append(cur)
    return sessions
