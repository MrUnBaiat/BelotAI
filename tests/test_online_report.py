"""
Reading belot.md's score table.

Four things distort a naive read, all observed in real captures, and each one
silently produces a plausible-looking wrong number rather than an error:

  * a bolted team's cell is the marker `BT-n`, not a score;
  * a cancelled deal appends a DUPLICATE cumulative row and scores nothing;
  * the table is cumulative, so a hand is a difference, not a cell;
  * **our seat is not fixed** -- the host may rotate players around the table
    before the match starts, and since the team is `seat % 2`, reading the seat
    from the first frame can invert the sign of every hand in a recording.

The fixture below is the real final table from a recorded session. Its per-team
totals must reconcile with its own last row -- which is the check that would have
caught every one of those four.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

# `online_report` puts the SDK's tools/ on the path itself, so import it first and
# skip only if that machinery is genuinely absent.
online_report = pytest.importorskip(
    "online_report", reason="needs the belotmd SDK's frame_inspector")
ci95, per_hand = online_report.ci95, online_report.per_hand

# A real captured match. Row 7 duplicates row 6: a cancelled deal.
REAL = [[1, 17], [17, "BT-1"], [27, 23], [43, 25], [48, 36], ["BT-1", 54],
        [59, 59], [59, 59], [77, "BT-2"], [90, 62], [108, "BT-3"]]


def test_a_cancelled_deal_is_not_a_hand():
    """Eleven rows, one of them a duplicate, is ten hands -- not eleven, and not
    eleven with a scoreless one."""
    assert len(REAL) == 11
    assert len(per_hand(REAL)) == 10


def test_totals_reconcile_with_the_final_row():
    """The arithmetic check that catches all three traps at once: summing the
    per-hand deltas must land exactly on the table's own last cumulative row."""
    rows = per_hand(REAL)
    assert round(sum(r[0] for r in rows)) == 108
    assert round(sum(r[1] for r in rows)) == 62


def test_a_bolted_team_scores_zero_not_a_marker():
    """`BT-n` is a marker, not a number. The bolted team's cumulative is carried
    forward unchanged, and the hand is flagged as a bolt."""
    rows = per_hand(REAL)
    d0, d1, b0, b1 = rows[1]              # [17, 'BT-1']
    assert b1 and not b0
    assert d1 == 0.0, "a bolted team scored something"
    assert d0 == 16.0

    d0, d1, b0, b1 = rows[5]              # ['BT-1', 54] -- scored hand 5
    assert b0 and not b1
    assert d0 == 0.0
    assert d1 == 18.0                     # 54 - 36, sixteen plus a declaration


def test_every_hand_is_a_difference_not_a_cell():
    """Cumulative rows: the first hand is the row itself, later ones are deltas."""
    rows = per_hand(REAL)
    assert rows[0][:2] == (1.0, 17.0)     # from an implicit [0, 0]
    assert rows[2][:2] == (10.0, 6.0)     # [27,23] minus [17,17-carried]


def test_an_empty_table_yields_no_hands():
    assert per_hand([]) == []
    assert per_hand([["", ""]]) == [(0.0, 0.0, False, False)]


def test_ci95_is_nan_on_a_single_observation():
    """One hand cannot have an interval, and reporting 0.000 would read as
    certainty about a sample of one."""
    import math
    assert math.isnan(ci95([3.0]))
    assert ci95([1.0, 2.0, 3.0]) > 0


# --------------------------------------------------------------- the seat

PID = "me-42"


def _frame(seat, phase, table, bot=False):
    """One recorded frame with us sitting at `seat`."""
    players = [{"id": f"p{i}", "position": i, "bot": False} for i in range(4)]
    players[seat] = {"id": PID, "position": seat, "bot": bot}
    return {"pid": PID,
            "state": {"currentPhase": phase, "players": players,
                      "scoreTable": json.dumps(table)}}


def _recording(tmp_path, frames, name="frames_x.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
    return str(p)


# Team 0 wins every hand by 16-0. Whichever seat we hold decides the sign.
TABLE = [[16, 0], [32, 0], [48, 0]]


def test_the_seat_comes_from_play_not_from_the_first_frame(tmp_path):
    """The bug that cost 7 of the first 109 hands their sign.

    We sit at seat 1 in the lobby, the host rotates us to seat 2 before the deal,
    and we play the whole match from seat 2 -- team 0, which won every hand here.
    Reading the lobby seat instead gives team 1 and inverts all three.
    """
    frames = ([_frame(1, 0, [])] * 3          # lobby, pre-rotation
              + [_frame(2, 2, [])]            # rotated, deck cut
              + [_frame(2, 6, TABLE[:1]),
                 _frame(2, 10, TABLE[:2]),
                 _frame(2, 14, TABLE)])
    path = _recording(tmp_path, frames)

    assert online_report.my_seat(online_report.load(path)) == 2
    hands, meta = online_report.analyse(path)
    assert meta["seat"] == [2]
    assert [h["diff"] for h in hands] == [16.0, 16.0, 16.0]


def _table_frame(phase, table, rnd, bot_seats=()):
    """Us at seat 0 (seat 2 is our partner). `bot_seats` are played by belot.md's bot."""
    players = [{"id": f"p{i}", "position": i, "bot": i in bot_seats} for i in range(4)]
    players[0] = {"id": PID, "position": 0, "bot": False}
    return {"pid": PID, "state": {"currentPhase": phase, "round": rnd,
                                  "players": players, "scoreTable": json.dumps(table)}}


def test_each_hand_is_tagged_with_the_bot_seats_played_during_it(tmp_path):
    """Hand 2 had a bot opponent, hand 3 a bot partner; hand 1 was all human.
    The tag follows the hand being played when its row landed, and the bot that
    was still seated in the lobby before hand 1 does not count."""
    frames = [_table_frame(2, [], 1, bot_seats=(1,)),        # deal: not counted
              _table_frame(10, [], 1),
              _table_frame(13, TABLE[:1], 1),
              _table_frame(10, TABLE[:1], 2, bot_seats=(3,)),
              _table_frame(13, TABLE[:2], 2),
              _table_frame(10, TABLE[:2], 3, bot_seats=(2,)),
              _table_frame(14, TABLE, 3)]
    hands, _ = online_report.analyse(_recording(tmp_path, frames))
    assert [h["bots"] for h in hands] == [[], ["opponent"], ["partner"]]
    assert [h["diff"] for h in hands] == [16.0, 16.0, 16.0]


def test_a_hand_already_on_the_table_when_recording_began_is_unobserved(tmp_path):
    frames = [_table_frame(10, TABLE[:1], 2),                 # joined after hand 1
              _table_frame(13, TABLE[:2], 2, bot_seats=(1,)),
              _table_frame(10, TABLE[:2], 3),
              _table_frame(14, TABLE, 3)]
    hands, _ = online_report.analyse(_recording(tmp_path, frames))
    assert [h["bots"] for h in hands] == [None, [], []]


def test_a_seat_held_throughout_is_unaffected(tmp_path):
    """The control: with no rotation the answer must not move."""
    frames = [_frame(1, 0, []), _frame(1, 6, TABLE[:1]),
              _frame(1, 10, TABLE[:2]), _frame(1, 14, TABLE)]
    hands, meta = online_report.analyse(_recording(tmp_path, frames))
    assert meta["seat"] == [1]
    assert [h["diff"] for h in hands] == [-16.0, -16.0, -16.0]


def test_a_takeover_is_found_at_our_rotated_seat(tmp_path):
    """`takeover_index` resolves the seat per frame, so a rotation cannot make it
    read some other player's `bot` flag -- or miss ours."""
    frames = [_frame(1, 0, []), _frame(2, 6, TABLE[:1]),
              _frame(2, 10, TABLE[:2], bot=True), _frame(2, 14, TABLE)]
    recs = online_report.load(_recording(tmp_path, frames))
    assert online_report.takeover_index(recs) == 2
    # and the hands after it are dropped rather than credited to us
    hands, meta = online_report.analyse(_recording(tmp_path, frames))
    assert meta["truncated_at_takeover"]
    assert len(hands) == 1


def test_a_table_that_does_not_reconcile_is_reported():
    """A silent wrong number is the failure mode this whole module guards
    against, so the arithmetic is checked against the server's own final row."""
    assert online_report.reconcile(REAL, per_hand(REAL)) == []
    # one hand dropped: the remaining deltas can no longer reach [108, 62]
    assert online_report.reconcile(REAL, per_hand(REAL)[1:])
