"""
Reading belot.md's score table.

Three things distort a naive read, all observed in real captures, and each one
silently produces a plausible-looking wrong number rather than an error:

  * a bolted team's cell is the marker `BT-n`, not a score;
  * a cancelled deal appends a DUPLICATE cumulative row and scores nothing;
  * the table is cumulative, so a hand is a difference, not a cell.

The fixture below is the real final table from a recorded session. Its per-team
totals must reconcile with its own last row -- which is the check that would have
caught every one of those three.
"""

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
