"""
test_platform_scoring.py -- the hand is scored the way belot.md scores it.

The simulator has no melds, so the solver's original conversion (`gp_diff_from_raw`)
uses a fixed bolt line and fixed 16-point stakes. The platform bolts on trick points
PLUS combination points and pays 16 + all combinations/10 -- a rule read off 897
recorded hands and reproduced on every one of them (research/v10_search FINDINGS §13).

Two things are pinned here:

  * with no combinations the platform conversion is the ORIGINAL one, exactly, on
    every reachable raw total -- so nothing measured offline moves;
  * with combinations it follows the recorded rule on hand-checked cases.
"""

import numpy as np
import pytest

from belot.search.composite import (TOTAL_RAW, gp_diff_from_raw, gp_diff_platform,
                                    platform_points, solve_world)


def test_no_combinations_is_the_original_conversion_everywhere():
    for raw0 in range(TOTAL_RAW + 1):
        for dt in (0, 1):
            for team in (0, 1):
                assert (gp_diff_platform(raw0, dt, team, (0, 0))
                        == gp_diff_from_raw(raw0, dt, team)), (raw0, dt, team)


@pytest.mark.parametrize("raw, c, dt, expected", [
    # made, no melds: defenders round 62 -> 6, declarer 10
    ((100, 62), (0, 0), 0, (10, 6)),
    # the fixed line: 80 is a bolt, 81 is not
    ((80, 82), (0, 0), 0, (0, 16)),
    ((81, 81), (0, 0), 0, (8, 8)),
    # a declarer on 100 trick points is BOLTED by a defending 50-run: 100 < 112.
    # The defenders take 16 + 5.
    ((100, 62), (0, 50), 0, (0, 21)),
    # the declarer's own melds count for it: 70 + 20 = 90 < 92 still bolts,
    # and the defenders take the declarer's 20 as well
    ((70, 92), (20, 0), 0, (0, 18)),
    # ...while a bigger one SAVES it: 70 + 40 = 110 > 92. Stakes 20, defenders 9.
    ((70, 92), (40, 0), 0, (11, 9)),
    # made with melds on both sides: stakes 20, defenders round 42 + 20 = 62 -> 6
    ((120, 42), (20, 20), 0, (14, 6)),
    # a tie on trick + combination points rounds each side's own total
    ((91, 71), (0, 20), 0, (9, 9)),
    # the same hand from the other side of the table
    ((62, 100), (50, 0), 1, (21, 0)),
])
def test_recorded_rule_on_hand_checked_cases(raw, c, dt, expected):
    assert platform_points(raw, c, dt) == expected


def test_stakes_grow_with_every_combination_on_the_table():
    """A bolt hands the defenders 16 + ALL combinations/10 -- the declarer's too."""
    assert platform_points((60, 102), (20, 40), 0) == (0, 22)      # 80 < 142
    assert platform_points((60, 102), (0, 0), 0) == (0, 16)
    # ...and a big enough meld of the declarer's own turns the same trick score
    # into a MADE hand: 60 + 100 = 160 > 142. Stakes 30, defenders round 142 -> 14.
    assert platform_points((60, 102), (100, 40), 0) == (16, 14)


def test_solve_world_with_no_combinations_is_bit_identical():
    """`solve_world(..., c=(0, 0))` must take the ORIGINAL code path, not an
    equivalent one, so the offline player is unchanged by construction."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        deck = rng.permutation(32).tolist()
        n = int(rng.integers(2, 6))
        hands = [sorted(deck[8 * s:8 * s + n]) for s in range(4)]
        seat, trump, declarer = 0, int(rng.integers(4)), int(rng.integers(4))
        raw0 = int(rng.integers(0, 60))
        base = solve_world(hands, seat, (), trump, declarer, True, raw0,
                           declarer % 2, hands[seat])
        again = solve_world(hands, seat, (), trump, declarer, True, raw0,
                            declarer % 2, hands[seat], c=(0, 0))
        assert np.array_equal(base, again)
        aslist = solve_world(hands, seat, (), trump, declarer, True, raw0,
                             declarer % 2, hands[seat], c=[0, 0])
        assert np.array_equal(base, aslist)


def test_the_third_bolt_costs_the_declarer_ten_more_and_nothing_else():
    """`env._calculate_final_rewards`: a bolt when the counter stands at 2 scores -10
    for the bolted team. Not on a made hand, not on a tie."""
    assert platform_points((70, 92), (0, 0), 0) == (0, 16)
    assert platform_points((70, 92), (0, 0), 0, third_bolt=True) == (-10, 16)
    assert platform_points((92, 70), (0, 0), 1, third_bolt=True) == (16, -10)
    assert platform_points((100, 62), (0, 0), 0, third_bolt=True) == (10, 6)
    assert platform_points((81, 81), (0, 0), 0, third_bolt=True) == (8, 8)
    # stacks with combinations: the defenders still take 16 + all/10
    assert platform_points((60, 102), (20, 40), 0, third_bolt=True) == (-10, 22)
    for raw0 in range(TOTAL_RAW + 1):
        for dt in (0, 1):
            for team in (0, 1):
                assert (gp_diff_platform(raw0, dt, team, (0, 0), third_bolt=False)
                        == gp_diff_from_raw(raw0, dt, team))


def test_research_scorer_agrees_with_the_deployed_one():
    """The rule was validated in research/; the deployed copy must not drift."""
    melds = pytest.importorskip("research.v10_search.core.melds")
    for raw0 in range(0, TOTAL_RAW + 1, 3):
        raw = (raw0, TOTAL_RAW - raw0)
        for c in ((0, 0), (20, 0), (0, 50), (100, 20), (70, 70), (150, 0)):
            for dt in (0, 1):
                for third in (False, True):
                    assert (platform_points(raw, c, dt, third_bolt=third)
                            == melds.platform_points(raw, c, dt, third_bolt=third))
