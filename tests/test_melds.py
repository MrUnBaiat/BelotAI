"""
test_melds.py -- the simulator can score the game belot.md scores.

`BelotEnv.melds` is off by default and nothing moves when it is; on, the hand is scored
with declared combinations in the bolt test and the stakes, the capot, and the third
bolt, by the one implementation in `belot/melds.py` that the deployed search also uses.
The research harness that priced the fix scores deals through its own plumbing
(`research/v10_search/x18_meld_ab.play`), so agreement with it on seeded deals checks the
env's snapshot timing, its trump/declarer/leader and its raw points, not just the formula.
"""

import numpy as np
import pytest

from belot import melds as M
from belot.env import BelotEnv
from belot.evaluation.swap_eval import BASE_SEED
from belot.heuristic import _heuristic_action


def _finish(deal, melds, bolts=(0, 0)):
    env = BelotEnv()
    env.melds = melds
    env.dealer = deal % 4
    np.random.seed(BASE_SEED + deal)
    env.reset()
    env.bolts_by_team = list(bolts)
    info = {}
    while not env.done:
        _, _, _, info = env.step(int(_heuristic_action(env)))
    return env, info["game_points"]


def _legacy(env):
    """`_calculate_final_rewards`' original arithmetic, bolts at zero."""
    gp = [0, 0]
    for t in (0, 1):
        if env.tricks_won_by_team[t] == 0:
            gp[t] = -10
    dt, dfn = env.declaring_team, env.defending_team
    if gp[0] != -10 and gp[1] != -10:
        rd, rf = env.raw_points_by_team[dt], env.raw_points_by_team[dfn]
        if rd <= 80:
            gp[dt], gp[dfn] = 0, 16
        else:
            b = rf // 10 + (1 if rf % 10 > 5 else 0)
            gp[dfn], gp[dt] = b, 16 - b
    elif gp[dfn] == -10:
        gp[dt] = 16
    else:
        gp[dfn] = 16
    return gp


def test_off_by_default_and_unchanged():
    for d in range(30):
        env, gp = _finish(d, melds=False)
        assert env.meld_totals is None
        assert gp[:2] == _legacy(env), d


def test_deal_melds_on_a_known_deal():
    # seat 0: 9-10-J-Q of diamonds (four-run, 50) and Q+K of hearts;
    # seat 1: 7-8-9 of clubs (three-run, 20); seats 2-3 nothing. Trump hearts.
    hands = [[2, 3, 4, 5, 13, 14, 20, 30],
             [16, 17, 18, 8, 9, 27, 28, 31],
             [0, 1, 10, 21, 22, 24, 25, 29],
             [6, 7, 11, 12, 15, 19, 23, 26]]
    m = M.deal_melds(hands, trump=1, leader=0)
    assert m["public"] == (50, 0)          # the four-run beats the three-run, which scores 0
    assert m["bela_seat"] == 0
    assert m["total"] == (70, 0)


# --------------------------------------------------- agreement with the SDK's copy
# The same point values live in two repositories: here, because the simulator must
# GENERATE combinations from hands, and in `belotmd`, because it must READ what the
# server settled. Neither may import the other -- the trainer must not require a
# platform SDK, and the SDK must not require a trainer -- so the duplication is
# deliberate and the drift is caught here, the way `tools/verify_online.py` catches
# rulebook drift between `BelotEnv` and `BelotState`.

def test_point_values_agree_with_the_sdk():
    combo = pytest.importorskip("belotmd.game.combinations")
    assert M.RUN_POINTS[3] == combo.RUN_POINTS[combo.TART]
    assert M.RUN_POINTS[4] == combo.RUN_POINTS[combo.JUMATE_DE_SUTA]
    assert M.RUN_POINTS[5] == combo.RUN_POINTS[combo.O_SUTA]
    assert M.BELA_POINTS == combo.BELA_POINTS
    for rank in range(8):
        assert M.quad_points(rank) == combo.FOUR_POINTS.get(rank, 100), rank


def test_an_uncontested_total_agrees_with_the_sdk_reader():
    """The SDK sums the field the server already settled; this module resolves the
    contest itself. With one team declaring and nothing against it they must agree."""
    combo = pytest.importorskip("belotmd.game.combinations")
    from belotmd.platform.protocol import ASCII_TO_ID
    assert combo.team_points(["", "2d", "", ""], ASCII_TO_ID) == (0, 50)
    assert M.resolve([[], [M.run_combo(0, 5, 4)], [], []], trump=2) == (0, 50)


# ------------------------------------------------------- the contest, owner-confirmed
# Rules 2 and 4 below are the platform owner's; the recordings confirm 4 and 5 and are
# silent on 2. Pinned here so none of them can revert silently.

def test_a_four_of_a_kind_beats_a_five_run_at_equal_points():
    """Owner's rule: at equal points a quad outranks a run, top card notwithstanding."""
    quad = [M.quad_combo(6)]                 # four kings, 100
    run5 = [M.run_combo(0, 7, 5)]            # 10-J-Q-K-A of diamonds, 100, ace high
    assert M.resolve([quad, run5, [], []], trump=2) == (100, 0)
    assert M.resolve([run5, quad, [], []], trump=2) == (0, 100)


def test_a_tie_goes_to_the_one_in_trump():
    plain = [M.run_combo(3, 4, 3)]           # J-high tercă, spades
    trumped = [M.run_combo(0, 4, 3)]         # J-high tercă, diamonds = trump
    assert M.resolve([plain, trumped, [], []], trump=0) == (0, 20)


def test_a_tie_with_neither_in_trump_cancels_BOTH_teams_ENTIRELY():
    """The recordings' clearest case: team 1 holds a K-high tercă AND a 9-high one,
    team 0 a K-high tercă in another plain suit. Published c was [0, 0] -- the
    teammate's weaker combination dies with the tie rather than inheriting it."""
    k_clubs = [M.run_combo(2, 6, 3)]
    k_spades_and_a_nine = [M.run_combo(3, 6, 3), M.run_combo(2, 2, 3)]
    assert M.resolve([k_clubs, k_spades_and_a_nine, [], []], trump=1) == (0, 0)


def test_bela_scores_through_a_cancellation():
    """Bela always scores for its holder, contest or not."""
    q_spades = [M.run_combo(3, 5, 3)]
    q_diamonds = [M.run_combo(0, 5, 3)]
    assert M.resolve([q_spades, q_diamonds, [], []], bela_seats=(1,), trump=1) == (0, 20)


def test_the_winning_team_scores_every_combination_it_holds():
    """No contest inside a team: the winner's weaker combinations score too."""
    strong_and_weak = [M.run_combo(0, 6, 4), M.run_combo(2, 2, 3)]   # 50 + 20
    weaker = [M.run_combo(3, 7, 3)]                                  # 20, ace high
    assert M.resolve([strong_and_weak, weaker, [], []], trump=1) == (70, 0)


def test_platform_points_including_the_capot():
    assert M.platform_points((100, 62), (0, 50), 0, (5, 3)) == (0, 21)     # bolted by a run
    assert M.platform_points((120, 42), (20, 20), 0, (6, 2)) == (14, 6)    # made, stakes 20
    assert M.platform_points((162, 0), (40, 100), 0, (8, 0)) == (20, -10)  # capot: theirs void
    assert M.platform_points((0, 162), (0, 0), 1, (0, 8)) == (-10, 16)
    assert M.platform_points((70, 92), (0, 0), 0, (3, 5), third_bolt=True) == (-10, 16)


def test_env_scores_its_own_final_state_the_platform_way():
    for d in range(40, 80):
        env, gp = _finish(d, melds=True)
        assert env.meld_totals is not None
        want = M.platform_points(env.raw_points_by_team, env.meld_totals,
                                 env.declaring_team, env.tricks_won_by_team)
        assert tuple(gp[:2]) == want, d
        assert gp[2:] == gp[:2]


def test_the_third_bolt_fires_on_the_same_hand_either_way():
    """Counters preset to 2 on both sides: a bolt costs the declarer 10 more, once, and
    the counter rolls over -- as in the original branch."""
    seen = 0
    for d in range(200, 400):
        env, gp = _finish(d, melds=True, bolts=(2, 2))
        dt = env.declaring_team
        c = env.meld_totals
        raw = env.raw_points_by_team
        bolted = 0 not in env.tricks_won_by_team and raw[dt] + c[dt] < raw[1 - dt] + c[1 - dt]
        if bolted:
            seen += 1
            assert gp[dt] == -10
            assert env.bolts_by_team[dt] == 0
        else:
            assert env.bolts_by_team[dt] == 2
        if seen >= 5:
            break
    assert seen >= 5, "no bolts in 200 deals?"


def test_env_agrees_with_the_research_harness():
    """Two independent plumbings of the same rule must give the same number."""
    x18 = pytest.importorskip("research.v10_search.x18_meld_ab")
    from belot.evaluation.swap_eval import _play
    h = _heuristic_action
    for d in range(500, 560):
        ours = _play(d, h, h, melds=True)
        theirs = x18.play(d, h, h, "unified", "trump")[0]
        assert ours == theirs, d
