"""
test_online_agent.py -- the online adapter, without a table and without a solver.

These run offline against `belotmd` alone. They pin the properties that are
expensive to get wrong live:

  * the search fires only from trick 3, never during bidding;
  * a returned action is always legal;
  * the recurrent state advances once per DECISION, including on searched
    turns, and a retry replays from the pre-turn state;
  * a mid-hand join disables the search rather than searching incomplete
    constraints.

Wire `_solve_world` and these keep working -- the stub only stands in for the
double-dummy solve.
"""

import time

import numpy as np
import pytest

from belotmd.game.state import BelotState

from belot.online.agent import SEARCH_FROM_TRICK, CompositeAgent


class StubComposite(CompositeAgent):
    """The real agent with the exact solve replaced by noise."""

    def __init__(self, **kw):
        kw.setdefault("device", "cpu")
        kw.setdefault("seed", 0)
        super().__init__(**kw)
        self.solves = 0
        self.scored = []          # (c, hands) handed to every solve

    def _solve_world(self, state, seat, hands, legal, c=(0, 0)):
        self.solves += 1
        self.scored.append((tuple(c), [list(h) for h in hands]))
        return self.rng.random(len(legal))


@pytest.fixture
def agent():
    return StubComposite(worlds=4)


def _mid_hand(tricks_played, cards_each=None):
    """A well-formed play state at a given trick."""
    cards_each = cards_each if cards_each is not None else 8 - tricks_played
    s = BelotState()
    s.phase = "PLAYING"
    s.trump, s.declarer, s.dealer = 2, 1, 3
    s.current_player, s.done = 0, False
    s.tricks_played = tricks_played
    s.declarer_has_played_trump = True
    s.hands = [list(range(p * cards_each, (p + 1) * cards_each)) for p in range(4)]
    s.graveyard = list(range(4 * cards_each, 32))
    s.current_trick, s.last_trick = [], []
    s.known_cards[:] = False
    s.impossible_cards[:] = False
    s.deadline = time.monotonic() + 20.0
    return s


def _act(agent, state):
    mask = state.get_legal_actions().astype(np.int8)
    assert mask.any(), "fixture produced no legal actions"
    action = agent.act(state, 0, [0, 0], mask)
    assert mask[action], f"returned masked-out action {action}"
    return action


def test_early_tricks_go_to_the_network():
    """Searching before trick 3 measured significantly WORSE offline
    (-0.348 +/- 0.309): with 24 unseen cards one world says almost nothing."""
    a = StubComposite(worlds=4)
    for trick in range(SEARCH_FROM_TRICK):
        _act(a, _mid_hand(trick))
    assert a.stats["searched"] == 0
    assert a.stats["network"] == SEARCH_FROM_TRICK
    assert a.solves == 0


def test_late_tricks_are_searched(agent):
    _act(agent, _mid_hand(SEARCH_FROM_TRICK))
    assert agent.stats["searched"] == 1
    assert agent.solves == agent.worlds


def test_bidding_is_never_searched(agent):
    s = BelotState()
    s.phase, s.bidding_round, s.done = "BIDDING", 1, False
    s.current_player, s.dealer = 0, 3
    s.face_up_suit = 1
    _act(agent, s)
    assert agent.stats["searched"] == 0


def test_a_mid_hand_join_disables_the_search(agent):
    """`beliefs_degraded` means constraints are INCOMPLETE, so a sampled world
    can contain a card that was already played."""
    s = _mid_hand(5)
    s.beliefs_degraded = True
    _act(agent, s)
    assert agent.stats["searched"] == 0
    assert agent.stats["degraded"] == 1


def test_the_recurrent_state_advances_once_per_decision(agent):
    """Including on searched turns -- otherwise the LSTM sees a different
    number of steps live than it did in training."""
    before = agent.snapshot()
    _act(agent, _mid_hand(SEARCH_FROM_TRICK))
    after = agent.snapshot()
    assert not _same_hidden(before, after), "hidden state did not advance"


def test_a_retry_replays_from_the_pre_turn_state(agent):
    """The SDK snapshots before a turn and restores before retrying a refused
    move, so a rejection must not advance the recurrence twice."""
    state = _mid_hand(SEARCH_FROM_TRICK)
    snap = agent.snapshot()
    _act(agent, state)
    once = agent.snapshot()

    agent.restore(snap)
    _act(agent, state)
    twice = agent.snapshot()
    assert _same_hidden(once, twice), "replay diverged from the first pass"


def test_the_clock_caps_the_search():
    """An already-expired deadline must not be searched into: overrunning
    forfeits the SEAT, not the turn."""
    a = StubComposite(worlds=1000)
    s = _mid_hand(5)
    s.deadline = time.monotonic()          # no budget at all
    _act(a, s)
    assert a.solves == 0
    assert a.stats["fallback"] == 1, "should have fallen back to the network"


def test_a_missing_clock_does_not_crash(agent):
    s = _mid_hand(SEARCH_FROM_TRICK)
    s.deadline = None
    _act(agent, s)
    assert agent.stats["searched"] == 1


# ------------------------------------------------------------ combinations
# belot.md bolts on trick points PLUS combinations and pays 16 + all/10; the search
# must score every world with the combinations the hand will actually be scored with.

def test_nothing_declared_scores_worlds_meld_free(agent):
    """The fixture's trump Q and K are already in the graveyard and nothing is
    declared, so every world is scored exactly as the offline player would."""
    _act(agent, _mid_hand(SEARCH_FROM_TRICK))
    assert agent.scored, "no world was solved"
    assert all(c == (0, 0) for c, _ in agent.scored)
    assert agent.stats["melded"] == 0


def test_declared_combinations_reach_every_solve(agent):
    """Seat 1: a four-run (50) and bela (20); seat 3: a three-run (20). Team 1
    therefore carries 90 into the bolt test, and every world must be scored with
    it -- bela is declared, so nothing is left to attribute per world."""
    s = _mid_hand(SEARCH_FROM_TRICK)
    s.combinations = ["", "2l|5k", "", "1t"]
    _act(agent, s)
    assert all(c == (0, 90) for c, _ in agent.scored)
    assert agent.stats["melded"] == 1


def test_a_hidden_bela_is_scored_per_world():
    """Trump 0: Q (5) and K (6) are unseen and undeclared. In a world where one
    seat holds both, that seat's team gets +20; otherwise nothing."""
    a = StubComposite(worlds=64)
    s = _mid_hand(SEARCH_FROM_TRICK)
    s.trump = 0
    s.declarer_has_played_trump = True
    _act(a, s)
    assert a.scored
    attributed = 0
    for c, hands in a.scored:
        holder = next((p for p, h in enumerate(hands) if 5 in h and 6 in h), None)
        if holder is None:
            assert c == (0, 0)
        else:
            attributed += 1
            assert c == ((20, 0) if holder % 2 == 0 else (0, 20))
    assert attributed > 0, "64 worlds and none put the pair in one hand"
    assert a.stats["melded"] == 0        # nothing DECLARED on the table


def test_an_older_state_without_the_field_still_searches(agent):
    s = _mid_hand(SEARCH_FROM_TRICK)
    del s.combinations
    _act(agent, s)
    assert agent.stats["searched"] == 1
    assert all(c == (0, 0) for c, _ in agent.scored)


def _same_hidden(a, b):
    import torch
    return all(torch.equal(x, y) for x, y in zip(a, b))
