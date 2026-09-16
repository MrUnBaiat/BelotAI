"""
test_isolation.py -- one agent's decisions cannot depend on another's cards.

Two of our accounts now play the same table as partners, in two processes. The
platform keeps the hands apart (a frame carries `cards` for its own player
only), and the SDK now refuses a frame that breaks that. This file pins the
remaining half of the guarantee, on our side: even if another seat's real cards
were somehow present in the state, the decision must not move.

The instrument is a permutation. The three other seats' hands are rotated among
themselves, which leaves everything legitimately visible identical -- our own
hand, every hand SIZE, the graveyard, so the set of unseen cards too -- and
changes only who holds what. A decision that shifts under it is reading
something it must not.
"""

import numpy as np
import pytest

from belot.online.agent import CompositeAgent
from test_online_agent import StubComposite, _mid_hand


def _rotate_others(state):
    """Same cards, same sizes, different seats -- for the three seats that are
    not ours."""
    rotated = state.hands[:]
    rotated[1], rotated[2], rotated[3] = (list(state.hands[2]),
                                          list(state.hands[3]),
                                          list(state.hands[1]))
    state.hands = rotated
    return state


def _decide(agent, state, seed=0):
    """One decision from a clean start, so two calls are comparable."""
    agent.rng = np.random.default_rng(seed)
    agent.reset()
    agent.scored = []
    mask = state.get_legal_actions().astype(np.int8)
    action = agent.act(state, 0, [0, 0], mask)
    assert mask[action], "returned a masked-out action"
    return action, list(agent.scored)


def test_the_permutation_really_changes_the_other_hands():
    """Guard on the instrument: if the rotation were a no-op, everything below
    would pass without proving anything."""
    plain = _mid_hand(4)
    moved = _rotate_others(_mid_hand(4))

    assert moved.hands[0] == plain.hands[0], "our own hand must not move"
    assert [len(h) for h in moved.hands] == [len(h) for h in plain.hands]
    assert moved.hands[1:] != plain.hands[1:], "the rotation did nothing"
    assert sorted(sum(moved.hands[1:], [])) == sorted(sum(plain.hands[1:], []))


def test_the_search_ignores_who_holds_what():
    """Trick 4, so the search is running. It may use hand SIZES and public
    cards; it may not use another seat's actual hand."""
    agent = StubComposite(worlds=8)

    before, worlds_before = _decide(agent, _mid_hand(4))
    after, worlds_after = _decide(agent, _rotate_others(_mid_hand(4)))

    assert before == after, "the partner's cards changed our card"
    assert worlds_before == worlds_after, (
        "the sampled worlds moved, so the real hands reached the sampler")
    assert worlds_before, "the search did not run; this test proves nothing"


def test_the_network_ignores_who_holds_what():
    """The same property for the other decision path: no search, so the answer
    comes from the observation the encoder builds."""
    agent = StubComposite(worlds=8, search=False)

    before, _ = _decide(agent, _mid_hand(4))
    after, _ = _decide(agent, _rotate_others(_mid_hand(4)))

    assert before == after, "the observation leaked another seat's hand"


def test_bidding_ignores_who_holds_what():
    """Bidding sees five cards and the face-up card, and nothing else."""
    agent = StubComposite(worlds=8)
    state = _mid_hand(0, cards_each=5)
    state.phase = "BIDDING"
    state.face_up_card = 30
    state.trump = None
    state.declarer = None

    rotated = _rotate_others(_mid_hand(0, cards_each=5))
    rotated.phase = "BIDDING"
    rotated.face_up_card = 30
    rotated.trump = None
    rotated.declarer = None

    assert _decide(agent, state)[0] == _decide(agent, rotated)[0]


# ----------------------------------------------------- two agents, no sharing
#
# The pair runs as two processes, so nothing CAN be shared. These tests make
# that a property of the code rather than of how we happen to launch it -- the
# next person to run both in one process should not find out the hard way.

def _is_fresh(hidden):
    """A recurrent state that has never been advanced is all zeros."""
    return all(float(t.abs().sum()) == 0.0 for t in hidden)


def test_two_agents_share_no_state():
    a, b = StubComposite(worlds=4), StubComposite(worlds=4)

    _decide(a, _mid_hand(4))

    assert not _is_fresh(a.hidden), "the acting agent did not advance; no test"
    assert _is_fresh(b.hidden), "one agent's recurrent state reached the other"
    assert all(v == 0 for v in b.stats.values()), "shared counters"
    assert a.hidden is not b.hidden
    assert a.rng is not b.rng
    assert a.stats is not b.stats


def test_one_agents_sampler_does_not_disturb_the_other():
    """A shared RNG would not leak cards, but it would couple the two searches
    -- and it is exactly the kind of module-level state this project has been
    bitten by before."""
    a, b = StubComposite(worlds=4, seed=7), StubComposite(worlds=4, seed=7)

    expected = b.rng.random(3)
    a.rng = np.random.default_rng(7)
    a.rng.random(3)
    _decide(a, _mid_hand(4), seed=7)

    assert np.array_equal(np.random.default_rng(7).random(3), expected), (
        "the generators are not independent")


def test_the_agent_holds_nothing_at_class_level():
    """The cheap structural check behind the two above: per-instance state has
    to be set in __init__, not on the class, or every agent shares it."""
    for name in ("hidden", "rng", "stats", "max_decision_s"):
        assert not hasattr(CompositeAgent, name), (
            f"{name} lives on the class, so two agents would share it")


@pytest.mark.parametrize("attr", ["hidden", "stats"])
def test_advancing_one_agent_leaves_the_other_untouched(attr):
    a, b = StubComposite(worlds=4), StubComposite(worlds=4)
    before = getattr(b, attr)
    if isinstance(before, dict):
        before = dict(before)

    for _ in range(3):
        _decide(a, _mid_hand(4))

    after = getattr(b, attr)
    if isinstance(after, dict):
        assert after == before
    else:
        assert after is before
