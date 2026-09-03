"""
The supervisor's stop conditions.

These are the rules that protect the account, and a live run does not exercise them
until it is already too late to learn they were wrong. So they are tested here against
a fake session runner: no network, no table, no agent.

The property that matters most: **the supervisor must stop** when something is
systematically wrong, rather than reconnecting forever with a broken agent. Losing the
seat repeatedly means every subsequent frame is the platform bot's play recorded as if
it were ours, which quietly poisons the dataset as well as wasting the run.
"""

import asyncio
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.play_online import supervise


def _args(**kw):
    base = dict(hours=10.0, break_min=0.0, max_seat_losses=3, max_short=5,
                max_sessions=0, once=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _row(seconds=600.0, seat_lost=False):
    return {"seconds": seconds, "seat_lost": seat_lost}


def _runner(rows):
    """A fake `one_session` that returns canned rows, then a long clean one forever."""
    seq = list(rows)
    calls = []

    async def run(agent, args):
        calls.append(1)
        return seq.pop(0) if seq else _row()

    run.calls = calls
    return run


def _run(args, runner, monkeypatch):
    # breaks and backoff are real sleeps; make them instant
    async def nosleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", nosleep)
    return asyncio.run(supervise(None, args, run_session=runner))


def test_a_single_clean_session_stops_with_once(monkeypatch):
    r = _runner([_row()])
    assert _run(_args(once=True), r, monkeypatch) == "--once"
    assert len(r.calls) == 1


def test_repeated_seat_losses_stop_the_run(monkeypatch):
    """The expensive failure. Three in a row and we stop rather than feed the
    recorder the platform bot's play under our own seat."""
    r = _runner([_row(seat_lost=True)] * 5)
    assert _run(_args(max_seat_losses=3), r, monkeypatch) == "seat lost repeatedly"
    assert len(r.calls) == 3


def test_one_seat_loss_does_not_stop_the_run(monkeypatch):
    """A single takeover can be bad luck; the counter resets on a clean session."""
    r = _runner([_row(seat_lost=True), _row(), _row(seat_lost=True), _row()])
    assert _run(_args(max_seat_losses=3, max_sessions=4), r, monkeypatch) \
        == "reached --max-sessions 4"
    assert len(r.calls) == 4


def test_sessions_that_end_immediately_stop_the_run(monkeypatch):
    """A dead cookie or a refused join looks like a table dissolving instantly.
    Reconnecting into that is a retry loop against a wall."""
    r = _runner([_row(seconds=2.0)] * 9)
    assert _run(_args(max_short=5), r, monkeypatch) \
        == "sessions keep ending immediately"
    assert len(r.calls) == 5


def test_a_clean_session_resets_the_short_counter(monkeypatch):
    r = _runner([_row(seconds=2.0), _row(seconds=2.0), _row(),
                 _row(seconds=2.0), _row()])
    assert _run(_args(max_short=3, max_sessions=5), r, monkeypatch) \
        == "reached --max-sessions 5"
    assert len(r.calls) == 5


def test_max_sessions_is_honoured(monkeypatch):
    r = _runner([])
    assert _run(_args(max_sessions=7), r, monkeypatch) \
        == "reached --max-sessions 7"
    assert len(r.calls) == 7


def test_seat_loss_takes_priority_over_a_short_session(monkeypatch):
    """A lost seat usually also ends the session quickly. It must be counted as a
    seat loss, which stops sooner, not merely as a short session."""
    r = _runner([_row(seconds=5.0, seat_lost=True)] * 4)
    assert _run(_args(max_seat_losses=2, max_short=5), r, monkeypatch) \
        == "seat lost repeatedly"
    assert len(r.calls) == 2
