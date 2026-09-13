"""
The supervisor's stop conditions.

These are the rules that protect the account, and a live run does not exercise them
until it is already too late to learn they were wrong. So they are tested here
against fakes: no network, no table, no agent.

Two layers, because the SDK now owns reconnection and `bot.run()` no longer returns
between tables:

  * `_watch` runs BESIDE a live bot and is the only thing that can see a seat
    takeover, a silent run, or a stretch deadline while play is under way;
  * `supervise` decides whether to start another stretch.

The property that matters most: **the supervisor must stop** when something is
systematically wrong, rather than playing on with a lost seat. Once belot.md flags
our seat, every subsequent frame is the platform bot's play recorded as if it were
ours, which quietly poisons the dataset as well as wasting the run.
"""

import asyncio
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.play_online as P
from scripts.play_online import supervise


def _args(**kw):
    base = dict(hours=10.0, break_min=0.0, max_seat_losses=3, max_short=5,
                max_idle_min=60.0, max_sessions=0, once=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _no_interrupt(monkeypatch):
    monkeypatch.setattr(P, "_interrupted", False)


# ------------------------------------------------------------- supervise()

def _row(seconds=600.0, seat_losses=0, idle=False):
    return {"seconds": seconds, "seat_losses": seat_losses,
            "stopped_for_idle": idle}


def _runner(rows):
    """A fake stretch that returns canned rows, then long clean ones forever."""
    seq = list(rows)
    calls = []

    async def run(agent, args):
        calls.append(1)
        return seq.pop(0) if seq else _row()

    run.calls = calls
    return run


def _run(args, runner, monkeypatch):
    async def nosleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", nosleep)
    return asyncio.run(supervise(None, args, run_session=runner))


def test_a_single_clean_stretch_stops_with_once(monkeypatch):
    r = _runner([_row()])
    assert _run(_args(once=True), r, monkeypatch) == "--once"
    assert len(r.calls) == 1


def test_repeated_seat_losses_stop_the_run(monkeypatch):
    """The expensive failure. Three and we stop rather than feed the recorder the
    platform bot's play under our own seat."""
    r = _runner([_row(seat_losses=3)] * 5)
    assert _run(_args(max_seat_losses=3), r, monkeypatch) == "seat lost repeatedly"
    assert len(r.calls) == 1


def test_seat_losses_accumulate_across_stretches(monkeypatch):
    """Two in one stretch and two in the next is the same problem as four in one."""
    r = _runner([_row(seat_losses=2), _row(seat_losses=2), _row()])
    assert _run(_args(max_seat_losses=3), r, monkeypatch) == "seat lost repeatedly"
    assert len(r.calls) == 2


def test_a_clean_stretch_resets_the_seat_loss_counter(monkeypatch):
    """A single takeover can be bad luck; a clean stretch clears the count."""
    r = _runner([_row(seat_losses=1), _row(), _row(seat_losses=1), _row()])
    assert _run(_args(max_seat_losses=3, max_sessions=4), r, monkeypatch) \
        == "reached --max-sessions 4"
    assert len(r.calls) == 4


def test_a_silent_run_stops_the_supervisor(monkeypatch):
    """The SDK retries an empty lobby forever. An expired cookie looks identical
    from out here, so a stretch that saw no frames at all is not retried."""
    r = _runner([_row(idle=True)])
    assert _run(_args(), r, monkeypatch) \
        == "nothing to play for a long time -- check the login cookie"
    assert len(r.calls) == 1


def test_stretches_that_end_immediately_stop_the_run(monkeypatch):
    """The SDK rejoins tables by itself, so a stretch ending in seconds is a
    bridge that will not start or a dead cookie -- not a dissolved table."""
    r = _runner([_row(seconds=2.0)] * 9)
    assert _run(_args(max_short=5), r, monkeypatch) \
        == "stretches keep ending immediately"
    assert len(r.calls) == 5


def test_a_clean_stretch_resets_the_short_counter(monkeypatch):
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


def test_an_interrupt_stops_after_the_current_stretch(monkeypatch):
    r = _runner([_row()])
    monkeypatch.setattr(P, "_interrupted", True)
    assert _run(_args(max_sessions=9), r, monkeypatch) == "interrupted"
    assert len(r.calls) == 1


# ----------------------------------------------------------------- _watch()

class FakeBot:
    def __init__(self, frames=True, lost_at=()):
        self.frames_seen = 0
        self.seat_bot_controlled = False
        self._frames = frames
        self._lost_at = set(lost_at)
        self.ticks = 0

    def tick(self):
        self.ticks += 1
        if self._frames:
            self.frames_seen += 1
        if self._lost_at:
            self.seat_bot_controlled = self.ticks in self._lost_at


class FakeTask:
    def __init__(self, done_after=10_000):
        self.n = 0
        self.done_after = done_after

    def done(self):
        self.n += 1
        return self.n >= self.done_after


def _watch(bot, args, monkeypatch, task=None, boundary_at=None, cap=4000):
    """Drive `_watch` on a fake clock: one POLL_S per iteration, no real time."""
    agent = types.SimpleNamespace(at_hand_boundary=False)
    task = task or FakeTask()
    clock = {"t": 1000.0}
    monkeypatch.setattr(P.time, "monotonic", lambda: clock["t"])

    async def sleep(d):
        clock["t"] += d or 0
        bot.tick()
        if boundary_at is not None and bot.ticks >= boundary_at:
            agent.at_hand_boundary = True
        if bot.ticks > cap:
            raise AssertionError("_watch never returned")

    monkeypatch.setattr(asyncio, "sleep", sleep)
    deadline = clock["t"] + args.hours * 3600
    return asyncio.run(P._watch(bot, agent, args, deadline, task))


def test_a_takeover_is_seen_while_the_run_is_still_going(monkeypatch):
    """`bot.run()` no longer returns between tables, and the SDK clears this flag
    when it joins a fresh one -- so a takeover counted only at the end is a
    takeover never counted at all."""
    bot = FakeBot(lost_at=range(5, 4000))
    reason, losses, idle = _watch(bot, _args(max_seat_losses=1), monkeypatch,
                                  boundary_at=10)
    assert losses == 1
    assert "seat was lost" in reason
    assert not idle


def test_each_takeover_is_counted_once_not_once_per_poll(monkeypatch):
    """The flag is level, not an edge. Counting polls instead of transitions
    would hit any limit within a second of the first takeover."""
    bot = FakeBot(lost_at={5, 6, 7, 20, 21})
    reason, losses, _ = _watch(bot, _args(max_seat_losses=2), monkeypatch,
                               boundary_at=25)
    assert losses == 2


def test_stopping_waits_for_the_end_of_the_hand(monkeypatch):
    """Walking out mid-trick forfeits the seat and leaves three humans waiting."""
    bot = FakeBot(lost_at=range(5, 4000))
    reason, _, _ = _watch(bot, _args(max_seat_losses=1), monkeypatch,
                          boundary_at=100)    # within the grace period
    assert bot.ticks == 100                   # not 5
    assert "no hand boundary" not in reason


def test_a_hand_that_never_ends_does_not_trap_the_run(monkeypatch):
    """...but only up to a point: past the grace period we stop anyway."""
    bot = FakeBot(lost_at=range(5, 4000))
    reason, _, _ = _watch(bot, _args(max_seat_losses=1), monkeypatch,
                          boundary_at=None)
    assert "no hand boundary" in reason
    assert bot.ticks <= 5 + P.HAND_GRACE_S / P.POLL_S + 2


def test_silence_stops_the_run_and_is_flagged_as_idle(monkeypatch):
    """No frames at all: an empty lobby the SDK is retrying every 5 minutes, or
    an expired cookie. Indistinguishable from here, so say both."""
    bot = FakeBot(frames=False)
    reason, losses, idle = _watch(bot, _args(max_idle_min=0.5), monkeypatch,
                                  boundary_at=1)
    assert idle and losses == 0
    assert "no frames" in reason and "cookie" in reason


def test_arriving_frames_keep_the_run_alive(monkeypatch):
    """The control for the test above: the same short idle limit must NOT fire
    while frames are arriving. The stretch deadline ends it instead."""
    bot = FakeBot(frames=True)
    reason, _, idle = _watch(bot, _args(max_idle_min=0.5, hours=100 / 3600),
                             monkeypatch, boundary_at=1)
    assert reason == "stretch over"
    assert not idle


def test_a_run_that_ends_by_itself_returns_at_once(monkeypatch):
    """4001/4004, or an exception. Nothing to defer to a hand boundary."""
    bot = FakeBot()
    reason, losses, idle = _watch(bot, _args(), monkeypatch,
                                  task=FakeTask(done_after=3))
    assert reason == "the session ended on its own"
    assert (losses, idle) == (0, False)


def test_an_interrupt_is_honoured_at_the_next_hand_boundary(monkeypatch):
    monkeypatch.setattr(P, "_interrupted", True)
    bot = FakeBot()
    reason, _, _ = _watch(bot, _args(), monkeypatch, boundary_at=30)
    assert reason == "interrupted"
    assert bot.ticks == 30


# ------------------------------------------------------------ one_session()
#
# A stretch is hours long. If its bookkeeping only survived the happy path, a run
# stopped by a watchdog or killed by an exception would leave no trace of what it
# did -- so the row is written from a `finally`, and that is worth checking rather
# than asserting.

class StubBot:
    """Stands in for LiveBelotBot: runs until cancelled, or raises."""
    raises = None

    def __init__(self, cfg, agent=None):
        self.cfg, self.agent = cfg, agent
        self.frames_seen = 0
        self.seat_bot_controlled = False
        self.sync_engine = types.SimpleNamespace(hand_id=7)
        self.client = types.SimpleNamespace(sessions_played=2)
        self.cancelled = False

    async def run(self):
        if type(self).raises:
            raise type(self).raises
        try:
            while True:
                self.frames_seen += 1
                self.agent.at_hand_boundary = True
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class StubAgent:
    def __init__(self):
        self.stats = {"network": 3, "searched": 1, "fallback": 0,
                      "solve_errors": 0}
        self.max_decision_s = 0.4
        self.agent_resets = 0
        self.at_hand_boundary = False

    def begin_session(self):
        self.at_hand_boundary = False


def _one_session(tmp_path, monkeypatch, raises=None, **argkw):
    monkeypatch.setattr(P, "SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(P, "POLL_S", 0.001)
    monkeypatch.setattr(P, "SupervisedBot", StubBot)
    monkeypatch.setattr(StubBot, "raises", raises)
    cfg = {}
    monkeypatch.setattr(P.Config, "from_env",
                        staticmethod(lambda **kw: cfg.update(kw)
                                     or types.SimpleNamespace(**kw)))
    args = _args(hours=0.0, **argkw)          # deadline already passed
    row = asyncio.run(P.one_session(StubAgent(), args))
    logged = [json.loads(x) for x in
              (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    return row, logged, cfg


def test_a_stretch_that_is_stopped_still_writes_its_row(tmp_path, monkeypatch):
    row, logged, _ = _one_session(tmp_path, monkeypatch)
    assert row["stop_reason"] == "stretch over"
    assert logged == [row]
    assert row["hands_dealt"] == 7 and row["tables"] == 2
    assert row["error"] is None
    assert row["started"] <= row["ended"]
    assert row["bot_hands"] == 0 and row["bot_partner_hands"] == 0


def test_hands_with_a_bot_seat_are_counted_once_by_relation():
    """Seat 1 is an opponent and seat 2 our partner (we sit at 0). A hand counts
    once however many frames show the bot, and deal/lobby frames do not count."""
    state = types.SimpleNamespace(bot_seats=[False, True, False, False])
    sync = types.SimpleNamespace(my_pos=0, hand_id=1, state=state)
    fake = types.SimpleNamespace(sync_engine=sync,
                                 bot_hands={"partner": set(), "opponent": set()})
    note = P.SupervisedBot._note_bot_seats

    note(fake, {"currentPhase": 10})
    sync.hand_id, state.bot_seats = 2, [False, True, True, False]
    note(fake, {"currentPhase": 10})
    note(fake, {"currentPhase": 11})
    sync.hand_id = 3
    note(fake, {"currentPhase": 2})                  # the deal: not counted
    note(fake, {"currentPhase": 13})                 # hand over: not counted
    sync.hand_id, state.bot_seats = 4, [True, True, True, True]
    note(fake, {"currentPhase": 10})                 # our own seat is never counted
    assert fake.bot_hands == {"opponent": {1, 2, 4}, "partner": {2, 4}}


def test_a_stretch_that_raises_still_writes_its_row(tmp_path, monkeypatch):
    """A crash inside the SDK must not also cost us the record of the run.

    `await task` in the cleanup would re-raise it straight past the row write,
    which is exactly what this caught.
    """
    row, logged, _ = _one_session(tmp_path, monkeypatch,
                                  raises=RuntimeError("boom"))
    assert row["error"] == "RuntimeError: boom"
    assert row["stop_reason"] == "the session ended on its own"
    assert logged == [row]


def test_once_is_passed_through_to_the_sdk(tmp_path, monkeypatch):
    """`--once` gating only this loop would leave the SDK playing on forever
    underneath it, since it now finds a new table by itself."""
    assert _one_session(tmp_path, monkeypatch, once=True)[2]["reconnect"] is False
    assert _one_session(tmp_path, monkeypatch, once=False)[2]["reconnect"] is True
