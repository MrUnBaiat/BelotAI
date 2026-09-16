"""
test_play_pair.py -- launching two accounts without letting them mix.

The launcher's whole job is separation: each child gets its own credentials
file, its own console log, its own seed, and none of this shell's account
variables. Getting that wrong does not crash -- both children simply play as
the same account, or one silently inherits the other's cookies -- so it is
pinned here rather than discovered live.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.play_pair as PP
from scripts.play_pair import child_spec, stop_all, wait_for_line

HOST_NAME, GUEST_NAME = "MyFirstAccount", "MySecondAccount"
STAMP = "20260916_120000"


def _host(**kw):
    kw.setdefault("environ", {})
    return child_spec("host", "alpha", ".env.alpha", GUEST_NAME, 1, STAMP, **kw)


def _guest(**kw):
    kw.setdefault("environ", {})
    return child_spec("guest", "beta", ".env.beta", HOST_NAME, 2, STAMP, **kw)


def _opt(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


# ------------------------------------------------------------ what each runs
def test_the_host_creates_a_table_and_the_guest_joins_it():
    host_argv, _, _ = _host()
    guest_argv, _, _ = _guest()

    assert _opt(host_argv, "--table") == "create"
    assert "--table-creator" not in host_argv, "the host has nobody to look for"

    assert _opt(guest_argv, "--table") == "join"
    assert _opt(guest_argv, "--table-creator") == HOST_NAME, (
        "without this the guest would sit at a stranger's table")


def test_each_child_is_told_who_its_partner_is():
    """Each one's partner is the OTHER account, which is what makes the paired
    hands countable afterwards."""
    assert _opt(_host()[0], "--partner") == GUEST_NAME
    assert _opt(_guest()[0], "--partner") == HOST_NAME


def test_each_child_reads_its_own_credentials_file():
    assert _opt(_host()[0], "--env") == ".env.alpha"
    assert _opt(_guest()[0], "--env") == ".env.beta"
    assert _opt(_host()[0], "--account") == "alpha"
    assert _opt(_guest()[0], "--account") == "beta"


def test_no_account_variable_from_this_shell_reaches_a_child():
    """THE BUG this prevents: one exported BELOT_COOKIES and both children play
    as the same account, against each other, with one rating to show for it."""
    shell = {"PATH": "/usr/bin", "BELOT_COOKIES": "PHPSESSID=leftover;",
             "BELOT_TABLE_MODE": "lobby", "BELOT_TABLE_ID": "new-1",
             "BELOT_TABLE_CREATOR": "Someone", "BELOT_FRAMES": "frames.jsonl"}

    _, env, _ = _host(environ=shell)

    assert "BELOT_COOKIES" not in env
    for name in PP.STRIPPED:
        assert name not in env, f"{name} would override the child's own file"
    assert env["PATH"] == "/usr/bin", "unrelated variables must survive"


def test_the_two_children_never_share_a_log():
    """Both print their own hand and their own declaration offers. Interleaved
    in one console, that is one player reading the other's cards."""
    _, _, host_log = _host()
    _, _, guest_log = _guest()

    assert host_log != guest_log
    assert host_log.endswith(f"console_{STAMP}_alpha.log")
    assert guest_log.endswith(f"console_{STAMP}_beta.log")


def test_the_two_searches_sample_independently():
    assert _opt(_host()[0], "--seed") != _opt(_guest()[0], "--seed")


def test_extra_arguments_reach_both_children():
    extra = ["--worlds", "32", "--hours", "4"]
    for argv, _, _ in (_host(extra=extra), _guest(extra=extra)):
        assert argv[-len(extra):] == extra


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError):
        child_spec("spectator", "x", ".env", "y", 1, STAMP)


# ------------------------------------------------------------- the waiting
class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, d):
        self.t += d


class FakeProc:
    """Exits after `exit_after` polls; never, if that is None."""

    def __init__(self, exit_after=None, returncode=0):
        self.exit_after = exit_after
        self.returncode = returncode
        self.polls = 0
        self.signals = []
        self.terminated = False

    def poll(self):
        self.polls += 1
        if self.exit_after is not None and self.polls > self.exit_after:
            return self.returncode
        return None

    def send_signal(self, sig):
        self.signals.append(sig)

    def terminate(self):
        self.terminated = True


def test_the_guest_waits_for_the_host_to_be_seated(tmp_path):
    """Looking for the table before the host has made it just burns retries."""
    log = tmp_path / "host.log"
    log.write_text("[SDK] Launching bridge...\n", encoding="utf-8")
    clock = Clock()

    def sleep(d):
        clock.sleep(d)
        log.write_text("[SDK] Joined Game Room: abc\n", encoding="utf-8")

    assert wait_for_line(str(log), PP.HOST_READY_LINE, 10.0,
                         now=clock.now, sleep=sleep) is True


def test_waiting_gives_up_rather_than_hanging(tmp_path):
    log = tmp_path / "host.log"
    log.write_text("nothing useful\n", encoding="utf-8")
    clock = Clock()
    assert wait_for_line(str(log), PP.HOST_READY_LINE, 5.0,
                         now=clock.now, sleep=clock.sleep) is False


def test_waiting_stops_early_if_the_host_died(tmp_path):
    clock = Clock()
    dead = FakeProc(exit_after=0)
    assert wait_for_line(str(tmp_path / "missing.log"), PP.HOST_READY_LINE,
                         60.0, proc=dead, now=clock.now,
                         sleep=clock.sleep) is False
    assert clock.now() == 0.0, "should not have waited at all"


# ------------------------------------------------------------- the stopping
def test_stopping_asks_both_children_to_finish_the_hand():
    """Not a kill: three humans are at that table, mid-trick."""
    clock = Clock()
    a, b = FakeProc(exit_after=1), FakeProc(exit_after=1)

    killed = stop_all([a, b], grace_s=30.0, now=clock.now, sleep=clock.sleep)

    assert killed == []
    assert a.signals and b.signals, "both must be asked to stop"
    assert not a.terminated and not b.terminated


def test_a_child_that_will_not_stop_is_terminated():
    clock = Clock()
    stubborn = FakeProc(exit_after=None)

    killed = stop_all([stubborn], grace_s=10.0, now=clock.now,
                      sleep=clock.sleep)

    assert killed == [stubborn]
    assert stubborn.terminated


def test_a_child_that_already_exited_is_left_alone():
    already = FakeProc(exit_after=0)
    assert stop_all([already], grace_s=1.0, now=Clock().now,
                    sleep=Clock().sleep) == []
    assert already.signals == []


def test_interrupting_a_dead_child_is_not_an_error():
    class Gone(FakeProc):
        def send_signal(self, sig):
            raise OSError("no such process")

    PP.interrupt(Gone())        # must not raise
