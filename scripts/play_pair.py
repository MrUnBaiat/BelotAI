"""
play_pair.py -- run two of our accounts at one table, as partners.

A random human partner decides half of every result. Two of our own agents
sitting opposite each other remove that, and the two other seats stay open to
humans, so the rating still means something.

Each account runs as its OWN PROCESS:

    host   --table create   makes the table and waits in it
    guest  --table join     finds that table in the lobby, by the host's
                            username, and sits down

Nothing is shared between them. Separate credentials files, separate bridges,
separate recordings, separate console logs -- the operating system, not our
care, is what keeps one player's cards out of the other's process.

    python scripts/play_pair.py \
        --host alpha --host-env .env.alpha --host-name MyFirstAccount \
        --guest beta --guest-env .env.beta --guest-name MySecondAccount \
        -- --worlds 32 --hours 4

Everything after `--` is passed to both children unchanged.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

SESSION_DIR = "sessions"
PLAY_ONLINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "play_online.py")

# Credentials and table settings are per account and come from each child's own
# --env file. Anything inherited from this shell could silently override that,
# which would put both children on the same account.
STRIPPED = ("BELOT_COOKIES", "BELOT_FRAMES", "BELOT_TABLE_MODE",
            "BELOT_TABLE_ID", "BELOT_TABLE_CREATOR")

# The GUEST starts first and must already be watching the lobby before the
# host creates anything: a fresh table is taken by strangers within seconds,
# and a guest still loading its checkpoint loses that race every time.
GUEST_READY_LINE = "Releasing join request"
GUEST_WAIT_S = 120.0

# A child finishes the match in progress when interrupted; past this it is not
# going to.
STOP_GRACE_S = 35 * 60.0      # longer than a child's own match grace (30 min)
POLL_S = 1.0


def _say(msg):
    print(f"[pair] {msg}", flush=True)


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def child_spec(role, account, env_file, partner_name, seed, stamp,
               extra=(), session_dir=SESSION_DIR, environ=None):
    """Everything about one child, as data: (argv, env, log_path).

    Pure, so the properties that matter -- each child gets its own credentials
    file, its own log, its own seed, and none of this shell's account
    variables -- are testable without starting anything.
    """
    if role not in ("host", "guest"):
        raise ValueError(f"role must be host or guest, not {role!r}")

    # -u because the child's stdout is a FILE, not a terminal: Python then
    # block-buffers it, and everything the SDK prints -- the room it joined,
    # the seat map, every error -- sits in an 8 KB buffer for minutes. Only
    # the supervisor's own lines use flush=True, so the logs looked alive
    # while saying nothing, and the launcher's wait for the host to be seated
    # could never match the line it was waiting for.
    argv = [sys.executable, "-u", PLAY_ONLINE,
            "--account", account,
            "--env", env_file,
            "--table", "create" if role == "host" else "join",
            "--partner", partner_name,
            "--seed", str(seed)]
    if role == "guest":
        # The guest looks for the host's table by the host's username, so the
        # two processes never have to talk to each other.
        argv += ["--table-creator", partner_name]
    argv += list(extra)

    source = os.environ if environ is None else environ
    env = {k: v for k, v in source.items() if k not in STRIPPED}

    log = os.path.join(session_dir, f"console_{stamp}_{account}.log")
    return argv, env, log


def wait_for_line(path, needle, timeout_s, proc=None, poll_s=POLL_S,
                  now=time.monotonic, sleep=time.sleep):
    """Wait until `needle` shows up in a child's log. -> True if it did.

    Returns False if the child died first or the wait ran out; the caller
    decides what that means.
    """
    deadline = now() + timeout_s
    while now() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                if needle in fh.read():
                    return True
        except FileNotFoundError:
            pass
        sleep(poll_s)
    return False


def interrupt(proc):
    """Ask a child to stop at the end of its match.

    play_online traps the interrupt and finishes the match rather than
    abandoning three humans mid-trick, so this is never a kill.
    """
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        pass


def stop_all(procs, grace_s=STOP_GRACE_S, poll_s=POLL_S,
             now=time.monotonic, sleep=time.sleep):
    """Interrupt every child, then insist. -> the ones that had to be killed."""
    alive = [p for p in procs if p.poll() is None]
    for proc in alive:
        interrupt(proc)

    deadline = now() + grace_s
    while now() < deadline:
        if all(p.poll() is not None for p in alive):
            return []
        sleep(poll_s)

    killed = [p for p in alive if p.poll() is None]
    for proc in killed:
        proc.terminate()
    return killed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True,
                    help="label for the account that CREATES the table")
    ap.add_argument("--host-env", required=True,
                    help="credentials file for the host account")
    ap.add_argument("--host-name", required=True,
                    help="the host account's belot.md username -- this is what "
                         "the guest looks for in the lobby")
    ap.add_argument("--guest", required=True,
                    help="label for the account that JOINS")
    ap.add_argument("--guest-env", required=True,
                    help="credentials file for the guest account")
    ap.add_argument("--guest-name", required=True,
                    help="the guest account's belot.md username")
    ap.add_argument("--seed", type=int, default=1,
                    help="the host's seed; the guest gets the next one, so the "
                         "two searches sample independently")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="after --, arguments passed to both children")
    args = ap.parse_args()

    extra = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    stamp = _stamp()
    os.makedirs(SESSION_DIR, exist_ok=True)

    # Guest first. It spends ten-odd seconds loading a checkpoint before its
    # first look at the lobby, so starting the host at the same moment would
    # have the table created -- and taken by strangers -- before anyone of
    # ours was watching for it.
    plan = [
        ("guest", args.guest, args.guest_env, args.host_name, args.seed + 1),
        ("host", args.host, args.host_env, args.guest_name, args.seed),
    ]

    procs, logs = [], []
    try:
        for role, account, env_file, partner, seed in plan:
            argv, env, log_path = child_spec(role, account, env_file, partner,
                                             seed, stamp, extra)
            _say(f"{role}: {account} -> {log_path}")
            handle = open(log_path, "a", buffering=1, encoding="utf-8")
            logs.append(handle)
            # Its own process group, so an interrupt reaches the child rather
            # than only this launcher.
            flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                     if os.name == "nt" else 0)
            procs.append(subprocess.Popen(argv, env=env, stdout=handle,
                                          stderr=subprocess.STDOUT,
                                          creationflags=flags))

            if role == "guest":
                _say("waiting for the guest to start watching the lobby...")
                if wait_for_line(log_path, GUEST_READY_LINE, GUEST_WAIT_S,
                                 proc=procs[0]):
                    _say("guest is watching; creating the table now")
                else:
                    _say("guest is not watching yet -- creating the table "
                         "anyway; it will find it on a later look")

        _say("both running. Ctrl-C stops them at the end of the current match.")
        while True:
            done = [p for p in procs if p.poll() is not None]
            if done:
                # Never leave one agent alone with a random partner -- and if
                # the HOST goes, belot.md removes the table anyway.
                _say(f"a child exited (code {done[0].returncode}); stopping the "
                     f"other")
                break
            time.sleep(POLL_S)

    except KeyboardInterrupt:
        _say("interrupted -- both children will finish the current match")
    finally:
        killed = stop_all(procs)
        if killed:
            _say(f"{len(killed)} child did not stop in "
                 f"{STOP_GRACE_S:.0f}s and was terminated")
        for handle in logs:
            handle.close()
        _say("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
