"""
Play belot.md continuously, and record everything.

    python scripts/play_online.py --hours 4 --break-min 20

Two purposes. It is how the player is measured against the one opponent population
every offline number missed -- actual humans -- and it is how the human game data for
the deferred belief/opponent-modelling routes gets collected. The SDK records raw
server frames by default, so the second comes free with the first.

WHY A SUPERVISOR AT ALL. `belotmd` plays one table until it dissolves and then returns;
a dissolved table at the end of a match is normal, not an error. Unattended collection
therefore needs something to rejoin, to back off when the transport fails, and above all
to STOP when something is systematically wrong rather than hammering the platform with a
broken agent.

THE FAILURE MODE THIS IS BUILT AROUND. If a turn times out -- a refused move, a slow
decision -- belot.md hands the seat to its own bot for the rest of the session. Nothing
raises. Every message we send afterwards is ignored while the log goes on printing the
cards we chose, and the frames from that point are the platform bot's play, not ours.
So the supervisor watches `seat_bot_controlled`, ends the session when it flips, and
stops entirely after a few in a row.

PACING. Bounded stretches with breaks rather than a hard 24/7 loop: it caps the damage
if something is wrong, and continuous automated play may not be within the platform's
terms. That is a judgement for the account's owner, not a technical default.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belotmd import Config

from belot.online.agent import CompositeAgent

DEFAULT_CKPT = os.path.join("checkpoints", "v8_exp", "expd_latest.pt")
SESSION_DIR = "sessions"

# A session that ends almost immediately is not a dissolved table -- it is a refused
# join, a dead cookie, or an agent that crashes on the first decision. Repeat it and
# the supervisor is a retry loop against a wall.
SHORT_SESSION_S = 60.0
BACKOFF_START_S = 30.0
BACKOFF_MAX_S = 900.0


class SupervisedAgent(CompositeAgent):
    """The agent, plus a count of hand boundaries.

    The SDK calls `reset()` at every hand boundary, which it detects itself and which
    is robust to dropped frames -- so counting resets is the cheapest honest measure of
    how much play a session actually got. It counts BOUNDARIES, not completed hands;
    `tools/online_report.py` derives exact hand outcomes from the frames.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.hand_boundaries = 0

    def reset(self):
        self.hand_boundaries += 1
        super().reset()

    def begin_session(self):
        for k in self.stats:
            self.stats[k] = 0
        self.max_decision_s = 0.0
        self.hand_boundaries = 0


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _log_session(row):
    os.makedirs(SESSION_DIR, exist_ok=True)
    with open(os.path.join(SESSION_DIR, "log.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _say(msg):
    print(f"[supervisor] {msg}", flush=True)


async def one_session(agent, args):
    """Play one table until it dissolves. Returns a row describing what happened."""
    from belotmd.bot import LiveBelotBot

    os.makedirs(SESSION_DIR, exist_ok=True)
    frames = os.path.join(SESSION_DIR, f"frames_{_stamp()}.jsonl")
    agent.begin_session()

    cfg = Config.from_env(frames_path=frames)
    bot = LiveBelotBot(cfg, agent=agent)

    started = time.monotonic()
    error = None
    try:
        await bot.run()
    except Exception as exc:                                   # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        _say(f"session raised: {error}")
    dur = time.monotonic() - started

    row = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(dur, 1),
        "frames": frames,
        "hand_boundaries": agent.hand_boundaries,
        "seat_lost": bool(getattr(bot, "seat_bot_controlled", False)),
        "max_decision_s": round(agent.max_decision_s, 3),
        "error": error,
        **{k: int(v) for k, v in agent.stats.items()},
    }
    _log_session(row)

    _say(f"session ended after {dur / 60:.1f} min: "
         f"{agent.hand_boundaries} hand boundaries, "
         f"{agent.stats['network']} network / {agent.stats['searched']} searched / "
         f"{agent.stats['fallback']} fallback, "
         f"worst decision {agent.max_decision_s:.2f}s"
         + ("   SEAT LOST" if row["seat_lost"] else ""))
    if agent.stats["solve_errors"]:
        _say(f"WARNING: {agent.stats['solve_errors']} solver faults this session -- "
             f"they fell back to the network, but they should not happen")
    return row


async def supervise(agent, args, run_session=None):
    """The stop conditions, separated from the connecting so they can be tested.

    `run_session` is injected by the tests; live it is `one_session`. Everything that
    protects the account lives in here, and none of it is exercised by a live run
    until it is too late to find out it was wrong.
    """
    run_session = run_session or one_session
    seat_losses = shorts = 0
    backoff = BACKOFF_START_S
    sessions = 0
    stop = None

    while stop is None:
        stretch_end = time.monotonic() + args.hours * 3600
        _say(f"stretch starting, up to {args.hours:g} h")

        while time.monotonic() < stretch_end and stop is None:
            row = await run_session(agent, args)
            sessions += 1

            if row["seat_lost"]:
                seat_losses += 1
                _say(f"seat lost ({seat_losses}/{args.max_seat_losses} in a row)")
                if seat_losses >= args.max_seat_losses:
                    stop = "seat lost repeatedly"
                    break
            else:
                seat_losses = 0

            if row["seconds"] < SHORT_SESSION_S:
                shorts += 1
                _say(f"session was only {row['seconds']:.0f}s "
                     f"({shorts}/{args.max_short} short in a row) -- "
                     f"backing off {backoff:.0f}s")
                if shorts >= args.max_short:
                    stop = "sessions keep ending immediately"
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)
            else:
                shorts = 0
                backoff = BACKOFF_START_S

            if args.once:
                stop = "--once"
                break
            if args.max_sessions and sessions >= args.max_sessions:
                stop = f"reached --max-sessions {args.max_sessions}"
                break

        if stop is None:
            _say(f"stretch over, pausing {args.break_min:g} min")
            await asyncio.sleep(args.break_min * 60)

    _say(f"STOPPED: {stop}   ({sessions} sessions this run)")
    return stop


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--worlds", type=int, default=8,
                    help="determinizations per searched decision (default 8, the "
                         "configuration the offline +0.974 was measured with)")
    ap.add_argument("--no-search", action="store_true",
                    help="network only -- what the FIRST live run should use, so a "
                         "clean session proves the encoder and the platform loop "
                         "before the search joins the list of suspects")
    ap.add_argument("--safety-margin", type=float, default=2.0,
                    help="seconds of the 25 s turn budget left unused")
    ap.add_argument("--check-worlds", action="store_true",
                    help="validate every sampled world against the constraint set; "
                         "cheap, worth leaving on for the first sessions")
    ap.add_argument("--hours", type=float, default=4.0,
                    help="length of a play stretch before a break")
    ap.add_argument("--break-min", type=float, default=20.0)
    ap.add_argument("--max-seat-losses", type=int, default=3)
    ap.add_argument("--max-short", type=int, default=5)
    ap.add_argument("--max-sessions", type=int, default=0,
                    help="0 = unlimited")
    ap.add_argument("--once", action="store_true",
                    help="play a single table, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the agent and print the plan, without connecting")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt}\n"
                 f"Weights are not distributed with the repo -- see the README.")

    agent = SupervisedAgent(
        checkpoint=args.ckpt,
        worlds=args.worlds,
        search=not args.no_search,
        safety_margin=args.safety_margin,
        check_worlds=args.check_worlds,
        seed=args.seed,
    )

    mode = "NETWORK ONLY" if args.no_search else f"network + exact search, D={args.worlds}"
    _say(f"agent: {mode}, device {agent.device}")
    _say(f"pacing: {args.hours:g} h stretches, {args.break_min:g} min breaks, "
         f"stop after {args.max_seat_losses} seat losses")
    _say(f"frames -> {SESSION_DIR}/frames_<utc>.jsonl, one file per session")

    if args.dry_run:
        _say("dry run: agent built, not connecting")
        return 0

    try:
        asyncio.run(supervise(agent, args))
    except KeyboardInterrupt:
        _say("interrupted -- stopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
