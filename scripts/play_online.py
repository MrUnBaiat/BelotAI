"""
Play belot.md continuously, and record everything.

    python scripts/play_online.py --hours 4 --break-min 20

Two purposes. It is how the player is measured against the one opponent population
every offline number missed -- actual humans -- and it is how the human game data for
the deferred belief/opponent-modelling routes gets collected. The SDK records raw
server frames by default, so the second comes free with the first.

WHO OWNS WHAT. `belotmd` owns reconnection, and does it properly: a table dissolving
is the NORMAL end of every match (close code 4003) and it goes and finds another; an
empty lobby waits `retry_delay_s` (300 s) and looks again; 4005 rejoins immediately
because seat indices moved; 4001/4004 stop. So `bot.run()` no longer returns between
tables -- it returns only when the account genuinely cannot keep playing.

This script therefore does NOT reimplement any of that. It is a watchdog and a
bookkeeper for a run that the SDK keeps alive, and it owns the three things the SDK
deliberately does not:

  1. THE SEAT. If a turn times out, belot.md hands the seat to its own bot for the
     rest of that table. Nothing raises. Every message we send afterwards is ignored
     while the log goes on printing the cards we chose, and the frames from that
     point are the platform bot's play recorded under our name -- which poisons the
     dataset as well as wasting the run. The SDK reports the flag and by design does
     not act on it. So the watchdog counts takeovers AS THEY HAPPEN and stops the run
     after a few.

  2. PACING. Bounded stretches with breaks rather than a hard 24/7 loop: it caps the
     damage if something is wrong, and continuous automated play may not be within
     the platform's terms. That is a judgement for the account's owner, not a
     technical default.

  3. KNOWING WHEN TO GIVE UP. The SDK retries an empty lobby forever, which is right
     for the SDK and wrong for an unattended overnight run: an EXPIRED COOKIE is
     indistinguishable from an empty lobby from outside, and produces exactly the
     same 5-minute retry, silently, all night. So a run with no frames at all for
     `--max-idle-min` stops and says so.

Stopping is always deferred to a HAND BOUNDARY where possible. Walking out mid-trick
costs the seat and leaves three humans waiting.
"""

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belotmd import Config
from belotmd.bot import LiveBelotBot

from belot.online.agent import CompositeAgent
from belot.search.composite import DEFAULT_D

DEFAULT_CKPT = os.path.join("checkpoints", "v8_exp", "expd_latest.pt")
SESSION_DIR = "sessions"

# A stretch that ends almost immediately is not a dissolved table -- the SDK would
# have rejoined that itself. It is a bridge that will not start, a dead cookie, or an
# agent that crashes on the first decision. Repeat it and this is a retry loop
# against a wall.
SHORT_SESSION_S = 60.0
BACKOFF_START_S = 30.0
BACKOFF_MAX_S = 900.0

# How often the watchdog looks. Fast enough to catch a takeover within a trick,
# slow enough to cost nothing.
POLL_S = 1.0

# How long to wait for a hand to finish before stopping mid-hand anyway. A hand is
# ~90 s; past this, whatever we were waiting for is not coming.
HAND_GRACE_S = 150.0

# Set by SIGINT. Polled by the watchdog so a Ctrl-C stops at the end of the hand
# rather than abandoning three humans mid-trick. Press twice to stop now.
_interrupted = False


class SupervisedAgent(CompositeAgent):
    """The agent, plus the two things the watchdog needs to see.

    `at_hand_boundary` is the graceful-stop signal. The SDK calls `reset()` at every
    hand boundary -- it detects them itself, robustly to dropped frames -- so that is
    the cheapest correct place to learn that a hand just ended.

    `agent_resets` is NOT a hand count and is not named as one. The SDK also calls
    `reset()` on every frame in the lobby and end phases, so over the first 13 live
    sessions this counted 326 against 119 hands actually dealt. `hand_id` on the
    synchronizer is the honest counter, and the session row carries that instead.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.agent_resets = 0
        self.at_hand_boundary = False

    def reset(self):
        self.agent_resets += 1
        self.at_hand_boundary = True
        super().reset()

    def begin_session(self):
        for k in self.stats:
            self.stats[k] = 0
        self.max_decision_s = 0.0
        self.agent_resets = 0
        self.at_hand_boundary = False


class SupervisedBot(LiveBelotBot):
    """`LiveBelotBot` that counts the frames it has been given.

    The watchdog needs a liveness signal, and frames are the only one that means
    what it needs: hands and decisions stop advancing for ordinary reasons (it is
    someone else's turn), but a run that is receiving no frames at all is either
    waiting on an empty lobby or shouting into a closed session.
    """

    def __init__(self, *a, partner=None, **kw):
        super().__init__(*a, **kw)
        self.frames_seen = 0
        # Hands in which belot.md's bot played another seat at some point during
        # bidding or play, by relation. The platform replaces a human whose turn
        # times out and often gives the seat back later; ~15% of recorded hands
        # had one, and those are not hands against humans.
        self.bot_hands = {"partner": set(), "opponent": set()}
        # Our other account's username, when we are playing as a pair. Hands it
        # partnered us for are the whole point of the exercise and have to be
        # countable separately from hands with a stranger as partner.
        self.partner = (partner or "").strip().casefold() or None
        self.partner_hands = set()

    async def on_state_update(self, raw_state, my_player_id):
        self.frames_seen += 1
        await super().on_state_update(raw_state, my_player_id)
        self._note_bot_seats(raw_state)
        self._note_partner(raw_state)

    def _note_partner(self, raw_state):
        """Record the hands our own second account sat opposite us for.

        Only the boolean outcome is ever written down: the seat's name is
        compared here and goes no further.
        """
        sync = self.sync_engine
        me = sync.my_pos
        if self.partner is None or me is None:
            return
        if not 6 <= (raw_state.get("currentPhase") or 0) <= 11:
            return
        players = raw_state.get("players") or []
        seat = (me + 2) % 4
        if seat < len(players):
            name = (players[seat].get("name") or "").strip().casefold()
            if name == self.partner:
                self.partner_hands.add(sync.hand_id)

    def _note_bot_seats(self, raw_state):
        sync = self.sync_engine
        me = sync.my_pos
        bots = getattr(sync.state, "bot_seats", None)
        # Bidding through the completed-trick display (phases 6-11); an end-phase
        # flag is someone timing out on the ready button, not during the hand.
        if me is None or not bots or not 6 <= (raw_state.get("currentPhase") or 0) <= 11:
            return
        for s, is_bot in enumerate(bots):
            if is_bot and s != me:
                rel = "partner" if s % 2 == me % 2 else "opponent"
                self.bot_hands[rel].add(sync.hand_id)


def _stamp():
    # Milliseconds, not seconds: two accounts started together used to land on
    # the same recording name, and the SDK opens it in append mode -- so both
    # players' hands, each one private to its own bot, went into one file.
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_session(row):
    os.makedirs(SESSION_DIR, exist_ok=True)
    with open(os.path.join(SESSION_DIR, "log.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _say(msg):
    print(f"[supervisor] {msg}", flush=True)


async def _watch(bot, agent, args, deadline, task):
    """Supervise a running bot. Returns (reason, seat_losses, idle).

    Every stop is decided here and then DEFERRED to the next hand boundary, so the
    only thing this returns early for is the run ending by itself.
    """
    losses = 0
    was_lost = False
    last_seen = -1
    last_progress = time.monotonic()
    idle_limit = args.max_idle_min * 60
    want = want_since = None
    idle = False

    while True:
        await asyncio.sleep(POLL_S)
        if task.done():
            return "the session ended on its own", losses, False

        if bot.frames_seen != last_seen:
            last_seen = bot.frames_seen
            last_progress = time.monotonic()

        # Count the takeover as it happens. Waiting for `run()` to return would
        # miss it entirely: the SDK moves on to a fresh table, whose frames clear
        # the flag again, so by the end of a long run it reads False no matter how
        # many seats were lost along the way.
        lost = bool(bot.seat_bot_controlled)
        if lost and not was_lost:
            losses += 1
            _say(f"SEAT LOST ({losses}/{args.max_seat_losses}) -- everything "
                 f"recorded at our seat from here is the platform bot's play")
        was_lost = lost

        now = time.monotonic()
        if want is None:
            if _interrupted:
                want = "interrupted"
            elif losses >= args.max_seat_losses:
                want = (f"the seat was lost {losses} time"
                        f"{'' if losses == 1 else 's'}")
            elif now - last_progress > idle_limit:
                idle = True
                want = (f"no frames for {args.max_idle_min:g} min -- either the "
                        f"lobby has been empty that whole time, or the login "
                        f"cookie has expired")
            elif now >= deadline:
                want = "stretch over"
            if want is not None:
                want_since = now
                agent.at_hand_boundary = False       # wait for the NEXT one
                _say(f"{want}; stopping at the end of this hand")
        elif agent.at_hand_boundary:
            return want, losses, idle
        elif now - want_since > HAND_GRACE_S:
            return f"{want} (no hand boundary within {HAND_GRACE_S:.0f}s)", losses, idle


async def one_session(agent, args):
    """Play one stretch. Returns a row describing what happened.

    A stretch is `--hours` of play, not one table: the SDK moves between tables by
    itself. It ends at the deadline, on repeated seat losses, on a long silence, on
    Ctrl-C, or because the account can no longer play at all.
    """
    os.makedirs(SESSION_DIR, exist_ok=True)
    tag = f"_{args.account}" if args.account else ""
    frames = os.path.join(SESSION_DIR, f"frames_{_stamp()}{tag}.jsonl")
    agent.begin_session()

    # `--once` is the SDK's own flag: it makes `run()` return when the table
    # dissolves instead of finding another. Without passing it through, --once
    # would gate this loop while the SDK played on forever underneath it.
    #
    # `env_file` is what makes a pair safe: each process takes its credentials
    # from its own file and from nowhere else, so neither the other account's
    # file nor a stray BELOT_COOKIES in the shell can decide who this is.
    cfg = Config.from_env(env_file=args.env,
                          frames_path=frames,
                          reconnect=not args.once,
                          table_mode=args.table,
                          table_id=args.table_id or None,
                          table_creator=args.table_creator or None,
                          partner=args.partner or None,
                          rotation_probe=args.rotate_probe or None)
    bot = SupervisedBot(cfg, agent=agent, partner=args.partner)

    started_at, started = _now(), time.monotonic()
    deadline = started + args.hours * 3600
    error = None
    reason = "?"
    losses = 0
    idle = False

    task = asyncio.create_task(bot.run())
    try:
        reason, losses, idle = await _watch(bot, agent, args, deadline, task)
    except asyncio.CancelledError:
        reason = "cancelled"
        raise
    finally:
        # The row is written on EVERY exit path, including a cancellation and a
        # Ctrl-C. A stretch is hours long now; losing its bookkeeping because the
        # process was stopped would lose the whole night's accounting.
        if not task.done():
            task.cancel()
        # `await task` would RE-RAISE whatever the SDK raised, straight out of
        # this `finally` and past the row write -- losing the record of the very
        # run that failed. gather() with return_exceptions collects it instead.
        await asyncio.gather(task, return_exceptions=True)
        if task.done() and not task.cancelled() and task.exception():
            exc = task.exception()
            error = f"{type(exc).__name__}: {exc}"
            _say(f"session raised: {error}")

        dur = time.monotonic() - started
        bot_hands = getattr(bot, "bot_hands", None) or {"partner": set(),
                                                        "opponent": set()}
        row = {
            "started": started_at,
            "ended": _now(),
            "seconds": round(dur, 1),
            "account": args.account or None,
            "role": args.table,
            "partner_hands": len(getattr(bot, "partner_hands", ())),
            "frames": frames,
            "frames_seen": bot.frames_seen,
            "hands_dealt": bot.sync_engine.hand_id,
            "tables": bot.client.sessions_played,
            "agent_resets": agent.agent_resets,
            "seat_losses": losses,
            "bot_hands": len(bot_hands["partner"] | bot_hands["opponent"]),
            "bot_partner_hands": len(bot_hands["partner"]),
            "bot_opponent_hands": len(bot_hands["opponent"]),
            "stopped_for_idle": idle,
            "stop_reason": reason,
            "max_decision_s": round(agent.max_decision_s, 3),
            "error": error,
            **{k: int(v) for k, v in agent.stats.items()},
        }
        _log_session(row)

        _say(f"stretch ended after {dur / 60:.1f} min ({reason}): "
             f"{row['tables']} table(s), {row['hands_dealt']} hands dealt, "
             f"{agent.stats['network']} network / {agent.stats['searched']} searched"
             f" / {agent.stats['fallback']} fallback, "
             f"worst decision {agent.max_decision_s:.2f}s, "
             f"{row['bot_hands']} hand(s) with belot.md's bot in another seat"
             + (f"   SEAT LOST x{losses}" if losses else ""))
        if agent.stats["solve_errors"]:
            _say(f"WARNING: {agent.stats['solve_errors']} solver faults this "
                 f"stretch -- they fell back to the network, but they should "
                 f"not happen")
    return row


async def supervise(agent, args, run_session=None):
    """The stop conditions, separated from the playing so they can be tested.

    `run_session` is injected by the tests; live it is `one_session`. Everything
    that protects the account lives in here and in `_watch`, and none of it is
    exercised by a live run until it is too late to find out it was wrong.
    """
    run_session = run_session or one_session
    seat_losses = shorts = sessions = 0
    backoff = BACKOFF_START_S
    stop = None

    while stop is None:
        _say(f"stretch starting, up to {args.hours:g} h")
        row = await run_session(agent, args)
        sessions += 1

        # Seat losses accumulate across consecutive stretches: two in one and two
        # in the next is the same problem as four in one.
        if row.get("seat_losses"):
            seat_losses += row["seat_losses"]
            _say(f"seat lost {seat_losses} time(s) in a row "
                 f"(limit {args.max_seat_losses})")
            if seat_losses >= args.max_seat_losses:
                stop = "seat lost repeatedly"
                break
        else:
            seat_losses = 0

        if row.get("stopped_for_idle"):
            stop = "nothing to play for a long time -- check the login cookie"
            break
        if _interrupted:
            stop = "interrupted"
            break

        if row["seconds"] < SHORT_SESSION_S:
            shorts += 1
            _say(f"stretch was only {row['seconds']:.0f}s "
                 f"({shorts}/{args.max_short} short in a row) -- "
                 f"backing off {backoff:.0f}s")
            if shorts >= args.max_short:
                stop = "stretches keep ending immediately"
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

        if row["seconds"] >= SHORT_SESSION_S and args.break_min:
            _say(f"pausing {args.break_min:g} min")
            await asyncio.sleep(args.break_min * 60)

    _say(f"STOPPED: {stop}   ({sessions} stretches this run)")
    return stop


def _install_sigint():
    """First Ctrl-C asks to stop at the end of the hand; a second one stops now."""
    # SIGBREAK as well as SIGINT. The pair launcher stops a child with
    # CTRL_BREAK_EVENT (the only interrupt Windows can send to ONE process
    # group), and Python's default action for SIGBREAK is to die on the spot:
    # the stretch's `finally` never ran, so the guest's whole session row --
    # hands, decisions, timings -- was lost every time the pair stopped.
    stoppable = [s for s in (signal.SIGINT, getattr(signal, "SIGBREAK", None))
                 if s is not None]

    def handler(signum, frame):
        global _interrupted
        if _interrupted:
            for sig in stoppable:
                signal.signal(sig, signal.SIG_DFL)
            raise KeyboardInterrupt
        _interrupted = True
        _say("interrupt received -- finishing the current hand. "
             "Press Ctrl-C again to stop immediately.")
    for sig in stoppable:
        with contextlib.suppress(ValueError, OSError):   # not the main thread
            signal.signal(sig, handler)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--worlds", type=int, default=DEFAULT_D,
                    help=f"determinizations per searched decision (default {DEFAULT_D}). "
                         f"Re-measured in the game belot.md scores: D=32 over D=8 is "
                         f"+0.303 +- 0.264, D=128 over D=32 is +0.031 +- 0.264 -- nothing "
                         f"measurable. Timing at D=128 on real positions: 0.20 s median "
                         f"but 23.2 s worst against the 25 s clock, the tail entirely at "
                         f"trick 3. --worlds 32 is ~4x cheaper for no measurable strength "
                         f"cost and leaves far more clock")
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
    ap.add_argument("--max-idle-min", type=float, default=120.0,
                    help="stop after this long with no frames at all. The SDK "
                         "retries an empty lobby every 5 min forever, and an "
                         "expired cookie looks exactly the same from out here "
                         "(default 120, about 24 retries)")
    ap.add_argument("--max-sessions", type=int, default=0,
                    help="0 = unlimited")
    ap.add_argument("--once", action="store_true",
                    help="play a single table, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the agent and print the plan, without connecting")
    ap.add_argument("--seed", type=int, default=None)

    # --- playing as a pair -------------------------------------------------
    ap.add_argument("--account", default=None,
                    help="a label for this account, used in the recording "
                         "filename and the session log. Required when two run "
                         "at once, so their recordings never share a name")
    ap.add_argument("--env", default=None,
                    help="credentials file for THIS account (e.g. .env.alpha). "
                         "Its BELOT_COOKIES is the only one consulted, so a "
                         "second account cannot inherit the first's login")
    ap.add_argument("--table", choices=("lobby", "create", "join"),
                    default="lobby",
                    help="lobby: the first open public table (default). "
                         "create: make a table and wait in it. join: sit at "
                         "the table --table-creator made")
    ap.add_argument("--table-creator", default=None,
                    help="with --table join: the username of our other "
                         "account, whose table we are looking for")
    ap.add_argument("--table-id", default=None,
                    help="with --table join: a specific table id instead "
                         "(a string, prefix included)")
    ap.add_argument("--partner", default=None,
                    help="our other account's username. Only used to count "
                         "the hands it partnered us for; nothing about the "
                         "other players is recorded")
    ap.add_argument("--rotate-probe", type=int, default=0, metavar="N",
                    help="diagnostic: rotate the seats N times before playing, "
                         "even when they are already correct, logging the "
                         "layout before and after each one. Host only. This is "
                         "how CHANGE_PLAYERS_POSITION gets verified; 0 = off")
    args = ap.parse_args()

    if args.table == "join" and not (args.table_creator or args.table_id):
        sys.exit("--table join needs --table-creator (or --table-id): "
                 "otherwise there is nothing to tell our partner's table from "
                 "a stranger's.")

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
         f"stop after {args.max_seat_losses} seat losses or "
         f"{args.max_idle_min:g} min with no frames")
    _say(f"frames -> {SESSION_DIR}/frames_<utc>.jsonl, one file per stretch")

    # Credentials resolve from --env if given, otherwise from the SDK's own
    # project root rather than the working directory, so this runs the same
    # from anywhere. Printed because "which account is this playing as" should
    # never be a mystery mid-session.
    from belotmd.config import ENV_FILE
    cfg = Config.from_env(env_file=args.env)
    _say(f"credentials: {args.env or ENV_FILE} "
         f"({'loaded' if cfg.cookies else 'MISSING -- set BELOT_COOKIES'})")
    if args.account:
        _say(f"account: {args.account}")
    _say("table: " + {
        "lobby": "the first open public table",
        "create": "creating one and waiting in it",
        "join": f"joining {args.table_creator or args.table_id}'s table",
    }[args.table])
    _say(f"reconnection is the SDK's: empty lobby -> retry in "
         f"{cfg.retry_delay_s:.0f}s, table dissolved -> new table in "
         f"{cfg.rejoin_delay_s:.0f}s")
    if not cfg.cookies and not args.dry_run:
        sys.exit("no BELOT_COOKIES; see the SDK's .env.example")

    if args.dry_run:
        _say("dry run: agent built, not connecting")
        return 0

    _install_sigint()
    try:
        asyncio.run(supervise(agent, args))
    except KeyboardInterrupt:
        _say("interrupted -- stopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
