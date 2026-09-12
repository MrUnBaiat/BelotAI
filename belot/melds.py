"""
Combinations (melds) and belot.md's hand scoring -- the game the platform actually pays.

The simulator this project trained and measured in has no melds: a hand is scored on
trick points alone, with a fixed bolt line (the declaring team on <= 80 of 162 scores
nothing and concedes 16) and fixed 16-point stakes. belot.md does not play that game.
Read off 897 recorded hands and reproduced on every one of them
(research/v10_search FINDINGS §13):

    bolted   iff  raw_dec + c_dec <  raw_def + c_def
                  the defenders take 16 + (c_dec + c_def) // 10, the declarer's
                  combinations included
    tie           raw_dec + c_dec == raw_def + c_def: each team rounds its own total;
                  the declarer is never bolted on a tie
    made          b_def = bile(raw_def + c_def),  b_dec = 16 + all // 10 - b_def
    capot         a team that took NO trick scores -10 and its combinations are void;
                  the other team scores 16 + its own // 10
    third bolt    a team's third bolt costs it 10 more (counter mod 3)

The line moves on two recorded hands in three, and the mean stake is 19.2, not 16.

WHAT IS DECLARED, as the platform offers it (belotmd/game/combinations.py):

    runs     natural rank order 7 8 9 10 J Q K A, one suit; 3 = 20, 4 = 50, 5+ = 100
    quads    9s = 150, Js = 200, 10/Q/K/A = 100; 7s and 8s = 0 (they carry effects)
    bela     Q+K of trump, 20; scores independently of every contest

THE CONTEST, as the project owner states it and the recordings confirm:

    the two TEAMS' best combinations compete -- there is no contest within a team
    rank by points, then four-of-a-kind over a run, then top card, then trump
    the winning team scores ALL of its combinations; the other team scores none
    equal points, equal top card, neither in trump -> BOTH teams are cancelled
        ENTIRELY, a teammate's weaker combination included
    bela always scores for whoever holds it, contest or not

Checked on 17 recorded hands that tied on (points, top card): in the 11 where neither was
trump both teams scored nothing -- including the 3 where a teammate also held a weaker
combination, which scored nothing too -- and in 5 of the 6 where one side was in trump
that side took it. No recorded hand distinguishes four-of-a-kind-over-run from
top-card-first, so that order is the owner's rule rather than a measurement.

TIMING: runs and quads may be declared any time up to the last card of trick 2. Once
trick 2 completes the server confirms the winners and removes the losers, so from trick 3
-- where the search starts -- what stands is what will score. Bela is announced when its
holder plays the first of the trump Q/K pair.

The simulator has every seat declare everything offered (the SDK's own policy; humans
leave 6.7% undeclared): its mean stake is 18.88 +- 0.05 against 19.13 +- 0.25 recorded.

With no combinations and no capot every function here reduces EXACTLY to the original
scoring, which is what lets `BelotEnv(melds=True)` and the deployed search nest the
meld-free versions bit for bit. This module imports nothing from the package so that
`env.py`, the solver and the SDK-facing agent can all use it.
"""

TOTAL_RAW = 162
RUN_POINTS = {3: 20, 4: 50, 5: 100}
BELA_POINTS = 20
RANK_SEVEN, RANK_EIGHT = 0, 1
QUEEN, KING = 5, 6

# The contest variant validated on the recordings (see the module docstring).
RULE, TIE = "unified", "trump"


def quad_points(rank):
    return {RANK_SEVEN: 0, RANK_EIGHT: 0, 2: 150, 4: 200}.get(rank, 100)


def hand_runs(hand):
    """Maximal runs of >= 3 consecutive ranks in one suit -> [(suit, top_rank, length)]."""
    held = set(hand)
    out = []
    for s in range(4):
        r = 0
        while r < 8:
            if s * 8 + r in held:
                start = r
                while r + 1 < 8 and s * 8 + r + 1 in held:
                    r += 1
                if r - start + 1 >= 3:
                    out.append((s, r, r - start + 1))
            r += 1
    return out


def hand_quads(hand):
    held = set(hand)
    return [r for r in range(8) if all(s * 8 + r in held for s in range(4))]


def has_bela(hand, trump):
    return trump * 8 + QUEEN in hand and trump * 8 + KING in hand


def run_combo(suit, top, length):
    """A combination as (kind, points, top_rank, suit); the contest key is (points, top)."""
    return ("run", RUN_POINTS[min(length, 5)], top, suit)


def quad_combo(rank):
    return ("quad", quad_points(rank), rank, None)


def detect(hand, trump, declare_four_eights=False):
    """What a player who announces everything offered declares from an 8-card hand.

    Returns (combos, bela). Four 8s is left out by default because the SDK declines it:
    it silences every combination except bela, the declarer's own included.
    """
    combos = [run_combo(s, top, n) for s, top, n in hand_runs(hand)]
    for r in hand_quads(hand):
        if r == RANK_EIGHT and not declare_four_eights:
            continue
        combos.append(quad_combo(r))
    return combos, has_bela(hand, trump)


def resolve(declared, bela_seats=(), rule=RULE, tie=TIE, leader=0, trump=None,
            score="all"):
    """Per-team combination points (c0, c1) under a contest rule.

    `declared[s]` is seat s's list of run/quad combos. The contest compares each TEAM's
    best by (points, four-of-a-kind over run, top rank); the winning team scores ALL its
    combinations in that category (`score="best"`: only its best one) and the other team
    scores none. On an exact tie the trump one wins, and if neither is trump both teams
    are cancelled entirely. Bela always scores. The defaults are the confirmed rule; the
    alternatives exist so `research/v10_search/x17_meld_contest.py` can re-run the
    validation that chose it.

      rule  "unified"   runs and quads in ONE contest
            "separate"  runs contest runs, quads contest quads
            "runs"      runs contest; quads always score for their holder
      tie   "none" / "both" / "first" (earliest in play order from `leader`) / "trump"

    A declared four 8s silences everything but bela.
    """
    c = [0, 0]
    silenced = any(k[0] == "quad" and k[2] == RANK_EIGHT for s in range(4)
                   for k in declared[s])
    if not silenced:
        if rule == "runs":
            for s in range(4):
                c[s % 2] += sum(k[1] for k in declared[s] if k[0] == "quad")
        cats = {"unified": (("run", "quad"),), "separate": (("run",), ("quad",)),
                "runs": (("run",),)}[rule]
        for kinds in cats:
            best = [None, None]
            for off in range(4):
                s = (leader + off) % 4
                for k in declared[s]:
                    if k[0] not in kinds:
                        continue
                    # points, then a four-of-a-kind over a run at equal points (owner's
                    # rule: a quad beats a five-run, both 100), then the top card
                    key = (k[1], 1 if k[0] == "quad" else 0, k[2])
                    t = s % 2
                    if best[t] is None or key > best[t][0]:
                        best[t] = (key, off, k[3])
            if best[0] is None and best[1] is None:
                continue
            if best[1] is None or (best[0] is not None and best[0][0] > best[1][0]):
                win = (0,)
            elif best[0] is None or best[1][0] > best[0][0]:
                win = (1,)
            elif tie == "both":
                win = (0, 1)
            elif tie == "first":
                win = (0,) if best[0][1] < best[1][1] else (1,)
            elif tie == "trump":
                win = tuple(t for t in (0, 1) if best[t][2] == trump)[:1]
            else:
                win = ()
            for t in win:
                if score == "best":
                    c[t] += best[t][0][0]
                    continue
                for s in (t, t + 2):
                    c[t] += sum(k[1] for k in declared[s] if k[0] in kinds)
    for s in bela_seats:
        c[s % 2] += BELA_POINTS
    return c[0], c[1]


def bile(x):
    q, r = divmod(int(x), 10)
    return q + (1 if r > 5 else 0)


def platform_points(raw, c, declaring_team, tricks_won=None, third_bolt=False):
    """belot.md's match points (b0, b1) for one hand: raw trick points per team, each
    team's combination points, the declaring team, optionally the trick counts (for the
    capot) and whether the declaring team's next bolt is its third.

    Every branch was read off the recordings (research/v10_search/x17_meld_contest.py,
    897/897 reproduced): capot 19/19, bolt 228/228, tie 7/7, made 643/643.

    `tricks_won=None` means "unknown" -- the double-dummy solver carries no trick counts,
    so the search scores worlds without the capot, exactly as the original conversion
    scored them without the zero-tricks rule.

    `third_bolt` applies to the bolt branch only: on the platform a capot is -10 without a
    bolt marker, and a DECLARER capot never occurred in 897 recorded hands (the original
    env counts it as a bolt), so that corner is left as the platform shows it.
    """
    dec, dfn = declaring_team, 1 - declaring_team
    total = 16 + (c[0] + c[1]) // 10
    b = [0, 0]
    if tricks_won is not None and 0 in tricks_won:
        z = 0 if tricks_won[0] == 0 else 1
        b[z], b[1 - z] = -10, 16 + c[1 - z] // 10
    elif raw[dec] + c[dec] < raw[dfn] + c[dfn]:
        b[dfn] = total
        if third_bolt:
            b[dec] = -10
    elif raw[dec] + c[dec] == raw[dfn] + c[dfn]:
        b[dec], b[dfn] = bile(raw[dec] + c[dec]), bile(raw[dfn] + c[dfn])
    else:
        b[dfn] = bile(raw[dfn] + c[dfn])
        b[dec] = total - b[dfn]
    return b[0], b[1]


def gp_diff(raw0, declaring_team, team, c, third_bolt=False):
    """`team`'s game-point margin for a hand in which team 0 took `raw0` trick points,
    scored the platform's way without the capot -- the solver's root conversion."""
    b = platform_points((raw0, TOTAL_RAW - raw0), c, declaring_team, None, third_bolt)
    return b[team] - b[1 - team]


def deal_melds(hands8, trump, leader, rule=RULE, tie=TIE):
    """Everything a simulator needs about one deal's combinations, from the four 8-card
    hands at the start of play, every seat declaring everything offered.

    Returns {"public": (c0, c1) without bela, "bela_seat": seat or None,
             "total": (c0, c1) with bela, "declared": per-seat combos}.
    """
    declared, bela_seat = [], None
    for s in range(4):
        combos, bela = detect(hands8[s], trump)
        declared.append(combos)
        if bela:
            bela_seat = s
    pub = resolve(declared, (), rule=rule, tie=tie, leader=leader, trump=trump)
    tot = list(pub)
    if bela_seat is not None:
        tot[bela_seat % 2] += BELA_POINTS
    return {"public": pub, "bela_seat": bela_seat, "total": (tot[0], tot[1]),
            "declared": declared}


def world_c(melds, env, world_hands):
    """Combination totals a search may score ONE sampled world with, using only what the
    acting seat could know: the public (non-bela) contest result, plus bela as the
    public record has it -- or, while both trump Q and K are still in hand, as THIS
    world has it. Bela is announced when the first of the pair is played."""
    c = list(melds["public"])
    t = env.trump
    q, k = t * 8 + QUEEN, t * 8 + KING
    gone = set(env.graveyard) | {cd for _, cd in env.current_trick}
    if q in gone or k in gone:
        if melds["bela_seat"] is not None:
            c[melds["bela_seat"] % 2] += BELA_POINTS
    else:
        for s in range(4):
            if q in world_hands[s] and k in world_hands[s]:
                c[s % 2] += BELA_POINTS
                break
    return c[0], c[1]
