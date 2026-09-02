"""
PIMC (Perfect Information Monte Carlo) player -- a yardstick ABOVE the model.

WHY THIS EXISTS. Every metric this project has is either saturated (vs random),
structurally capped (per-hand win rate), exploitable (vs a reference that sits in
the training pool -- audit v4 defect D-B), or about to be out-grown (vs the greedy
heuristic, which the model already beats by +2.4). You cannot detect progress
against a yardstick you have saturated. PIMC is a search player with NO learning,
so it is a fixed, non-saturating reference that never changes across runs.

AUDIT v3 claimed +3.06 pts/hand vs the greedy heuristic at D=32. That file was
never committed and is gone, so the number is UNVERIFIED and is re-measured here.

HOW IT WORKS. At each card-play decision:
  1. Sample D determinizations -- full assignments of the unseen cards to the three
     opponents, consistent with hand sizes, `known_cards` (forced) and
     `impossible_cards` (forbidden, i.e. the void/failed-overruff inferences the
     env already tracks).
  2. For each determinization and each legal action, play the action and then roll
     the hand out to completion with the greedy heuristic in all four seats.
  3. Score each playout by this team's game-point difference and pick the action
     with the best mean.

Bidding is delegated to the greedy heuristic, so this isolates CARD PLAY strength
and keeps the comparison against the heuristic a clean single-variable one.

TWO DELIBERATE PROPERTIES, both needed for paired evaluation to stay valid:
  * It never touches the GLOBAL numpy RNG. Determinizations come from a private
    Generator and the scratch env is built with object.__new__ to avoid the
    reset() call that would consume global RNG. So re-seeding numpy still
    reproduces identical deals across conditions (AUDIT_HANDOFF section 8.3).
  * It is deterministic given its seed.
"""
import numpy as np

from belot.env import BelotEnv
from belot.heuristic import _heuristic_action

_PLAY_FIELDS = ("hands", "trump", "declarer", "declaring_team", "defending_team",
                "declarer_has_played_trump", "current_trick", "current_player",
                "tricks_played", "tricks_won_by_team", "raw_points_by_team",
                "graveyard", "bolts_by_team")


def _scratch_env():
    """A BelotEnv that never ran reset(), so it consumes no global RNG."""
    e = object.__new__(BelotEnv)
    e.action_space_size = 38
    e.num_players = 4
    e.phase = "PLAYING"
    e.done = False
    e.impossible_cards = np.zeros((4, 32), dtype=bool)
    e.known_cards = np.zeros((4, 32), dtype=bool)
    e.last_trick = []
    e.trick_history = []
    e.accumulated_dense_rewards = [0.0, 0.0, 0.0, 0.0]
    e.face_up_card = None
    e.face_up_suit = None
    e.deck = []
    e.dealer = 0
    e.bidding_round = 1
    e.passes_in_round = 0
    return e


def _load(scratch, env, hands):
    scratch.hands = [list(h) for h in hands]
    scratch.trump = env.trump
    scratch.declarer = env.declarer
    scratch.declaring_team = env.declaring_team
    scratch.defending_team = env.defending_team
    scratch.declarer_has_played_trump = env.declarer_has_played_trump
    scratch.current_trick = list(env.current_trick)
    scratch.current_player = env.current_player
    scratch.tricks_played = env.tricks_played
    scratch.tricks_won_by_team = list(env.tricks_won_by_team)
    scratch.raw_points_by_team = list(env.raw_points_by_team)
    scratch.graveyard = list(env.graveyard)
    scratch.bolts_by_team = list(env.bolts_by_team)
    scratch.phase = "PLAYING"
    scratch.done = False
    scratch.impossible_cards[:] = False
    scratch.known_cards[:] = False

    # MEMORY FIX. `env.step` appends to `trick_history` on every completed trick,
    # and `_scratch_env()` builds that list ONCE and reuses it for every playout.
    # Without clearing here, each playout appends eight more tricks to a list that
    # is never emptied, so a long-running search grows without bound in the total
    # number of playouts -- measured at ~5.4 MB per hand of D=32 target generation,
    # i.e. 6.3 GB after a thousand hands.
    #
    # Nothing read during a playout touches either field (`_calculate_final_rewards`
    # uses raw_points_by_team, tricks_won_by_team and bolts_by_team), so this is a
    # pure memory fix and DECISIONS ARE UNCHANGED -- asserted by requiring
    # bit-identical action choices against the unpatched version.
    scratch.trick_history.clear()
    scratch.last_trick.clear()


def sample_determinization(env, me, rng, tries=24):
    """Assign the unseen cards to the three opponents respecting hand sizes,
    known_cards (forced) and impossible_cards (forbidden). Returns 4 hands, or
    None if no consistent assignment was found."""
    others = [(me + 1) % 4, (me + 2) % 4, (me + 3) % 4]
    seen = set(env.hands[me]) | set(env.graveyard) | {c for _, c in env.current_trick}
    unseen = [c for c in range(32) if c not in seen]

    need = {q: len(env.hands[q]) for q in others}
    forced = {q: [] for q in others}
    for q in others:
        for c in np.flatnonzero(env.known_cards[q]):
            c = int(c)
            if c in unseen:
                forced[q].append(c)
    pool = [c for c in unseen if not any(c in forced[q] for q in others)]
    for q in others:
        need[q] -= len(forced[q])
        if need[q] < 0:
            return None

    for _ in range(tries):
        cap = dict(need)
        out = {q: list(forced[q]) for q in others}
        # most-constrained card first, ties broken randomly
        order = sorted(pool, key=lambda c: (sum(not env.impossible_cards[q, c]
                                                for q in others), rng.random()))
        ok = True
        for c in order:
            cand = [q for q in others if cap[q] > 0 and not env.impossible_cards[q, c]]
            if not cand:
                ok = False
                break
            # weight by remaining capacity so hand sizes fill evenly
            w = np.array([cap[q] for q in cand], dtype=np.float64)
            q = cand[int(rng.choice(len(cand), p=w / w.sum()))]
            out[q].append(c)
            cap[q] -= 1
        if ok and all(v == 0 for v in cap.values()):
            hands = [None] * 4
            hands[me] = list(env.hands[me])
            for q in others:
                hands[q] = out[q]
            return hands

    # Relaxation: the void inferences can be jointly unsatisfiable in rare states
    # (they are sound individually). Falling back to a size-only assignment keeps
    # the search running rather than silently skipping the decision.
    rest = list(pool)
    rng.shuffle(rest)
    hands = [None] * 4
    hands[me] = list(env.hands[me])
    i = 0
    for q in others:
        take = need[q]
        hands[q] = list(forced[q]) + rest[i:i + take]
        i += take
    return hands


def _playout(scratch, env, hands, action, team):
    """Play `action`, then roll out with the greedy heuristic. Returns this
    team's game-point difference for the hand."""
    _load(scratch, env, hands)
    a = action
    while True:
        scratch._handle_playing_action(a)
        if scratch.done:
            break
        a = _heuristic_action(scratch)
    gp = scratch._calculate_final_rewards()
    return gp[team] - gp[1 - team]


def make_pimc(D=16, seed=0):
    """Returns action_fn(belot) -> int. One scratch env and one RNG per player."""
    rng = np.random.default_rng(seed)
    scratch = _scratch_env()

    def act(env):
        if env.phase == "BIDDING":
            return _heuristic_action(env)
        legal = np.flatnonzero(env.get_legal_actions())
        if len(legal) == 1:
            return int(legal[0])
        me = env.current_player
        team = me % 2
        totals = np.zeros(len(legal))
        n = 0
        for _ in range(D):
            hands = sample_determinization(env, me, rng)
            if hands is None:
                continue
            for i, a in enumerate(legal):
                totals[i] += _playout(scratch, env, hands, int(a), team)
            n += 1
        if n == 0:
            return _heuristic_action(env)
        return int(legal[int(np.argmax(totals))])

    return act
