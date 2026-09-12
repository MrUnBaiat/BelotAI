"""
The composite player -- the strongest Belot agent this project produced.

    model bidding + model card play at tricks 0-2 + exact-solve PIMC (D=128) from trick 3

Measured on the swap-paired instrument, identical-policy control exactly zero:

    +1.270 +- 0.197 pts/hand over the bare agent          (n=1000, D=128)
    +0.835 +- 0.219 pts/hand over the bare agent          (n=1000, D=8)
    +0.936 +- 0.243 pts/hand vs a held-out opponent       (n=500,  D=8)

WHY THIS CONFIGURATION AND NOT ANOTHER. Every parameter below is a measurement, not
a preference:

  min_trick = 3   Extending the search into tricks 0-2 was measured at
                  -0.348 +- 0.309 pts/hand -- significantly WORSE. At trick 0 there
                  are 24 unseen cards, so a single determinization's verdict is
                  nearly uninformative; searching there adds variance, not skill.
                  Searching tricks 0-2 on their own saturates near +0.2 even at
                  D=128, so the window is not worth buying at any affordable price.

  D = 128         Worth +0.435 +- 0.208 pts/hand over D=8, swap-paired on identical
                  deals: +0.378 +- 0.291 on deals 0-499 and +0.492 +- 0.297 on a
                  REPLICATION over deals 500-999 that no earlier measurement had
                  touched (the two agree, z = +0.54). Pooled n=1000: bootstrap
                  [+0.226, +0.645] over 20k resamples, and the arms diverged on 458
                  deals of which D=128 won 272 and lost 186, sign test p = 0.00007.

                  THIS OVERTURNS WHAT THIS DOCSTRING USED TO SAY. It claimed "D=16
                  was measured as equal strength at equal cost... the determinization
                  axis is flat in this window", and generalised a null measured at
                  ONE doubling into a property of the whole axis. The 8-vs-16 null
                  stands; the axis is flat near 8 and rises by 128. The audit's own
                  record already disagreed with the general claim -- 07_BUILD.md
                  reports the rollout-PIMC teacher going +0.003 (D=8) -> +0.317
                  (D=16) -> +0.483 (D=32) against the model -- so the evidence was
                  there and the summary had smoothed it away.

                  Cost is not the constraint it was assumed to be. Measured on real
                  captured positions, per exact solve: 0.031 s at trick 3, 0.005 s at
                  trick 4, 0.0007 s at trick 5, 0.0001 s at trick 6. So a whole D=128
                  decision runs 0.52 s median and 7.94 s worst against belot.md's 25 s
                  turn clock -- see `tools/verify_online.py --worlds 128`.

                  Where on the 8 -> 128 curve the gain begins is NOT measured. D=32
                  is 5x cheaper per decision and may well capture most of it.

  exact solve     An exact double-dummy solve of each determinization beats the
                  cheap greedy rollout by +0.360 +- 0.204 from trick 3 onward. That
                  advantage vanishes earlier in the hand (+0.015 +- 0.339 at trick
                  2), which is the other half of why the search starts at trick 3.

  model bidding   Bidding is left to the network in every arm. A search bidder is a
                  downgrade at any reachable number of determinizations, and a
                  perfect distributional bidder is worth at most +0.255 +- 0.063
                  over the model's -- the bidding "headroom" is dominated by
                  clairvoyance over card luck, not by skill.

THE BASE IS NOT A DETAIL. The same search on a greedy-heuristic base scores -1.188;
on the model base it scores +0.614 on the identical construction. The network's job
is to hand good positions to the search, and it is worth the entire effect.
"""

import numpy as np
import torch

from belot import melds
from belot.heuristic import _heuristic_action
from belot.model import RecurrentMAPPOModel
from belot.observation import build_observation
from belot.search.dd_solver import hands_to_masks, solve_root
from belot.search.pimc import _playout, _scratch_env, sample_determinization

HIDDEN = 512
TOTAL_RAW = 162          # 152 in cards + 10 for the last trick

# Determinizations per searched decision. One number, so the offline player, the
# live player and every tool move together -- a config that drifts between them is
# how a measured result stops describing what is actually deployed.
DEFAULT_D = 128


def gp_diff_from_raw(raw0, declaring_team, team):
    """`env._calculate_final_rewards`' main branch as a function of team 0's raw points.

    The solver returns raw card points; strength is measured in GAME points, and the
    mapping between them is non-linear (a declaring team on 80 or fewer scores zero
    and concedes 16). Searching on raw points would therefore optimise the wrong
    objective near the bolt threshold, so every determinization's value is converted
    here before being averaged.

    The zero-tricks (-10) rule is deliberately NOT modelled: it needs final trick
    counts, which the solver does not carry. From trick 3 a team that has already
    taken a trick cannot trigger it, and a team still on zero after three tricks
    taking none of the remaining five is rare. Stated rather than hidden.
    """
    raw = (raw0, TOTAL_RAW - raw0)
    dec, dfn = declaring_team, 1 - declaring_team
    gp = [0, 0]
    if raw[dec] <= 80:
        gp[dec], gp[dfn] = 0, 16
    else:
        r = raw[dfn]
        bile = (r // 10) + (1 if (r % 10) > 5 else 0)
        gp[dfn], gp[dec] = bile, 16 - bile
    return gp[team] - gp[1 - team]


def platform_points(raw, c, declaring_team, third_bolt=False):
    """belot.md's match points (b0, b1) for one hand, from each team's raw trick points
    AND its combination (meld) points -- the rule the platform actually pays.

    The simulator has no melds, so `gp_diff_from_raw` scores every world against a fixed
    line (raw <= 80) with fixed 16-point stakes. The platform does not: read off 897
    recorded hands (research/v10_search FINDINGS §13, 897/897 reproduced) it is

        bolted  iff  raw_dec + c_dec <  raw_def + c_def
                     the defenders take 16 + (c_dec + c_def) // 10, the declarer's
                     combinations included
        tie          raw_dec + c_dec == raw_def + c_def: each team rounds its own
                     total; the declarer is never bolted on a tie
        made         b_def = bile(raw_def + c_def),  b_dec = 16 + all // 10 - b_def

    The line moves on two recorded hands in three, and the mean stake is 19.2, not 16.

    With c == (0, 0) this IS `gp_diff_from_raw`'s branch: the two raw totals sum to 162,
    so raw_dec < raw_def is raw_dec <= 80, the only tie is 81/81 and scores 8/8 either
    way, and the stakes are 16. `tests/test_platform_scoring.py` pins that identity on
    every raw0. The capot (-10, melds void) is not modelled, for the reason the
    zero-tricks rule is not: the solver carries no trick counts.

    `third_bolt`: a team's THIRD bolt costs it a further 10, on the platform and in the
    env alike (`env._calculate_final_rewards`, counter mod 3). The declaring team goes
    into 9.5% of recorded hands on two bolts. MEASURED AND NOT WIRED INTO THE AGENT: with
    the third bolt live on every deal, a search that knows it scores +0.005 +- 0.134 over
    one that does not (research/v10_search x24, n=400) -- the search already aims at the
    bolt line, and the size of the penalty rarely changes which card clears it. The
    parameter serves the research harness.

    The rule itself lives in `belot.melds`, which `BelotEnv(melds=True)` scores whole
    hands with; this is the solver's view of it (no trick counts, so no capot).
    """
    return melds.platform_points(raw, c, declaring_team, None, third_bolt)


def gp_diff_platform(raw0, declaring_team, team, c, third_bolt=False):
    """`gp_diff_from_raw` under the platform's rule; identical to it when c == (0, 0)
    and the third bolt is not live."""
    return melds.gp_diff(raw0, declaring_team, team, c, third_bolt)


def solve_world(hands, seat, trick, trump, declarer, declarer_has_played_trump,
                raw_points_team0, declaring_team, legal, c=(0, 0), third_bolt=False):
    """Exactly solve ONE determinization and score every legal card, in game points.

    This is the innermost step of the search, factored out so that the offline player
    and the live one call the same code rather than two copies that drift. It takes
    plain values, not an environment, because online the position arrives as a
    `belotmd.game.state.BelotState` and offline as a `belot.env.BelotEnv` -- the two
    carry identical field names but are unrelated classes.

    `hands` is four card lists indexed by seat, already consistent with everything the
    acting seat knows. `legal` is the card actions to score. Returns one score per entry
    of `legal`, in the acting team's game points, so worlds sum comparably.

    A card the solver has no value for scores 0 for this world -- it cannot be chosen on
    that world's evidence, but neither is it penalised.

    `c` is each team's COMBINATION points to score this world with. Offline it is
    (0, 0) -- the simulator has no melds -- and the conversion is `gp_diff_from_raw`
    unchanged, by construction rather than by equivalence. Online it is what the
    platform will add to the trick points (`gp_diff_platform`). `third_bolt` says the
    declaring team's next bolt is its third (-10 more).
    """
    team = seat % 2
    masks = hands_to_masks(hands)
    _, vals, _ = solve_root(masks, seat, trick, trump, declarer,
                            declarer_has_played_trump)
    platform = bool(c[0] or c[1]) or third_bolt
    out = np.zeros(len(legal))
    for i, a in enumerate(legal):
        rem0 = vals.get(int(a))
        if rem0 is None:
            continue
        r0 = raw_points_team0 + rem0
        out[i] = (gp_diff_platform(r0, declaring_team, team, c, third_bolt) if platform
                  else gp_diff_from_raw(r0, declaring_team, team))
    return out


def solve_world_for(position, seat, hands, legal, c=(0, 0), third_bolt=False):
    """`solve_world` reading the position off an object with `env.py`'s field names.

    Works on a `BelotEnv` and on the SDK's `BelotState` alike, which is the whole point:
    the live adapter and the offline search share one solve.

    `declaring_team` IS DERIVED RATHER THAN READ, and that is not defensive padding.
    Offline it is set by `_finalize_bidding()` when the auction is played through. Live
    it is not: the SDK reconstructs a state from server frames and assigns `declarer`
    and `trump` directly, so `_finalize_bidding()` never runs. Measured on a real
    capture: `declaring_team` is None in 410 of 410 play states. Reading it would send
    None into `gp_diff_from_raw`, which evaluates `1 - declaring_team` -- a TypeError on
    every searched decision, no move dispatched, the turn times out, and belot.md takes
    the seat for the rest of the session. `declarer % 2` is the SDK's own definition
    (`belotmd/game/state.py`), so deriving it is exact, not a guess.
    """
    declaring_team = position.declaring_team
    if declaring_team is None:
        declaring_team = position.declarer % 2
    return solve_world(
        hands, seat,
        tuple((p, card) for p, card in position.current_trick),
        position.trump, position.declarer, position.declarer_has_played_trump,
        position.raw_points_by_team[0], declaring_team, legal, c=c,
        third_bolt=third_bolt)


def make_dd_pimc(D=DEFAULT_D, seed=0, min_trick=3):
    """Perfect-information Monte Carlo whose leaf evaluation is an EXACT solve.

    Sample `D` worlds consistent with everything the acting seat can see, solve each
    one exactly for every legal card, convert to game points, and play the card with
    the best total. Falls back to the heuristic when no consistent world can be
    sampled.

    Returns `(act, reseed)`. **Use `reseed` in any paired comparison**: the
    determinization draw is not cancelled by pairing, and letting one generator run
    continuously through both arms of an evaluation leaves uncancelled search noise
    inside the result -- measured once at 67% of a headline interval.
    """
    state = {"rng": np.random.default_rng(seed)}

    def act(env):
        if env.phase == "BIDDING":
            return _heuristic_action(env)
        legal = np.flatnonzero(env.get_legal_actions())
        if len(legal) == 1:
            return int(legal[0])
        if env.tricks_played < min_trick:
            return _heuristic_action(env)
        me = env.current_player
        collected0 = env.raw_points_by_team[0]
        # hoisted out of the loop: neither changes between worlds
        trick = tuple((p, c) for p, c in env.current_trick)
        tot = np.zeros(len(legal))
        n = 0
        for _ in range(D):
            hands = sample_determinization(env, me, state["rng"])
            if hands is None:
                continue
            tot += solve_world(hands, me, trick, env.trump, env.declarer,
                               env.declarer_has_played_trump, collected0,
                               env.declaring_team, legal)
            n += 1
        if n == 0:
            return _heuristic_action(env)
        return int(legal[int(np.argmax(tot))])

    def reseed(s):
        state["rng"] = np.random.default_rng(s)

    return act, reseed


def make_heur_pimc(D=8, seed=0, min_trick=3):
    """The same search with the CHEAP evaluator: one greedy rollout per legal card
    instead of an exact solve.

    This is `pimc.py`'s shipped evaluator, gated to the same tricks so the two are
    directly comparable. It exists as the honest baseline for the exact solver --
    the +0.360 +- 0.204 quoted in this module's docstring is the paired difference
    between the two, measured on identical deals.

    D STAYS AT 8 HERE, deliberately, while the deployed player moved to DEFAULT_D.
    That +0.360 was measured with BOTH sides at D=8; raising only this one would
    silently redefine the baseline and make the recorded number describe a
    comparison nobody ran. Pass D explicitly to sweep it.
    """
    state = {"rng": np.random.default_rng(seed)}
    scratch = _scratch_env()

    def act(env):
        if env.phase == "BIDDING":
            return _heuristic_action(env)
        legal = np.flatnonzero(env.get_legal_actions())
        if len(legal) == 1:
            return int(legal[0])
        if env.tricks_played < min_trick:
            return _heuristic_action(env)
        me = env.current_player
        team = me % 2
        tot = np.zeros(len(legal))
        n = 0
        for _ in range(D):
            hands = sample_determinization(env, me, state["rng"])
            if hands is None:
                continue
            for i, a in enumerate(legal):
                tot[i] += _playout(scratch, env, hands, int(a), team)
            n += 1
        if n == 0:
            return _heuristic_action(env)
        return int(legal[int(np.argmax(tot))])

    def reseed(s):
        state["rng"] = np.random.default_rng(s)

    return act, reseed


def model_backed(model, search_act, min_trick=3, device=None):
    """The composite: the MODEL plays everything except unforced card decisions from
    `min_trick`, which the search plays.

    The network is queried on EVERY decision of that seat, including the ones the
    search overrides, so its LSTM sees the same sequence it would see playing alone.
    Skipping those forwards would leave the hidden state out of step with the hand
    and quietly change the model's own decisions in tricks 0-2.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hc = {}

    def reset():
        for s in range(4):
            hc[s] = (torch.zeros(1, 1, HIDDEN, device=device),
                     torch.zeros(1, 1, HIDDEN, device=device))

    @torch.no_grad()
    def fn(env):
        s = env.current_player
        local, glob, mask = build_observation(env, s, [0, 0])
        dist, _, hc[s] = model(
            torch.from_numpy(local).unsqueeze(0).to(device),
            torch.from_numpy(glob).unsqueeze(0).to(device), hc[s],
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device),
            is_sequence=False)
        own = int(dist.probs.argmax(-1).item())
        if env.phase == "BIDDING":
            return own
        legal = np.flatnonzero(env.get_legal_actions())
        if len(legal) == 1:
            return int(legal[0])
        if env.tricks_played < min_trick:
            return own
        return int(search_act(env))

    reset()
    return fn, reset


def load_model(ckpt, device=None):
    """Load a trained Recurrent MAPPO checkpoint in eval mode."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    net.eval()
    return net


def build_player(ckpt, D=DEFAULT_D, min_trick=3, seed=0, device=None):
    """The deployable player, in one call.

    Returns `(act, reset, reseed)`:
        act(env)     -> the action to take
        reset()      -> call once per hand, to clear the LSTM state
        reseed(s)    -> call once per deal in any paired evaluation
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    search_act, reseed = make_dd_pimc(D=D, seed=seed, min_trick=min_trick)
    act, reset = model_backed(model, search_act, min_trick=min_trick, device=device)
    return act, reset, reseed
