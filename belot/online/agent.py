"""
The composite player, adapted to a live belot.md table.

Routing is identical to the offline player, and every gate is a measurement:

    bidding              -> network
    play, tricks 0-2     -> network      (searching earlier: -0.348 +- 0.309)
    play, tricks 3-7     -> exact-solve PIMC over sampled worlds
    beliefs degraded     -> network      (see `_should_search`)

Everything platform-shaped belongs to the SDK and is not here. This file only
decides.

WHAT IS REUSED, NOT COPIED. `belot.observation.build_observation` runs on the SDK's
`BelotState` unchanged -- verified over 400 real captured states -- because the SDK
deliberately kept `env.py`'s field names. So the live agent encodes with the same
function, and consumes the same network and the same solver, as the +0.974 was
measured with. There is no second copy of the encoder to drift.

The one genuinely online input is free: our encoder already reads `known_cards` and
`impossible_cards`, and that is exactly where the SDK writes the card pins it decodes
from declared combinations. Self-play never had those, so the worlds sampled here are
better constrained than the ones the offline number was measured on.
"""

import time

import numpy as np
import torch

from belotmd import Infeasible, constraints, sample_determinization

from belot.model import RecurrentMAPPOModel
from belot.observation import build_observation
from belot.search.composite import DEFAULT_D, solve_world_for

HIDDEN_DIM = 512

# Play switches to search at this trick. Measured offline: searching earlier is
# significantly WORSE (-0.348 +- 0.309), because with 24 unseen cards a single
# world's verdict is nearly uninformative and the search adds variance, not skill.
SEARCH_FROM_TRICK = 3

# Leave this much of the turn budget unused. belot.md gives 25 s to play a card, and
# overrunning does not forfeit the turn -- it forfeits the SEAT, for the rest of the
# session, after which every message we send is ignored while the hand keeps moving.
# Two seconds of margin costs nothing by comparison.
SAFETY_MARGIN_S = 2.0

# Fallback when the state carries no clock (offline, or a frame without one).
DEFAULT_BUDGET_S = 20.0


class CompositeAgent:
    """Network base with exact-solve PIMC from trick 3."""

    name = "composite"

    def __init__(self, checkpoint=None, worlds=DEFAULT_D, device=None,
                 search=True, safety_margin=SAFETY_MARGIN_S, seed=None,
                 check_worlds=False):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.worlds = int(worlds)
        self.search_enabled = str(search).lower() not in ("0", "false", "no")
        self.safety_margin = float(safety_margin)
        # Validate every sampled world against the constraint set. Cheap, and worth
        # leaving on for the first sessions -- it holds the sampler to the same
        # standard whether it is the SDK's or ours.
        self.check_worlds = str(check_worlds).lower() not in ("0", "false", "no")
        self.rng = np.random.default_rng(seed)

        self.model = RecurrentMAPPOModel(hidden_dim=HIDDEN_DIM).to(self.device)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[composite] weights from {checkpoint} "
                  f"(epoch {ckpt.get('epoch', '?')})")
        else:
            print("[composite][CRITICAL] no checkpoint -- RANDOM WEIGHTS")
        self.model.eval()

        self.hidden = self._zero_hidden()
        self.stats = {"network": 0, "searched": 0, "fallback": 0,
                      "infeasible": 0, "degraded": 0, "worlds": 0,
                      "solve_errors": 0}
        # Worst wall-clock spent on one decision. The number that says whether the
        # 25 s budget is comfortable or whether the seat is one slow turn from lost.
        self.max_decision_s = 0.0

    # ------------------------------------------------------------- lifecycle
    def _zero_hidden(self):
        return (torch.zeros(1, 1, HIDDEN_DIM, device=self.device),
                torch.zeros(1, 1, HIDDEN_DIM, device=self.device))

    def reset(self):
        """New hand: zero the recurrent state, as at the start of training."""
        self.hidden = self._zero_hidden()

    def snapshot(self):
        return self.hidden

    def restore(self, snapshot):
        self.hidden = snapshot

    # ------------------------------------------------------------------ act
    def act(self, state, seat, match_scores, legal_mask):
        started = time.monotonic()
        try:
            # The LSTM must advance exactly once per DECISION, including on turns the
            # search decides -- otherwise it sees a different number of steps live
            # than it did in training. So the network runs first, unconditionally,
            # and only then do we consider overriding its choice.
            net_action = self._network_action(state, seat, match_scores, legal_mask)

            if not self._should_search(state, seat, legal_mask):
                self.stats["network"] += 1
                return net_action

            searched = self._search_action(state, seat, legal_mask)
            if searched is None:
                self.stats["fallback"] += 1
                return net_action

            self.stats["searched"] += 1
            return searched
        finally:
            self.max_decision_s = max(self.max_decision_s,
                                      time.monotonic() - started)

    def _should_search(self, state, seat, legal_mask):
        if not self.search_enabled:
            return False
        if state.phase != "PLAYING":
            return False                       # bidding is the network's
        if state.tricks_played < SEARCH_FROM_TRICK:
            return False                       # measured worse before trick 3
        if int(np.asarray(legal_mask).sum()) < 2:
            return False                       # nothing to choose between
        if getattr(state, "beliefs_degraded", False):
            # Joined mid-hand: the constraints are INCOMPLETE rather than merely
            # uncertain, so a sampled world can contain a card that was already
            # played. Searching that is worse than not searching.
            self.stats["degraded"] += 1
            return False
        return True

    # -------------------------------------------------------------- network
    def _network_action(self, state, seat, match_scores, legal_mask):
        local_obs, global_obs, _ = build_observation(state, seat, list(match_scores))

        # `build_observation` embeds its own FULL legal mask as a feature at offset
        # 443, which is what the network trained on and must not be narrowed.
        # `legal_mask` may already be narrower -- the SDK narrows it when the server
        # refuses a move and retries the turn -- and it applies to the logits only.
        local_t = torch.from_numpy(local_obs).unsqueeze(0).to(self.device)
        glob_t = torch.from_numpy(global_obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(
            np.ascontiguousarray(legal_mask, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _, new_hidden = self.model(
                local_t, glob_t, self.hidden, mask_t, is_sequence=False)
            action = int(torch.argmax(dist.probs, dim=-1).item())

        self.hidden = new_hidden
        return action

    # --------------------------------------------------------------- search
    def _deadline(self, state):
        raw = getattr(state, "deadline", None)
        if raw is None:
            return time.monotonic() + DEFAULT_BUDGET_S - self.safety_margin
        return raw - self.safety_margin

    def _search_action(self, state, seat, legal_mask):
        """Exact-solve PIMC. Returns None if it could not finish a single world."""
        deadline = self._deadline(state)
        cons = constraints(state, seat)
        if not cons.is_consistent():
            self.stats["infeasible"] += 1
            return None

        legal = np.flatnonzero(np.asarray(legal_mask))
        totals = np.zeros(38, dtype=np.float64)
        solved = 0

        while solved < self.worlds and time.monotonic() < deadline:
            try:
                hands = sample_determinization(state, seat, self.rng)
                if self.check_worlds:
                    cons.check(hands)
            except Infeasible:
                # Contradictory constraints point at a synchronisation problem, not
                # bad luck. Give up on searching this turn rather than retrying into
                # the clock.
                self.stats["infeasible"] += 1
                break

            try:
                totals[legal] += self._solve_world(state, seat, hands, legal)
            except Exception as exc:                          # noqa: BLE001
                # NEVER let a solver fault reach the SDK. An exception escaping `act`
                # means no move is dispatched, the turn times out, and the seat is
                # gone for the session -- so a bad world costs us the search, not the
                # seat. Logged loudly because it should never happen.
                self.stats["solve_errors"] += 1
                print(f"[composite][ERROR] solve failed, falling back to the "
                      f"network: {type(exc).__name__}: {exc}")
                return None
            solved += 1

        if solved == 0:
            return None
        self.stats["worlds"] += solved

        scores = np.where(np.asarray(legal_mask).astype(bool), totals, -np.inf)
        return int(np.argmax(scores))

    def _solve_world(self, state, seat, hands, legal):
        """One exact double-dummy solve, scored in game points.

        Shares `belot.search.composite.solve_world` with the offline player, so the
        live search is the same computation rather than a reimplementation of it.
        """
        return solve_world_for(state, seat, hands, legal)


# ------------------------------------------------------------------ factories
def build(**kwargs):
    """Build a CompositeAgent. Keyword values may arrive as strings, so coerce."""
    return CompositeAgent(
        checkpoint=kwargs.get("checkpoint"),
        worlds=int(kwargs.get("worlds", DEFAULT_D)),
        device=kwargs.get("device"),
        search=kwargs.get("search", True),
        safety_margin=float(kwargs.get("safety_margin", SAFETY_MARGIN_S)),
        seed=(int(kwargs["seed"]) if kwargs.get("seed") is not None else None),
        check_worlds=kwargs.get("check_worlds", False),
    )


def build_network_only(**kwargs):
    """The network with the search disabled -- what the first live session runs.

    Isolating the network first means a clean session proves the encoder, the action
    space and the platform loop before the search is added to the list of suspects.
    """
    kwargs = dict(kwargs)
    kwargs["search"] = False
    return build(**kwargs)
