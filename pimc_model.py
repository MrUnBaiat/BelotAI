"""
PIMC with a MODEL rollout policy -- and a critic-at-leaf variant.

HYPOTHESIS (v6 EXP-1). pimc.py rolls out with the greedy heuristic in all four
seats, and the trained model is +2.263 +- 0.470 pts/hand stronger than that
heuristic. So the search may be limited by the quality of the policy inside its
rollouts rather than by D. Supporting prior measurement, D=8 held fixed, only the
rollout policy varied, 260 swap-paired deals:

    PIMC(heuristic rollouts) - heuristic : +2.642 +- 0.558
    PIMC(random    rollouts) - heuristic : +1.300 +- 0.536
    PAIRED difference                    : +1.342 +- 0.676  SIGNIFICANT

Upgrading random -> heuristic doubles the search's edge at identical D. The model
sits another +2.26 above the heuristic, so model rollouts should push the teacher
well past pimc.py's measured +3.015.

DESIGN NOTES
  * Determinization logic is IMPORTED from pimc.py, not reimplemented -- that is
    the part with the executed correctness suite (v5_01: 0 illegal, 0 malformed
    worlds, 100% impossible_cards respected, global RNG untouched).
  * Rollout decisions use a ZERO LSTM hidden state. Re-verified for this exact
    checkpoint in v6_00: zeroed - carried = +0.021 +- 0.248, a null. So no
    per-(world, seat) recurrent threading is needed.
  * Rollouts are BATCHED. One decision spawns D x |legal| worlds; they are stepped
    in lockstep with ONE forward pass per macro-step, the same trick vec_env.py
    uses for training. Without this the per-decision forward passes dominate and
    the teacher is too slow to generate data with.
  * Rollout policy is GREEDY (argmax), matching the determinism of the heuristic
    rollout policy it is being compared against, so the only variable is policy
    quality.
  * Worlds inherit the real game's `impossible_cards` / `known_cards`, so the
    rollout policy's own belief matrix is built from the same inferences the real
    seat would hold. pimc.py's _load zeroes these (harmless there -- the heuristic
    ignores observations entirely -- but wrong for a model rollout policy).

LEAF MODES
  "rollout" : play each world to the end of the hand with the model. Exact, slow.
  "critic"  : apply the candidate action, then score the resulting state with the
              privileged critic (held-out EV +0.621 in v4_07). Once a world is
              determinized all four hands are known, so the global observation the
              critic wants is exactly available. Vastly cheaper -- one batched
              forward per DECISION instead of one per macro-step.
              CAVEAT, stated because it limits the result: the critic was fit on
              ON-POLICY states, and determinized hypotheticals are a distribution
              shift, so it may underperform its headline EV.
"""
import numpy as np
import torch

from eval import _heuristic_action
from observation import build_observation
from pimc import _scratch_env, sample_determinization

HIDDEN = 512


def _load_full(w, env, hands):
    """Like pimc._load but ALSO carries the inference arrays, the dealer and the
    last trick, all of which feed build_observation and therefore the model."""
    w.hands = [list(h) for h in hands]
    w.trump = env.trump
    w.declarer = env.declarer
    w.declaring_team = env.declaring_team
    w.defending_team = env.defending_team
    w.declarer_has_played_trump = env.declarer_has_played_trump
    w.current_trick = list(env.current_trick)
    w.current_player = env.current_player
    w.tricks_played = env.tricks_played
    w.tricks_won_by_team = list(env.tricks_won_by_team)
    w.raw_points_by_team = list(env.raw_points_by_team)
    w.graveyard = list(env.graveyard)
    w.bolts_by_team = list(env.bolts_by_team)
    w.dealer = env.dealer
    w.last_trick = list(env.last_trick)
    np.copyto(w.impossible_cards, env.impossible_cards)
    np.copyto(w.known_cards, env.known_cards)
    w.phase = "PLAYING"
    w.done = False


def make_model_pimc(model, device, D=8, seed=0, leaf="rollout", scores=(0, 0),
                    opponent_rollout=None):
    """Returns action_fn(belot) -> int, plus a .stats dict of timing counters.

    `opponent_rollout`: if "heuristic", seats NOT on the searcher's team are rolled
    out with the greedy heuristic instead of the model.

    WHY THIS OPTION EXISTS. v6 EXP-1 measured PIMC(model rollouts) at -0.529 +-
    0.364 BELOW PIMC(heuristic rollouts) -- a stronger rollout policy made the
    search worse. But PIMC rolls out all four seats, and in that evaluation the
    real opponents WERE the greedy heuristic, so the heuristic-rollout variant was
    modelling its opponents exactly correctly while the model-rollout variant was
    modelling them wrongly. The comparison therefore confounds rollout-policy
    STRENGTH with opponent-model ACCURACY. This flag separates them: own team
    modelled by the model, opponents modelled by the heuristic they actually are.
    """
    rng = np.random.default_rng(seed)
    pool = []
    stats = {"decisions": 0, "worlds": 0, "forwards": 0}

    def world(i):
        while len(pool) <= i:
            pool.append(_scratch_env())
        return pool[i]

    def batched_forward(envs):
        n = len(envs)
        loc = np.empty((n, 513), dtype=np.float32)
        glb = np.empty((n, 332), dtype=np.float32)
        msk = np.empty((n, 38), dtype=np.float32)
        for j, w in enumerate(envs):
            l, g, m = build_observation(w, w.current_player, list(scores))
            loc[j], glb[j], msk[j] = l, g, m
        h = torch.zeros(1, n, HIDDEN, device=device)          # zero state, see v6_00
        c = torch.zeros(1, n, HIDDEN, device=device)
        with torch.no_grad():
            dist, value, _ = model(torch.from_numpy(loc).to(device),
                                   torch.from_numpy(glb).to(device), (h, c),
                                   torch.from_numpy(msk).to(device), is_sequence=False)
        stats["forwards"] += 1
        return dist.probs.argmax(-1).cpu().numpy(), value.squeeze(-1).cpu().numpy()

    def critic_only(envs, me):
        """Value of each world from seat `me`'s perspective."""
        n = len(envs)
        glb = np.empty((n, 332), dtype=np.float32)
        loc = np.zeros((n, 513), dtype=np.float32)
        msk = np.ones((n, 38), dtype=np.float32)
        for j, w in enumerate(envs):
            _, g, _ = build_observation(w, me, list(scores))
            glb[j] = g
        h = torch.zeros(1, n, HIDDEN, device=device)
        c = torch.zeros(1, n, HIDDEN, device=device)
        with torch.no_grad():
            _, value, _ = model(torch.from_numpy(loc).to(device),
                                torch.from_numpy(glb).to(device), (h, c),
                                torch.from_numpy(msk).to(device), is_sequence=False)
        stats["forwards"] += 1
        return value.squeeze(-1).cpu().numpy()

    def act(env):
        if env.phase == "BIDDING":
            return _heuristic_action(env)
        legal = np.flatnonzero(env.get_legal_actions())
        if len(legal) == 1:
            return int(legal[0])
        me = env.current_player
        team = me % 2
        stats["decisions"] += 1

        # --- build the D x |legal| root worlds ---
        live, idx_of, finished = [], [], []
        k = 0
        for _ in range(D):
            hands = sample_determinization(env, me, rng)
            if hands is None:
                continue
            for i, a in enumerate(legal):
                w = world(k); k += 1
                _load_full(w, env, hands)
                w._handle_playing_action(int(a))
                (finished if w.done else live).append((w, i))
        stats["worlds"] += k
        if k == 0:
            return _heuristic_action(env)

        totals = np.zeros(len(legal))
        counts = np.zeros(len(legal))

        if leaf == "critic":
            # A world that already ended scores exactly; the rest are estimated.
            for w, i in finished:
                gp = w._calculate_final_rewards()
                totals[i] += gp[team] - gp[1 - team]; counts[i] += 1
            if live:
                vals = critic_only([w for w, _ in live], me)
                for (w, i), v in zip(live, vals):
                    # critic predicts return-to-go on the /16 zero-sum scale
                    totals[i] += float(v) * 16.0; counts[i] += 1
        else:
            while live:
                if opponent_rollout == "heuristic":
                    # Opponents are rolled out by the heuristic they actually are;
                    # only the searcher's own team uses the model. Batch the model
                    # seats together and take the heuristic seats directly.
                    mine = [(w, i) for w, i in live if w.current_player % 2 == team]
                    theirs = [(w, i) for w, i in live if w.current_player % 2 != team]
                    acts_map = {}
                    if mine:
                        a_mine, _ = batched_forward([w for w, _ in mine])
                        for (w, _), a in zip(mine, a_mine):
                            acts_map[id(w)] = int(a)
                    for w, _ in theirs:
                        acts_map[id(w)] = _heuristic_action(w)
                    acts = [acts_map[id(w)] for w, _ in live]
                else:
                    acts, _ = batched_forward([w for w, _ in live])
                nxt = []
                for (w, i), a in zip(live, acts):
                    w._handle_playing_action(int(a))
                    (finished if w.done else nxt).append((w, i))
                live = nxt
            for w, i in finished:
                gp = w._calculate_final_rewards()
                totals[i] += gp[team] - gp[1 - team]; counts[i] += 1

        mean = totals / np.maximum(counts, 1)
        return int(legal[int(np.argmax(mean))])

    act.stats = stats
    return act
