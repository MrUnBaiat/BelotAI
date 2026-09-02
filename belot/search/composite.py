"""
The composite player -- the strongest Belot agent this project produced.

    model bidding + model card play at tricks 0-2 + exact-solve PIMC (D=8) from trick 3

Measured on the swap-paired instrument, identical-policy control exactly zero:

    +0.974 +- 0.175 pts/hand over the bare agent          (n=1500)
    +0.936 +- 0.243 pts/hand against a held-out opponent  (n=500)

WHY THIS CONFIGURATION AND NOT ANOTHER. Every parameter below is a measurement, not
a preference:

  min_trick = 3   Extending the search into tricks 0-2 was measured at
                  -0.348 +- 0.309 pts/hand -- significantly WORSE. At trick 0 there
                  are 24 unseen cards, so a single determinization's verdict is
                  nearly uninformative; searching there adds variance, not skill.
                  Searching tricks 0-2 on their own saturates near +0.2 even at
                  D=128, so the window is not worth buying at any affordable price.

  D = 8           Doubling to D=16 was measured as equal strength at equal cost, so
                  D=8 is the cheaper of two equals. The determinization axis is flat
                  in this window.

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

from belot.heuristic import _heuristic_action
from belot.model import RecurrentMAPPOModel
from belot.observation import build_observation
from belot.search.dd_solver import hands_to_masks, solve_root
from belot.search.pimc import sample_determinization

HIDDEN = 512
TOTAL_RAW = 162          # 152 in cards + 10 for the last trick


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


def make_dd_pimc(D=8, seed=0, min_trick=3):
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
        team = me % 2
        collected0 = env.raw_points_by_team[0]
        trick = tuple((p, c) for p, c in env.current_trick)
        tot = np.zeros(len(legal))
        n = 0
        for _ in range(D):
            hands = sample_determinization(env, me, state["rng"])
            if hands is None:
                continue
            masks = hands_to_masks(hands)
            _, vals, _ = solve_root(masks, me, trick, env.trump, env.declarer,
                                    env.declarer_has_played_trump)
            for i, a in enumerate(legal):
                rem0 = vals.get(int(a))
                if rem0 is None:
                    continue
                tot[i] += gp_diff_from_raw(collected0 + rem0,
                                           env.declaring_team, team)
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


def build_player(ckpt, D=8, min_trick=3, seed=0, device=None):
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
