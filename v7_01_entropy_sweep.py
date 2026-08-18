"""
AUDIT v7 / EXP-B1 -- does the entropy bonus move entropy at ANY coefficient?

v5 shipped a "re-heat" at coefficient 0.025 that was INERT: entropy went 0.086 ->
0.077 in bidding, i.e. it did not move. That was predictable from a number already
measured in v4_03 (|G_entropy| = 0.00039 vs |G_policy| = 0.1138, so 0.34% of the
update at coef 0.005) and not applied. This is the precondition check that was
skipped.

THE MECHANISM, verified read-only before running. For a Categorical,
dH/dz_i = -p_i (log p_i + H), and its norm collapses as the policy sharpens --
exactly where exploration is needed. At k=8 legal actions:

    max prob   ||dH/dlogits||
    0.983      1.07e-1
    0.998      1.74e-2
    0.9997     3.23e-3

The measured operating point (pi(sampled) median 1.000 bidding / 0.989 card play)
is the bottom row. Linear scaling off v4_03's anchor predicts ~1.7% of the update
at 0.025 (measured inert) and ~14% at 0.2.

FALSIFIABLE PREDICTION UNDER TEST: 0.05 and 0.1 near-inert, 0.2 the first
coefficient that moves anything, with a sharply NON-LINEAR response.

WHAT IS MEASURED. Entropy is the target here, not strength -- if entropy does not
move, strength cannot be attributed to exploration and the bonus is simply the
wrong mechanism. Reports H(pi) by phase and the median pi(sampled action), which
is the quantity that actually says whether the policy is still deterministic.

DECISION: no movement even at 0.2 -> skip the coefficient entirely and go straight
to the mixture behaviour policy (EXP-B2), which cannot saturate by construction.
"""
import copy
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, '.')
import train_v4 as T4
from memory import make_minibatch
from model import RecurrentMAPPOModel
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COEFS = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                            else ["0.05", "0.1", "0.2"])]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TARGET_GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
LR_MULT = 24.0
START = "checkpoints/v6_exp/c1_continued.pt"
START_EPOCH = 2848


@torch.no_grad()
def policy_stats(model, vec, device, n_games=256):
    """H(pi) by phase and median pi(sampled) -- measured on freshly collected
    on-policy data, not on the training minibatch, so it is not contaminated by
    the update that just happened."""
    eps, _ = T4.collect_rollout(model, vec, n_games, device, [])
    # make_minibatch pads ep.advantages/ep.returns, which only update() populates.
    # These are unused here (we only read the policy), but they must exist.
    for ep in eps:
        ep.returns, ep.advantages = ep.compute_gae(T4.GAMMA, T4.LAM)
    b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(eps[:512])
    to = lambda x: x.to(device)
    dist, _, _ = model(to(b_obs), to(b_gobs),
                       (torch.zeros(1, b_obs.size(0), T4.HIDDEN, device=device),
                        torch.zeros(1, b_obs.size(0), T4.HIDDEN, device=device)),
                       to(b_masks), is_sequence=True)
    pad_b = to(pad).bool()
    ent = dist.entropy()
    p_a = torch.exp(dist.log_prob(to(b_actions)))
    bid = (to(b_masks)[..., 32:].sum(-1) > 0) & pad_b
    play = (~bid) & pad_b
    return {
        "H_bid": float(ent[bid].mean()) if bid.any() else float("nan"),
        "H_play": float(ent[play].mean()) if play.any() else float("nan"),
        "p_med_bid": float(p_a[bid].float().median()) if bid.any() else float("nan"),
        "p_med_play": float(p_a[play].float().median()) if play.any() else float("nan"),
    }


def run(coef, base_sd, base_opt_sd):
    torch.manual_seed(0); np.random.seed(0)
    model = RecurrentMAPPOModel(hidden_dim=T4.HIDDEN).to(DEVICE)
    model.load_state_dict(base_sd)
    opt = optim.Adam(model.parameters(), lr=T4.LR_START)
    opt.load_state_dict(copy.deepcopy(base_opt_sd))
    vec = VectorizedBelot(T4.NUM_ENVS)

    before = policy_stats(model, vec, DEVICE)
    t0 = time.time()
    for i in range(EPOCHS):
        lr = T4.anneal(T4.LR_START, T4.LR_END, START_EPOCH + i,
                       T4.LR_ANNEAL_EPOCHS) * LR_MULT
        for g in opt.param_groups:
            g["lr"] = lr
        eps, _ = T4.collect_rollout(model, vec, TARGET_GAMES, DEVICE, [])
        T4.update(model, opt, eps, DEVICE, entropy_coef=coef)
    after = policy_stats(model, vec, DEVICE)
    return before, after, time.time() - t0


def main():
    ck = torch.load(START, map_location=DEVICE, weights_only=False)
    base_sd = ck["model_state_dict"]
    base_opt = ck["optimizer_state_dict"]
    print(f"start {START} (epoch {ck.get('epoch')}), {EPOCHS} epochs x "
          f"{TARGET_GAMES} games per coefficient\n")
    print("v5 measured coefficient 0.025 as INERT (bidding H 0.086 -> 0.077).")
    print("Prediction under test: 0.05/0.1 near-inert, 0.2 the first to move.\n")
    print(f"{'coef':>6} {'H_bid before->after':>26} {'H_play before->after':>26} "
          f"{'pi_med bid':>16} {'pi_med play':>16} {'s':>5}")
    rows = []
    for c in COEFS:
        b, a, dt = run(c, base_sd, base_opt)
        rows.append((c, b, a))
        print(f"{c:>6} {b['H_bid']:>11.4f} ->{a['H_bid']:>11.4f} "
              f"{b['H_play']:>11.4f} ->{a['H_play']:>11.4f} "
              f"{b['p_med_bid']:>7.3f}->{a['p_med_bid']:>7.3f} "
              f"{b['p_med_play']:>7.3f}->{a['p_med_play']:>7.3f} {dt:>5.0f}",
              flush=True)

    print("\n" + "=" * 78)
    moved = [(c, a['H_bid'] - b['H_bid'], a['H_play'] - b['H_play'])
             for c, b, a in rows]
    for c, db, dp in moved:
        print(f"  coef {c:<6} dH_bid {db:+.4f}   dH_play {dp:+.4f}")
    best = max(moved, key=lambda r: r[1] + r[2])
    if best[1] + best[2] > 0.02:
        print(f"\n=> The bonus DOES move entropy at coef {best[0]}. Check whether the")
        print("   response is non-linear across the sweep, then test strength (EXP-B2).")
    else:
        print("\n=> INERT AT EVERY COEFFICIENT TESTED, including the largest. The entropy")
        print("   bonus is the wrong mechanism at this operating point -- exactly what")
        print("   the dH/dlogits collapse predicts. Skip the coefficient and go straight")
        print("   to the mixture behaviour policy, which cannot saturate by construction.")


if __name__ == "__main__":
    main()
