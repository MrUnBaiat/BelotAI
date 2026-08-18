"""
AUDIT v4 / EXP-6 -- the variance budget, and the critic's ceiling.

EXP-1/4 showed the optimizer step is noise-dominated. Two fixes are available:
raise the batch (brute force), or LOWER tr(Sigma). This measures how much of the
second is even possible.

Advantages are unit-normalised before the update, so tr(Sigma) is essentially
E[||grad log pi||^2 * A^2]. Cutting it means making A less noisy, and the amount
of reducible noise is bounded by the variance decomposition of the episode target

    target = (gp_us - gp_them)/16

into a BETWEEN-DEALS component and a WITHIN-DEAL component:

  * BETWEEN-DEALS is the variance of E[target | deal]. It is pure card luck. A
    perfect privileged critic -- which is exactly what this project's critic is,
    since global_obs contains all four hands -- removes ALL of it. So the
    between-deals share is the critic's explained-variance CEILING, and the gap
    between that ceiling and the measured EV ~0.65 is the free variance still on
    the table.
  * WITHIN-DEAL is what the four agents' own action sampling produces on the SAME
    cards. No state-value baseline can remove it. It contains the actual learning
    signal plus three other seats' noise.

METHOD. Deterministic deal replay: re-seed numpy per deal so the identical 32
cards are dealt K times, and let torch's RNG run free so each replay samples
different actions. Pure self-play (the 70% majority of training data), current
policy, stochastic sampling exactly as in rollout.

WHY IT MATTERS. If between-deals is large and the critic is far below it, the
critic is the cheapest variance win available. If between-deals is small, the
noise is irreducible by any baseline and only batch size (or duplicate-deal
sampling) can help.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import train as T
from env import BelotEnv
from model import RecurrentMAPPOModel
from observation import build_observation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_DEALS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
K_REPLAY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE_SEED = 900_000


@torch.no_grad()
def play_hand(model, deal_seed, dealer):
    """Play one hand of pure self-play. Deal fixed by deal_seed; actions sampled
    from torch's RNG, which is deliberately NOT reset between replays."""
    env = BelotEnv()
    env.dealer = dealer
    np.random.seed(deal_seed)          # <- the deal, and only the deal
    env.reset()
    scores = [0, 0]
    hc = {s: (torch.zeros(1, 1, T.HIDDEN, device=DEVICE),
              torch.zeros(1, 1, T.HIDDEN, device=DEVICE)) for s in range(4)}
    vals0, info = [], {}
    while not env.done:
        seat = env.current_player
        local, glob, mask = build_observation(env, seat, scores)
        dist, value, hc[seat] = model(
            torch.from_numpy(local).unsqueeze(0).to(DEVICE),
            torch.from_numpy(glob).unsqueeze(0).to(DEVICE),
            hc[seat],
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(DEVICE),
            is_sequence=False)
        if seat == 0:
            vals0.append(float(value.item()))
        _, _, _, info = env.step(int(dist.sample().item()))
    gp = info["game_points"]
    return (gp[0] - gp[1]) / 16.0, vals0


def main():
    ck = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    torch.manual_seed(31337)

    tgt = np.zeros((D_DEALS, K_REPLAY))
    v_first = np.zeros((D_DEALS, K_REPLAY))   # critic at seat 0's FIRST decision
    v_last = np.zeros((D_DEALS, K_REPLAY))    # critic at seat 0's LAST decision
    for d in range(D_DEALS):
        for k in range(K_REPLAY):
            t, vs = play_hand(model, BASE_SEED + d, d % 4)
            tgt[d, k] = t
            v_first[d, k] = vs[0] if vs else np.nan
            v_last[d, k] = vs[-1] if vs else np.nan
        if (d + 1) % 50 == 0:
            print(f"  {d + 1}/{D_DEALS} deals", flush=True)

    flat = tgt.reshape(-1)
    deal_means = tgt.mean(1)
    # unbiased one-way ANOVA decomposition
    within = tgt.var(axis=1, ddof=1).mean()
    between_raw = deal_means.var(ddof=1)
    between = between_raw - within / K_REPLAY
    total = between + within
    print("\n" + "=" * 78)
    print(f"RETURN VARIANCE DECOMPOSITION  ({D_DEALS} deals x {K_REPLAY} replays "
          f"= {D_DEALS * K_REPLAY} hands)")
    print("=" * 78)
    print(f"  target = (gp_us - gp_them)/16 : mean {flat.mean():+.4f}  "
          f"sd {flat.std(ddof=1):.4f}  (= {flat.std(ddof=1) * 16:.2f} game points)")
    print(f"  total variance                : {total:.5f}")
    print(f"  BETWEEN deals (card luck)     : {between:.5f}   {between / total:6.1%}"
          "   <- perfect privileged critic removes this")
    print(f"  WITHIN deal  (action sampling): {within:.5f}   {within / total:6.1%}"
          "   <- irreducible by any state baseline")

    # bootstrap CI on the between share
    rng = np.random.default_rng(0)
    sh = []
    for _ in range(3000):
        s = rng.integers(0, D_DEALS, D_DEALS)
        t2 = tgt[s]
        w = t2.var(axis=1, ddof=1).mean()
        b = t2.mean(1).var(ddof=1) - w / K_REPLAY
        sh.append(b / (b + w))
    lo, hi = np.percentile(sh, [2.5, 97.5])
    print(f"  between-deals share 95% CI    : [{lo:.1%}, {hi:.1%}]")

    print("\n  CRITIC vs ITS CEILING")
    # Only the FIRST decision is a valid comparison against the episode target:
    # return-to-go at t=0 IS the whole target (gamma=0.999 over ~9 steps). For any
    # t>0, V(s_t) predicts the RESIDUAL return-to-go, so scoring it against the
    # full episode target measures nothing. Per-timestep EV is done properly in
    # v4_07_critic.py against real discounted returns-to-go.
    ok = ~np.isnan(v_first)
    ev0 = 1.0 - (tgt[ok] - v_first[ok]).var(ddof=1) / tgt[ok].var(ddof=1)
    print(f"    EV of the real critic at the hand's FIRST decision    : {ev0:+.3f}")
    print(f"    EV ceiling at that state (between-deals share)        : "
          f"{between / total:+.3f}")
    print(f"    -> critic error contributes "
          f"{(between / total - ev0) / (1 - ev0):.0%} of the residual variance there")
    print(f"    EV logged during training (Diag/ExplainedVariance)    : +0.658")

    print("\n  IMPLICATION FOR tr(Sigma)")
    gap = between / total - 0.658
    if gap > 0.02:
        print(f"    The critic leaves {gap:.1%} of total variance unexplained that a")
        print(f"    perfect privileged critic could remove. Closing it would cut the")
        print(f"    advantage variance by {gap / (1 - 0.658):.1%} and B_simple by the same")
        print(f"    factor -- real, but far short of the 80x-1500x the batch is short by.")
    else:
        print(f"    The critic is already at/near its ceiling ({between / total:.1%});")
        print(f"    no baseline improvement can meaningfully cut tr(Sigma). The noise")
        print(f"    is dominated by the WITHIN-deal term, so only more episodes per")
        print(f"    optimizer step (or duplicate-deal sampling) can raise the SNR.")


if __name__ == "__main__":
    main()
