"""
AUDIT v6 / EXP-1 + EXP-2 -- is the teacher capped by its rollout policy?

HYPOTHESIS. pimc.py rolls out with the greedy heuristic, and the model is
+2.263 +- 0.470 pts/hand stronger than that heuristic, so the search is limited by
the policy inside its rollouts rather than by D. Prior measurement at fixed D=8,
varying ONLY the rollout policy: random -> heuristic doubled the search's edge
(+1.300 -> +2.642, paired difference +1.342 +- 0.676, significant).

CONDITIONS, all as team 0 vs the greedy heuristic as team 1, paired on identical
deals at the SAME D:
    heuristic                    sanity floor, expect ~0
    model (greedy)               the current agent
    PIMC(heuristic rollouts)     pimc.py -- the existing teacher
    PIMC(model rollouts)         pimc_model.py -- EXP-1
    PIMC(critic at leaf)         pimc_model.py -- EXP-2, ~20x cheaper

Wall-clock ms/hand is recorded for every condition, because feasibility is a
decision criterion here, not a footnote.

INSTRUMENT NOTE. Effects below ~0.5 pts/hand are invisible to the 250-match eval
(CI +-0.40) and must be read off the paired instrument. Applied literally to C1,
the 250-match rule would have declared this project's only successful fix a
failure. Everything below is paired.

DECISION RULE, fixed in advance:
  PIMC(model) - PIMC(heuristic) >= +0.5 significant AND <= 2 s/hand
        -> teacher upgrade CONFIRMED, re-price distillation against it
  >= +0.5 but > 10 s/hand
        -> strength confirmed, generation infeasible; fall back to EXP-2
  +0.0 to +0.3, or not significant
        -> teacher is NOT rollout-limited; do NOT distil, continue C1
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from env import BelotEnv
from eval import _heuristic_action
from model import RecurrentMAPPOModel
from observation import build_observation
from pimc import make_pimc
from pimc_model import make_model_pimc, HIDDEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DEALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
D = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CKPT = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/v4_exp/c1_latest.pt"
BASE = 770_000


def model_actor(model):
    hc = {}

    def reset():
        for s in range(4):
            hc[s] = (torch.zeros(1, 1, HIDDEN, device=DEVICE),
                     torch.zeros(1, 1, HIDDEN, device=DEVICE))

    @torch.no_grad()
    def act(env):
        seat = env.current_player
        l, g, m = build_observation(env, seat, [0, 0])
        dist, _, hc[seat] = model(
            torch.from_numpy(l).unsqueeze(0).to(DEVICE),
            torch.from_numpy(g).unsqueeze(0).to(DEVICE), hc[seat],
            torch.from_numpy(m.astype(np.float32)).unsqueeze(0).to(DEVICE),
            is_sequence=False)
        return int(dist.probs.argmax(-1).item())
    reset()
    return act, reset


def duel(act0, reset0, n_deals):
    """act0 plays seats 0&2; greedy heuristic plays 1&3. Same deals every time."""
    out = np.empty(n_deals)
    t0 = time.time()
    for d in range(n_deals):
        env = BelotEnv(); env.dealer = d % 4
        np.random.seed(BASE + d); env.reset(); env.bolts_by_team = [0, 0]
        if reset0:
            reset0()
        info = {}
        while not env.done:
            a = act0(env) if env.current_player % 2 == 0 else _heuristic_action(env)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        out[d] = gp[0] - gp[1]
    return out, (time.time() - t0) / n_deals * 1000.0


def ci(d):
    return 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def paired(name, a, b):
    d = a - b
    v = "SIGNIFICANT" if abs(d.mean()) > ci(d) else "not significant"
    return f"  {name:<44s} {d.mean():+.3f} +- {ci(d):.3f}  [{v}]"


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print(f"model: {CKPT} (epoch {ck.get('epoch')})")
    print(f"{N_DEALS} paired deals, D={D}, device {DEVICE}\n")

    net_act, net_reset = model_actor(model)
    conds = {}
    ms = {}

    for name, act, rst in (
        ("heuristic", lambda e: _heuristic_action(e), None),
        ("model (greedy)", net_act, net_reset),
        ("PIMC heuristic-rollouts", make_pimc(D=D, seed=0), None),
        ("PIMC critic-leaf", make_model_pimc(model, DEVICE, D=D, seed=0, leaf="critic"), None),
        ("PIMC model-rollouts", make_model_pimc(model, DEVICE, D=D, seed=0, leaf="rollout"), None),
    ):
        conds[name], ms[name] = duel(act, rst, N_DEALS)
        print(f"  {name:<26s} {conds[name].mean():+.3f} +- {ci(conds[name]):.3f} pts/hand"
              f"   {ms[name]:>8.0f} ms/hand", flush=True)

    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCES on identical deals")
    print("=" * 78)
    print(paired("PIMC(model) - PIMC(heuristic)  [EXP-1 TEST]",
                 conds["PIMC model-rollouts"], conds["PIMC heuristic-rollouts"]))
    print(paired("PIMC(model) - heuristic",
                 conds["PIMC model-rollouts"], conds["heuristic"]))
    print(paired("PIMC(model) - model  [distillation headroom]",
                 conds["PIMC model-rollouts"], conds["model (greedy)"]))
    print(paired("PIMC(critic) - PIMC(heuristic)  [EXP-2]",
                 conds["PIMC critic-leaf"], conds["PIMC heuristic-rollouts"]))
    print(paired("PIMC(critic) - model",
                 conds["PIMC critic-leaf"], conds["model (greedy)"]))
    print(paired("PIMC(heuristic) - model  [v5 baseline]",
                 conds["PIMC heuristic-rollouts"], conds["model (greedy)"]))
    print(paired("model - heuristic",
                 conds["model (greedy)"], conds["heuristic"]))

    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    d = conds["PIMC model-rollouts"] - conds["PIMC heuristic-rollouts"]
    sec = ms["PIMC model-rollouts"] / 1000.0
    sig = abs(d.mean()) > ci(d)
    print(f"  PIMC(model) - PIMC(heuristic) = {d.mean():+.3f} +- {ci(d):.3f}"
          f"   at {sec:.2f} s/hand")
    if d.mean() >= 0.5 and sig and sec <= 2.0:
        print("  -> CONFIRMED and feasible. Re-price distillation against this teacher.")
    elif d.mean() >= 0.5 and sig:
        print(f"  -> Strength confirmed but {sec:.1f} s/hand. Generation cost is the")
        print("     binding constraint; prefer the critic-leaf variant if it holds up.")
    elif d.mean() <= 0.3 or not sig:
        print("  -> NOT rollout-limited. Do NOT distil: at ~46% retention it buys less")
        print("     than continuing C1. Move to the section-5 fallback ranking.")
    else:
        print("  -> Between +0.3 and +0.5: inconclusive, needs more deals.")

    dh = conds["PIMC model-rollouts"] - conds["model (greedy)"]
    print(f"\n  distillation headroom vs the model = {dh.mean():+.3f} +- {ci(dh):.3f}")
    print(f"  at v3's measured ~46% retention that is "
          f"{0.46 * dh.mean():+.3f} pts/hand for a full BC run,")
    print(f"  against C1's measured +0.353 +- 0.218 for 150 epochs of ordinary training.")


if __name__ == "__main__":
    main()
