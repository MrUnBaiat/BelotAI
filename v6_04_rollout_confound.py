"""
AUDIT v6 / EXP-1b -- separating rollout STRENGTH from opponent-model ACCURACY.

EXP-1 measured PIMC(model rollouts) - PIMC(heuristic rollouts) = -0.529 +- 0.364,
SIGNIFICANT: a stronger rollout policy made the search WORSE. Before accepting
"stronger rollouts hurt" as the finding, there is a confound to rule out.

PIMC rolls out ALL FOUR seats. In that evaluation the real opponents were the
greedy heuristic. So:
  * PIMC(heuristic rollouts) modelled its opponents EXACTLY correctly, and
  * PIMC(model rollouts) modelled them wrongly, as copies of the model.
The comparison therefore mixes two effects with opposite expected signs. If
opponent-model accuracy dominates, the "stronger rollouts hurt" reading is an
artifact of the test setup rather than a property of MC search.

THE SEPARATING CONDITION. Roll out the searcher's own team with the model and the
opponents with the greedy heuristic they actually are.

    mixed - heuristic-rollouts > 0   -> rollout strength DOES help once the
                                        opponent model is right; EXP-1's negative
                                        result was the confound.
    mixed ~ model-rollouts           -> opponent modelling was not the issue;
                                        model rollouts are genuinely bad here,
                                        and "stronger != better rollout policy"
                                        stands as the finding.

Paired on the same deals as EXP-1 so every number is directly comparable.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from env import BelotEnv
from eval import _heuristic_action
from model import RecurrentMAPPOModel
from pimc import make_pimc
from pimc_model import make_model_pimc, HIDDEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DEALS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CKPT = "checkpoints/v4_exp/c1_latest.pt"
BASE = 770_000


def duel(act0, n_deals):
    out = np.empty(n_deals)
    t0 = time.time()
    for d in range(n_deals):
        env = BelotEnv(); env.dealer = d % 4
        np.random.seed(BASE + d); env.reset(); env.bolts_by_team = [0, 0]
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
    return f"  {name:<46s} {d.mean():+.3f} +- {ci(d):.3f}  [{v}]"


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print(f"{N_DEALS} paired deals, D={D}\n")

    res, ms = {}, {}
    for name, act in (
        ("PIMC heuristic-rollouts", make_pimc(D=D, seed=0)),
        ("PIMC model-rollouts", make_model_pimc(model, DEVICE, D=D, seed=0)),
        ("PIMC mixed (model own / heuristic opp)",
         make_model_pimc(model, DEVICE, D=D, seed=0, opponent_rollout="heuristic")),
    ):
        res[name], ms[name] = duel(act, N_DEALS)
        print(f"  {name:<40s} {res[name].mean():+.3f} +- {ci(res[name]):.3f}"
              f"   {ms[name]:>7.0f} ms/hand", flush=True)

    print("\n" + "=" * 82)
    print("PAIRED DIFFERENCES")
    print("=" * 82)
    print(paired("mixed - heuristic-rollouts  [THE TEST]",
                 res["PIMC mixed (model own / heuristic opp)"],
                 res["PIMC heuristic-rollouts"]))
    print(paired("mixed - model-rollouts",
                 res["PIMC mixed (model own / heuristic opp)"],
                 res["PIMC model-rollouts"]))
    print(paired("model-rollouts - heuristic-rollouts  [EXP-1]",
                 res["PIMC model-rollouts"], res["PIMC heuristic-rollouts"]))

    a = res["PIMC mixed (model own / heuristic opp)"] - res["PIMC heuristic-rollouts"]
    b = res["PIMC mixed (model own / heuristic opp)"] - res["PIMC model-rollouts"]
    print("\n" + "=" * 82)
    if a.mean() > 0 and abs(a.mean()) > ci(a):
        print("VERDICT: the confound was real. Rollout-policy strength DOES help once")
        print("the opponent model is correct; EXP-1's negative result came from")
        print("mis-modelling opponents, not from stronger rollouts being harmful.")
    elif abs(b.mean()) <= ci(b):
        print("VERDICT: opponent modelling was NOT the issue -- the mixed variant sits")
        print("with the model-rollout variant. 'Stronger is not better as a rollout")
        print("policy' stands, which is a known MC-search effect (Gelly & Silver 2007).")
    else:
        print("VERDICT: partial. The mixed variant sits between the two; both effects")
        print("contribute and neither explanation alone accounts for EXP-1.")


if __name__ == "__main__":
    main()
