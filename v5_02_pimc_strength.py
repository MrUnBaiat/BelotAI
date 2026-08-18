"""
AUDIT v5 / EXP-2 -- how strong is PIMC, and is it above the model?

AUDIT v3 claimed PIMC scores +3.06 pts/hand vs the greedy heuristic at D=32. That
file was never committed and is gone, so the number is unverified. It matters a
lot: if a search player with NO learning outscores the 2,400-epoch model, then
(a) there is provable headroom above the current policy, (b) PIMC is a valid
non-saturating yardstick, and (c) search distillation becomes the obvious lever.

Everything is PAIRED on identical deals (handoff section 8.3). pimc.py never
touches the global numpy RNG (verified in v5_01), so re-seeding reproduces the
same cards for every condition and the deal-luck term -- 73.8% of outcome
variance per v4_06 -- cancels in the differences.

Conditions, all as team 0 against the greedy heuristic as team 1:
    heuristic-vs-heuristic   sanity floor, must be ~0.0
    PIMC(D)                  the candidate yardstick
    model (epoch 2450)       the current agent
and then the head-to-head that actually decides it: model vs PIMC.
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DEALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
D = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HIDDEN = 512
BASE = 770_000          # same deal seeds as v4_paired_eval


def net_actor(model):
    hc = {}

    def reset():
        hc.clear()
        for s in range(4):
            hc[s] = (torch.zeros(1, 1, HIDDEN, device=DEVICE),
                     torch.zeros(1, 1, HIDDEN, device=DEVICE))

    @torch.no_grad()
    def act(env, scores):
        seat = env.current_player
        local, glob, mask = build_observation(env, seat, scores)
        dist, _, hc[seat] = model(
            torch.from_numpy(local).unsqueeze(0).to(DEVICE),
            torch.from_numpy(glob).unsqueeze(0).to(DEVICE), hc[seat],
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(DEVICE),
            is_sequence=False)
        return int(dist.probs.argmax(-1).item())
    return act, reset


def duel(act0, reset0, act1, reset1, n_deals):
    """act_i(env, scores) -> action. Team 0 = seats 0&2, team 1 = seats 1&3."""
    out = np.empty(n_deals)
    for d in range(n_deals):
        env = BelotEnv()
        env.dealer = d % 4
        np.random.seed(BASE + d)
        env.reset()
        env.bolts_by_team = [0, 0]
        scores = [0, 0]
        if reset0:
            reset0()
        if reset1:
            reset1()
        info = {}
        while not env.done:
            a = (act0 if env.current_player % 2 == 0 else act1)(env, scores)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        out[d] = gp[0] - gp[1]
    return out


def ci(d):
    return 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def report(name, d):
    print(f"  {name:<34s} {d.mean():+.3f} +- {ci(d):.3f} pts/hand")


def paired(name, a, b):
    d = a - b
    v = "SIGNIFICANT" if abs(d.mean()) > ci(d) else "not significant"
    print(f"  {name:<40s} {d.mean():+.3f} +- {ci(d):.3f}  [{v}]")


def main():
    ck = torch.load("checkpoints/best_model.pt", map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"model: epoch {ck.get('epoch')} metric {ck.get('selection_metric')!r}\n")

    heur = lambda env, sc: _heuristic_action(env)
    pimc_fn = make_pimc(D=D, seed=0)
    pimc = lambda env, sc: pimc_fn(env)
    net_act, net_reset = net_actor(model)

    print(f"{N_DEALS} paired deals, PIMC D={D}\n")
    print("=" * 78)
    print("ABSOLUTE, as team 0 vs the greedy heuristic")
    print("=" * 78)
    t = time.time()
    d_h = duel(heur, None, heur, None, N_DEALS)
    report("heuristic (sanity floor, expect ~0)", d_h)

    t = time.time()
    d_p = duel(pimc, None, heur, None, N_DEALS)
    report(f"PIMC D={D}", d_p)
    print(f"    ({time.time()-t:.0f}s, {(time.time()-t)/N_DEALS*1000:.0f} ms/hand)")

    d_m = duel(net_act, net_reset, heur, None, N_DEALS)
    report("model (epoch 2450)", d_m)

    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCES on identical deals")
    print("=" * 78)
    paired("PIMC - model  (vs heuristic)", d_p, d_m)
    paired("PIMC - heuristic", d_p, d_h)
    paired("model - heuristic", d_m, d_h)

    print("\n" + "=" * 78)
    print("HEAD TO HEAD: model (team 0) vs PIMC (team 1)")
    print("=" * 78)
    d_mp = duel(net_act, net_reset, pimc, None, N_DEALS)
    report("model vs PIMC", d_mp)
    print(f"\n  v3 claimed PIMC = +3.06 vs heuristic at D=32; measured here "
          f"{d_p.mean():+.3f} +- {ci(d_p):.3f} at D={D}")
    if d_p.mean() - d_m.mean() > 0:
        print("  -> PIMC is ABOVE the model: a valid non-saturating yardstick, and")
        print("     provable headroom above the current policy.")
    else:
        print("  -> PIMC is NOT above the model at this D; raise D or improve the")
        print("     rollout policy before using it as a yardstick.")


if __name__ == "__main__":
    main()
