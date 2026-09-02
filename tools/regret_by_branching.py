"""
S1 -- REGRET BY BRANCHING FACTOR, for the model and for both searches.

THE GAP THIS FILLS. `06_ROUTES.md` §0.2 measured the heuristic evaluator's agreement with
exact play falling with k (0.855 / 0.629 / 0.436 at k = 2/3/4) while `s2_gap` measured the
value at stake RISING with k (+0.323 at k=2 to +0.787 at k=5). Those two facts point in
opposite directions and NEITHER settles the gate, because what a gate needs is not how
good the search is, nor how much is at stake, but

        (model regret  -  search regret)  x  frequency   per unit of cost

and the model's own regret by k has never been measured. This measures all three terms on
identical states.

THE REFERENCE, and its two honest limitations. The yardstick is exact-solve PIMC at
D_ref determinizations, scored in GAME points through `gp_diff_from_raw` -- the same
objective `make_dd_pimc` optimises, so the units are the ones the headline metric uses.

  1. It is NOT ground truth. E4 measured exact-solve PIMC at +2.110 +- 0.349 below a
     perfect-information player, so the reference is itself a long way from optimal. It
     is the best decision rule this project has, which is what a gate has to be built
     against.
  2. Exact-PIMC-at-play-D is estimating the SAME objective as the reference, so it would
     be flattered by shared samples. The reference draws its determinizations from a
     private stream that is DISJOINT from the players' (different seed base), so what is
     measured for it is genuine sampling noise at play D, not self-agreement.

FREE RIDERS. The same pass records, per decision: the model's top-1 probability (S2's
gate), and declarer/defender, trump-lead-availability and partner-winning (S6's gate). No
extra search cost, and it means those two proposals are tested on identical states rather
than in separate runs.

OUTPUT: `_s1_regret.npz` with one row per decision, plus the implied gate shape.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import torch


import belot.search.dd_solver as DD                              # noqa: E402
from belot.evaluation.swap_eval import ci
from belot.search.composite import (gp_diff_from_raw, load_model,
                                    make_dd_pimc, make_heur_pimc)  # noqa: E402
from belot.env import BelotEnv                            # noqa: E402
from belot.observation import build_observation           # noqa: E402
from belot.search.pimc import sample_determinization             # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 512


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def reference_values(env, rng, D_ref):
    """Exact-solve PIMC over D_ref determinizations, in GAME points for the mover's
    team. Returns (legal, value vector) or (legal, None) if no determinization drew."""
    legal = np.flatnonzero(env.get_legal_actions())
    me = env.current_player
    team = me % 2
    collected0 = env.raw_points_by_team[0]
    trick = tuple((p, c) for p, c in env.current_trick)
    tot = np.zeros(len(legal))
    n = 0
    for _ in range(D_ref):
        hands = sample_determinization(env, me, rng)
        if hands is None:
            continue
        masks = DD.hands_to_masks(hands)
        _, vals, _ = DD.solve_root(masks, me, trick, env.trump, env.declarer,
                                   env.declarer_has_played_trump)
        for i, a in enumerate(legal):
            rem0 = vals.get(int(a))
            if rem0 is None:
                continue
            tot[i] += gp_diff_from_raw(collected0 + rem0, env.declaring_team, team)
        n += 1
    if n == 0:
        return legal, None
    return legal, tot / n


def main():
    n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    D_ref = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    D_play = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    min_trick = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    budget = float(sys.argv[5]) if len(sys.argv) > 5 else 900.0

    st = np.random.get_state()
    net = load_model()
    ref_rng = np.random.default_rng(4_000_000)          # DISJOINT from the players'
    heur, heur_re = make_heur_pimc(D_play, 7_000_000, 0)
    exact, exact_re = make_dd_pimc(D_play, 7_000_000, 0)

    print(f"S1: reference = exact-solve PIMC D={D_ref} (game points), disjoint stream")
    print(f"    players   = model / rollout-PIMC D={D_play} / exact-PIMC D={D_play}")
    print(f"    states from model self-play, trick >= {min_trick}, "
          f"{n_hands} hands, budget {budget:.0f}s\n")

    rows = []
    t0 = time.time()
    for g in range(n_hands):
        if time.time() - t0 > budget:
            print(f"  budget reached after {g} hands", flush=True)
            break
        env = BelotEnv()
        env.dealer = g % 4
        np.random.seed(3_100_000 + g)
        env.reset()
        env.bolts_by_team = [0, 0]
        hc = {s: (torch.zeros(1, 1, HIDDEN, device=DEV),
                  torch.zeros(1, 1, HIDDEN, device=DEV)) for s in range(4)}
        heur_re(7_000_000 + g)
        exact_re(7_000_000 + g)
        while not env.done:
            s = env.current_player
            local, glob, mask = build_observation(env, s, [0, 0])
            with torch.no_grad():
                dist, _, hc[s] = net(
                    torch.from_numpy(local).unsqueeze(0).to(DEV),
                    torch.from_numpy(glob).unsqueeze(0).to(DEV), hc[s],
                    torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(DEV),
                    is_sequence=False)
            probs = dist.probs.squeeze(0).cpu().numpy()
            a_model = int(probs.argmax())
            legal = np.flatnonzero(env.get_legal_actions())

            if env.phase == "PLAYING" and len(legal) > 1 \
                    and env.tricks_played >= min_trick:
                lg, qref = reference_values(env, ref_rng, D_ref)
                if qref is not None:
                    idx = {int(c): i for i, c in enumerate(lg)}
                    t1 = time.time(); a_h = int(heur(env)); c_h = time.time() - t1
                    t1 = time.time(); a_e = int(exact(env)); c_e = time.time() - t1
                    best = float(qref.max())
                    # S6 context flags
                    cur_w = (max(env.current_trick, key=lambda pc: pc[1])[0]
                             if env.current_trick else -1)
                    rows.append((
                        len(lg), env.tricks_played, float(probs[a_model]),
                        best - float(qref[idx.get(a_model, 0)]),
                        best - float(qref[idx[a_h]]),
                        best - float(qref[idx[a_e]]),
                        c_h, c_e,
                        1 if s == env.declarer else 0,
                        1 if not env.current_trick else 0,
                        1 if (cur_w >= 0 and cur_w % 2 == s % 2) else 0,
                        float(qref.max() - qref.min()),
                    ))
            env.step(a_model)
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{n_hands} hands, {len(rows)} decisions, "
                  f"{time.time()-t0:.0f}s", flush=True)
    np.random.set_state(st)

    R = np.array(rows, float)
    cols = ("k trick ptop1 r_model r_heur r_exact c_heur c_exact "
            "is_decl is_lead partner_win spread").split()
    np.savez_compressed(os.path.join(_HERE, "_s1_regret.npz"), R=R,
                        cols=np.array(cols), D_ref=D_ref, D_play=D_play)
    C = {c: R[:, i] for i, c in enumerate(cols)}
    print(f"\n{len(R)} decisions in {time.time()-t0:.0f}s\n")

    print("=" * 78)
    print("A.  REGRET BY BRANCHING FACTOR  (game points per decision, vs the reference)")
    print("=" * 78)
    print(f"{'k':>3s} {'n':>5s} {'freq':>6s} | {'model':>15s} {'rollout':>15s} "
          f"{'exact':>15s} | {'spread':>7s}")
    for k in range(2, 9):
        m = C["k"] == k
        if m.sum() < 15:
            continue
        print(f"{k:>3d} {int(m.sum()):>5d} {m.mean()*100:>5.1f}% | "
              f"{C['r_model'][m].mean():>7.3f}+-{ci(C['r_model'][m]):<6.3f} "
              f"{C['r_heur'][m].mean():>7.3f}+-{ci(C['r_heur'][m]):<6.3f} "
              f"{C['r_exact'][m].mean():>7.3f}+-{ci(C['r_exact'][m]):<6.3f} | "
              f"{C['spread'][m].mean():>7.3f}")
    print(f"{'ALL':>3s} {len(R):>5d} {'100.0%':>6s} | "
          f"{C['r_model'].mean():>7.3f}+-{ci(C['r_model']):<6.3f} "
          f"{C['r_heur'].mean():>7.3f}+-{ci(C['r_heur']):<6.3f} "
          f"{C['r_exact'].mean():>7.3f}+-{ci(C['r_exact']):<6.3f} | "
          f"{C['spread'].mean():>7.3f}")

    print("\n" + "=" * 78)
    print("B.  THE GATE QUANTITY:  (model regret - search regret), paired, by k")
    print("=" * 78)
    print(f"{'k':>3s} {'n':>5s} | {'gain rollout':>20s} {'gain exact':>20s} | "
          f"{'ms/dec':>7s} {'ms/dec':>7s}")
    tot_h = tot_e = 0.0
    for k in range(2, 9):
        m = C["k"] == k
        if m.sum() < 15:
            continue
        gh = C["r_model"][m] - C["r_heur"][m]
        ge = C["r_model"][m] - C["r_exact"][m]
        tot_h += gh.sum(); tot_e += ge.sum()
        print(f"{k:>3d} {int(m.sum()):>5d} | {gh.mean():>+8.3f} +- {ci(gh):<8.3f} "
              f"{ge.mean():>+8.3f} +- {ci(ge):<8.3f} | "
              f"{C['c_heur'][m].mean()*1e3:>7.1f} {C['c_exact'][m].mean()*1e3:>7.1f}")
    gh = C["r_model"] - C["r_heur"]
    ge = C["r_model"] - C["r_exact"]
    print(f"{'ALL':>3s} {len(R):>5d} | {gh.mean():>+8.3f} +- {ci(gh):<8.3f} "
          f"{ge.mean():>+8.3f} +- {ci(ge):<8.3f} | "
          f"{C['c_heur'].mean()*1e3:>7.1f} {C['c_exact'].mean()*1e3:>7.1f}")

    print("\n" + "=" * 78)
    print("C.  VALUE PER UNIT COST -- the quantity a gate should sort on")
    print("=" * 78)
    print(f"{'k':>3s} | {'share of total gain':>20s} | {'gain per second of exact search':>32s}")
    for k in range(2, 9):
        m = C["k"] == k
        if m.sum() < 15:
            continue
        ge_k = (C["r_model"][m] - C["r_exact"][m]).sum()
        cost_k = C["c_exact"][m].sum()
        print(f"{k:>3d} | {100*ge_k/max(tot_e,1e-9):>18.1f}% | "
              f"{ge_k/max(cost_k,1e-9):>32.3f}")

    print("\n" + "=" * 78)
    print("D.  THE IMPLIED GATE -- greedy by gain-per-second, cumulative")
    print("=" * 78)
    order = []
    for k in range(2, 9):
        m = C["k"] == k
        if m.sum() < 15:
            continue
        ge_k = (C["r_model"][m] - C["r_exact"][m]).sum()
        order.append((ge_k / max(C["c_exact"][m].sum(), 1e-9), k, ge_k,
                      C["c_exact"][m].sum(), int(m.sum())))
    order.sort(reverse=True)
    cg = cc = 0.0
    n_dec = len(R)
    print(f"{'add k':>6s} {'rate':>9s} | {'cum gain/dec':>13s} {'cum ms/dec':>11s} "
          f"{'cum decisions searched':>23s}")
    for rate, k, g_, c_, nk in order:
        cg += g_; cc += c_
        print(f"{k:>6d} {rate:>9.3f} | {cg/n_dec:>13.4f} {cc/n_dec*1e3:>11.2f} "
              f"{'':>10s}{100*sum(o[4] for o in order[:order.index((rate,k,g_,c_,nk))+1])/n_dec:>6.1f}%")
    print("\n  read: each row adds that k bucket to the searched set, best rate first.")

    print("\n" + "=" * 78)
    print("E.  S2 -- THE MODEL'S OWN CONFIDENCE, and whether it separates")
    print("=" * 78)
    p = C["ptop1"]
    print(f"  pi(top1) over unforced card decisions: median {np.median(p):.4f}   "
          f"p10 {np.quantile(p,0.1):.4f}   p25 {np.quantile(p,0.25):.4f}   "
          f"p75 {np.quantile(p,0.75):.4f}")
    qs = [0, 0.25, 0.5, 0.75, 1.0]
    edges = [np.quantile(p, q) for q in qs]
    print(f"{'pi(top1) quartile':>26s} {'n':>5s} {'model reg':>10s} "
          f"{'exact reg':>10s} {'gain':>16s}")
    for i in range(4):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == 3 else \
            (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() < 15:
            continue
        g_ = C["r_model"][m] - C["r_exact"][m]
        print(f"  [{edges[i]:.4f},{edges[i+1]:.4f}]{'':>4s} {int(m.sum()):>5d} "
              f"{C['r_model'][m].mean():>10.3f} {C['r_exact'][m].mean():>10.3f} "
              f"{g_.mean():>+8.3f} +- {ci(g_):<6.3f}")

    print("\n" + "=" * 78)
    print("F.  S6 -- DOES CONTEXT SEPARATE WITHIN A k BUCKET?  (k=2 and k=3)")
    print("=" * 78)
    for k in (2, 3):
        mk = C["k"] == k
        if mk.sum() < 40:
            continue
        print(f"  k={k}  (n={int(mk.sum())})")
        for lab, col in (("declarer", "is_decl"), ("on lead", "is_lead"),
                         ("partner winning", "partner_win")):
            for v in (0, 1):
                m = mk & (C[col] == v)
                if m.sum() < 15:
                    continue
                g_ = C["r_model"][m] - C["r_exact"][m]
                print(f"    {lab:<16s}={v}  n={int(m.sum()):>4d}  "
                      f"gain {g_.mean():>+7.3f} +- {ci(g_):<6.3f}   "
                      f"model reg {C['r_model'][m].mean():>6.3f}")

    print("\n" + "=" * 78)
    print("G.  BY TRICK, for completeness")
    print("=" * 78)
    for t in range(8):
        m = C["trick"] == t
        if m.sum() < 15:
            continue
        g_ = C["r_model"][m] - C["r_exact"][m]
        print(f"  trick {t}  n={int(m.sum()):>4d}  k={C['k'][m].mean():.2f}  "
              f"model reg {C['r_model'][m].mean():>6.3f}  "
              f"gain {g_.mean():>+7.3f} +- {ci(g_):<6.3f}  "
              f"{C['c_exact'][m].mean()*1e3:>7.1f} ms/dec")


if __name__ == "__main__":
    main()
