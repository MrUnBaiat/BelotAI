"""
AUDIT v4 / EXP-10 -- H9: is the belief matrix's residual error costing strength?

CONTEXT. Handoff section 5 measured that zeroing the belief matrix costs
1.83 +- 0.47 pts/hand -- more than the model's entire edge over the heuristic. It
is the load-bearing feature. audit_3 (re-run today) shows it is still imperfect
under random play: ordinary column mass 0.983 (p05 0.878), and calibration drifts
in the confident buckets -- predicted 0.798 vs empirical 0.954 at [0.70,0.90).

The suspect is the FIXED 6 rounds of iterative proportional fitting in
build_observation(). IPF converges geometrically but not in 6 passes on every
state, so the feature the policy leans on hardest carries a state-dependent error.

TWO SEPARATE QUESTIONS, MEASURED SEPARATELY -- conflating them would be a mistake:
  Q1 ACCURACY. Do more IPF rounds actually make the belief better calibrated,
     under TRAINED play (not random play, which is what audit_3 measured)?
  Q2 STRENGTH. Does the better belief make the CURRENT policy stronger, with no
     retraining? This can easily come out negative or null: the network was
     trained on the 6-round feature, so sharpening it at test time is a
     distribution shift. A null here does NOT refute Q1; it only says the fix
     needs retraining to pay off.

Paired on identical deals throughout.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import observation as OB
from env import BelotEnv
from model import RecurrentMAPPOModel
from v4_paired_eval import hand_diffs, paired, summarise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DEALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

_ORIG = OB.build_observation


def make_ipf(rounds):
    """Rebuild build_observation with a different IPF budget by monkeypatching the
    loop count. Done by re-implementing only the belief block would duplicate a
    lot of code, so instead patch numpy-level: run the original, then recompute
    the belief slice at the requested precision."""
    def patched(belot, abs_id, match_scores):
        obs, g_obs, mask = _ORIG(belot, abs_id, match_scores)
        obs[339:435] = belief_block(belot, abs_id, rounds)
        return obs, g_obs, mask
    return patched


def belief_block(belot, abs_id, rounds):
    """Exact re-implementation of observation.py's belief block, with a
    configurable number of IPF passes."""
    unseen = np.ones(32, dtype=bool)
    unseen[belot.hands[abs_id]] = False
    unseen[belot.graveyard] = False
    for _, c in belot.current_trick:
        unseen[c] = False
    if belot.phase == "BIDDING" and belot.face_up_card is not None:
        unseen[belot.face_up_card] = False

    others = [(abs_id + 1) % 4, (abs_id + 2) % 4, (abs_id + 3) % 4]
    known_any = (belot.known_cards[others[0]] | belot.known_cards[others[1]]
                 | belot.known_cards[others[2]])
    W = np.zeros((3, 32), dtype=np.float32)
    for i, p in enumerate(others):
        W[i, unseen & ~belot.impossible_cards[p] & ~known_any] = 1.0
    remaining = np.array([len(belot.hands[p]) - belot.known_cards[p].sum()
                          for p in others], dtype=np.float32)

    P = W.copy()
    for _ in range(rounds):
        rs = P.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        P = P * (remaining[:, np.newaxis] / rs)
        cs = P.sum(axis=0, keepdims=True)
        P = P * np.where(cs > 1.0, 1.0 / np.maximum(cs, 1e-8), 1.0)
    P = np.clip(P, 0.0, 1.0)

    out = np.zeros(96, dtype=np.float32)
    for i, p in enumerate(others):
        b = np.zeros(32, dtype=np.float32)
        b[belot.known_cards[p]] = 1.0
        unres = ~belot.known_cards[p]
        b[unres] = P[i, unres]
        out[i * 32:(i + 1) * 32] = b
    return out


@torch.no_grad()
def accuracy(model, n_deals, rounds_list, seed=880_000):
    """Q1: column mass + calibration of each IPF budget, on TRAINED play."""
    from eval import _heuristic_action
    stats = {r: dict(mass=[], pred=[], truth=[]) for r in rounds_list}
    for d in range(n_deals):
        env = BelotEnv(); env.dealer = d % 4
        np.random.seed(seed + d); env.reset(); env.bolts_by_team = [0, 0]
        scores = [0, 0]
        hc = {s: (torch.zeros(1, 1, 512, device=DEVICE),
                  torch.zeros(1, 1, 512, device=DEVICE)) for s in range(4)}
        while not env.done:
            seat = env.current_player
            if env.phase == "PLAYING":
                others = [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]
                truth = np.zeros((3, 32), dtype=np.float32)
                for i, p in enumerate(others):
                    truth[i, env.hands[p]] = 1.0
                for r in rounds_list:
                    B = belief_block(env, seat, r).reshape(3, 32)
                    live = truth.sum(0) > 0
                    stats[r]["mass"].append(B.sum(0)[live])
                    stats[r]["pred"].append(B.reshape(-1))
                    stats[r]["truth"].append(truth.reshape(-1))
            if seat % 2 == 0:
                local, glob, mask = _ORIG(env, seat, scores)
                dist, _, hc[seat] = model(
                    torch.from_numpy(local).unsqueeze(0).to(DEVICE),
                    torch.from_numpy(glob).unsqueeze(0).to(DEVICE), hc[seat],
                    torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(DEVICE),
                    is_sequence=False)
                a = int(dist.probs.argmax(-1).item())
            else:
                a = _heuristic_action(env)
            env.step(a)
    return stats


def main():
    ck = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=512).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    print("=" * 78)
    print("Q1  BELIEF ACCURACY UNDER TRAINED PLAY (audit_3 measured random play)")
    print("=" * 78)
    st = accuracy(model, 250, [6, 24, 100])
    for r in (6, 24, 100):
        mass = np.concatenate(st[r]["mass"])
        pred = np.concatenate(st[r]["pred"]); truth = np.concatenate(st[r]["truth"])
        err = np.abs(pred - truth).mean()
        brier = ((pred - truth) ** 2).mean()
        print(f"\n  IPF rounds = {r}")
        print(f"    live-card column mass : mean {mass.mean():.4f}  "
              f"p05 {np.percentile(mass, 5):.4f}  min {mass.min():.4f}")
        print(f"    mean |p - truth|      : {err:.4f}     Brier {brier:.4f}")
        m = (pred > 0.70) & (pred < 0.90)
        if m.sum():
            print(f"    bucket [0.70,0.90)    : pred {pred[m].mean():.3f}  "
                  f"empirical {truth[m].mean():.3f}  n={m.sum():,}")

    print("\n" + "=" * 78)
    print(f"Q2  DOES IT MOVE STRENGTH WITHOUT RETRAINING?  ({N_DEALS} paired deals)")
    print("=" * 78)
    res = {}
    for r in (6, 24):
        OB.build_observation = _ORIG if r == 6 else make_ipf(r)
        import v4_paired_eval as PE
        PE.build_observation = OB.build_observation
        res[r] = hand_diffs(model, N_DEALS, DEVICE)
        print("  " + summarise(f"IPF rounds = {r}", res[r]))
    OB.build_observation = _ORIG
    print("\n  " + paired("IPF 24 - IPF 6 (same deals)", res[24], res[6]))
    print("\n  Reminder: a null here does not refute Q1. The network was TRAINED on")
    print("  the 6-round feature; sharpening it at test time is a distribution shift,")
    print("  so the accuracy fix has to be evaluated by retraining, not by swapping.")


if __name__ == "__main__":
    main()
