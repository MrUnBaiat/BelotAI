"""
Does the LSTM memory actually matter? Does the model read its inference features?

Usage:  python probe_lstm.py checkpoints/latest_model.pt [n_hands]

Design
------
Six conditions, each played over the SAME deals (paired experiment):
  baseline        normal state carry, full observation
  no-lstm-memory  hidden state zeroed before EVERY decision (policy goes reactive)
  no-belief       obs[339:435] = 0   (belief matrix hidden)
  no-graveyard    obs[481:513] = 0   (graveyard hidden)
  no-last-trick   obs[195:339] = 0   (last-trick block hidden)
  no-lstm+belief  both ablations together (tests whether one compensates the other)

Pairing: the model plays greedy and the heuristic opponent is deterministic, so
the only RNG consumption is the deal permutation inside env.reset(). Re-seeding
numpy identically per condition therefore reproduces the exact same deal
sequence, and per-hand point-diff differences can be compared PAIRED, which
kills most of the variance. The script verifies deal identity and falls back to
unpaired stats if anything desyncs.

Also collected on the baseline trajectories: at every model decision, the
policy is re-run under each ablation on the same state, logging KL(base||abl)
and top-1 action agreement, split by phase and trick number. This measures how
much each information source changes DECISIONS, independent of outcome noise.

Interpretation
--------------
  no-lstm-memory ~ baseline  ->  the LSTM carries nothing your features don't
                                 already say; recurrence is not the bottleneck
  no-lstm-memory << baseline ->  memory matters; consider whether rollout/train
                                 state handling or capacity limits it
  no-belief ~ baseline       ->  the net IGNORES the belief matrix (expected if
                                 it learned the features are contradictory --
                                 see audit_3); fixing beliefs + finetune should
                                 then unlock card-inference play
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')  # run from the project root
from env import BelotEnv
from observation import build_observation
from model import RecurrentMAPPOModel

HIDDEN = 512

NT_POWER = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 3: 6, 7: 7}
T_POWER  = {0: 0, 1: 1, 5: 2, 6: 3, 3: 4, 7: 5, 2: 6, 4: 7}
T_PTS    = {0: 0, 1: 0, 2: 14, 3: 10, 4: 20, 5: 3, 6: 4, 7: 11}
NT_PTS   = {0: 0, 1: 0, 2: 0, 3: 10, 4: 2, 5: 3, 6: 4, 7: 11}


def power(c, tr): return (1, T_POWER[c % 8]) if c // 8 == tr else (0, NT_POWER[c % 8])
def pts(c, tr):   return T_PTS[c % 8] if c // 8 == tr else NT_PTS[c % 8]


def heuristic_action(env):
    """Deterministic greedy opponent (same as probe_checkpoint.py)."""
    legal = np.flatnonzero(env.get_legal_actions())
    hand = env.hands[env.current_player]
    if env.phase == "BIDDING":
        def strength(s, extra=None):
            cards = [c for c in hand if c // 8 == s] + \
                    ([extra] if extra is not None and extra // 8 == s else [])
            return len(cards), sum(T_PTS[c % 8] for c in cards)
        if 33 in legal:
            n, p = strength(env.face_up_suit, extra=env.face_up_card)
            return 33 if (n >= 3 and p >= 20) or n >= 4 else 32
        suits = [a - 34 for a in legal if a >= 34]
        best = max(suits, key=lambda s: strength(s))
        n, p = strength(best)
        return 32 if (32 in legal and not (n >= 3 and p >= 20)) else 34 + best
    tr, trick, cards = env.trump, env.current_trick, list(legal)
    if not trick:
        return max(cards, key=lambda c: power(c, tr)[1] - (3 if c // 8 == tr and T_POWER[c % 8] < 6 else 0))
    cur = max(power(c, tr) for _, c in trick)
    cur_w = max(trick, key=lambda pc: power(pc[1], tr))[0]
    partner = (cur_w % 2) == (env.current_player % 2)
    winners = [c for c in cards if power(c, tr) > cur]
    if partner and len(trick) >= 2:
        return (max(cards, key=lambda c: (pts(c, tr), -power(c, tr)[1])) if len(trick) == 3
                else min(cards, key=lambda c: (pts(c, tr), power(c, tr)[1])))
    if winners:
        return min(winners, key=lambda c: power(c, tr)[1]) if len(trick) == 3 \
            else max(winners, key=lambda c: power(c, tr)[1])
    return min(cards, key=lambda c: (pts(c, tr), power(c, tr)[1]))


BEL, GRV, LTK = (339, 435), (481, 513), (195, 339)
CONDITIONS = [
    ("baseline",        dict(zero_lstm=False, zero=[])),
    ("no-lstm-memory",  dict(zero_lstm=True,  zero=[])),
    ("no-belief",       dict(zero_lstm=False, zero=[BEL])),
    ("no-graveyard",    dict(zero_lstm=False, zero=[GRV])),
    ("no-last-trick",   dict(zero_lstm=False, zero=[LTK])),
    ("no-lstm+belief",  dict(zero_lstm=True,  zero=[BEL])),
]


def zero_state(device):
    return (torch.zeros(1, 1, HIDDEN, device=device),
            torch.zeros(1, 1, HIDDEN, device=device))


@torch.no_grad()
def model_dist(model, env, seat, hc, device, zero_ranges=()):
    local, glob, mask = build_observation(env, seat, [0, 0])
    for a, b in zero_ranges:
        local[a:b] = 0.0
    lt = torch.from_numpy(local).unsqueeze(0).to(device)
    gt = torch.from_numpy(glob).unsqueeze(0).to(device)
    mt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
    dist, _, new_hc = model(lt, gt, hc, mt, is_sequence=False)
    return dist, new_hc


@torch.no_grad()
def play_condition(model, cond, n_hands, device, seed):
    np.random.seed(seed)
    diffs, deal_sigs = [], []
    for g in range(n_hands):
        env = BelotEnv(); env.dealer = g % 4; env.reset()
        if g < 5:
            deal_sigs.append(tuple(sorted((p, c) for p in range(4) for c in env.hands[p])))
        hc = {s: zero_state(device) for s in (0, 2)}
        info = {}
        while not env.done:
            s = env.current_player
            if s % 2 == 0:
                state = zero_state(device) if cond["zero_lstm"] else hc[s]
                dist, new_hc = model_dist(model, env, s, state, device, cond["zero"])
                if not cond["zero_lstm"]:
                    hc[s] = new_hc
                a = int(dist.probs.argmax(dim=-1).item())
            else:
                a = heuristic_action(env)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        diffs.append(gp[0] - gp[1])
    return np.array(diffs, dtype=np.float64), deal_sigs


@torch.no_grad()
def decision_divergence(model, n_hands, device, seed):
    """On baseline trajectories, compare the decision distribution under each
    ablation at the very same states."""
    np.random.seed(seed)
    ablations = [(name, c) for name, c in CONDITIONS if name != "baseline"]
    stats = {name: {"kl": [], "agree": [], "trick": []} for name, _ in ablations}
    for g in range(n_hands):
        env = BelotEnv(); env.dealer = g % 4; env.reset()
        hc = {s: zero_state(device) for s in (0, 2)}
        while not env.done:
            s = env.current_player
            if s % 2 == 0:
                base_dist, new_hc = model_dist(model, env, s, hc[s], device)
                p = base_dist.probs.squeeze(0)
                a_base = int(p.argmax().item())
                if env.phase == "PLAYING":
                    for name, c in ablations:
                        st = zero_state(device) if c["zero_lstm"] else hc[s]
                        d2, _ = model_dist(model, env, s, st, device, c["zero"])
                        q = d2.probs.squeeze(0)
                        kl = float((p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum())
                        stats[name]["kl"].append(kl)
                        stats[name]["agree"].append(int(q.argmax().item()) == a_base)
                        stats[name]["trick"].append(env.tricks_played)
                hc[s] = new_hc
                a = a_base
            else:
                a = heuristic_action(env)
            env.step(a)
    return stats


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest_model.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentMAPPOModel().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"checkpoint: {path} | {n} paired hands vs deterministic heuristic\n")

    results, base = {}, None
    for name, cond in CONDITIONS:
        diffs, sigs = play_condition(model, cond, n, device, seed=123)
        results[name] = (diffs, sigs)
        if name == "baseline":
            base = diffs
            print(f"{name:15s} avg point diff {diffs.mean():+6.2f} | "
                  f"win {np.mean(diffs > 0):5.1%} tie {np.mean(diffs == 0):.1%}")
        else:
            paired = sigs == results["baseline"][1]
            d = diffs - base
            se = d.std(ddof=1) / np.sqrt(len(d))
            tag = "PAIRED" if paired else "UNPAIRED (RNG desync!)"
            print(f"{name:15s} avg point diff {diffs.mean():+6.2f} | "
                  f"win {np.mean(diffs > 0):5.1%} | "
                  f"delta vs baseline {d.mean():+5.2f} +-{1.96*se:.2f}  [{tag}]")

    print("\nPer-decision divergence from baseline policy (PLAYING phase only):")
    stats = decision_divergence(model, min(n, 400), device, seed=123)
    print(f"{'ablation':15s} {'mean KL':>8} {'agree':>7}   agree by trick 0..7")
    for name in stats:
        kl = np.array(stats[name]["kl"]); ag = np.array(stats[name]["agree"], dtype=float)
        tr = np.array(stats[name]["trick"])
        by_trick = " ".join(f"{ag[tr == t].mean():.2f}" if (tr == t).any() else "  - "
                            for t in range(8))
        print(f"{name:15s} {kl.mean():8.4f} {ag.mean():6.1%}   {by_trick}")

    print("\nHow to read this:")
    print("  * 'no-lstm-memory' delta ~ 0 and agreement ~ 1.0  -> recurrence carries")
    print("    nothing beyond your engineered features; the LSTM is NOT the lever.")
    print("  * 'no-belief' delta ~ 0 -> the net learned to ignore the (inconsistent)")
    print("    belief matrix; fixing it (observation_fixed.py) + finetuning is the play.")
    print("  * agreement dropping in later tricks for 'no-lstm-memory' would be the")
    print("    signature of genuine memory use (endgame card counting).")


if __name__ == "__main__":
    main()