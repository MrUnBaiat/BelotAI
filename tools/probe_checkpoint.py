"""
Behaviour probe for a trained checkpoint.

Usage:
  python probe_checkpoint.py MODEL [N_HANDS] [REFERENCE]

  MODEL      checkpoint to probe          (default checkpoints/latest_model.pt)
  N_HANDS    hands per opponent           (default 2000)
  REFERENCE  frozen opponent checkpoint   (default checkpoints/reference_model.pt,
                                           falling back to checkpoints/best_model.pt)

Opponents:
  1. the frozen REFERENCE network -- "am I better than my own ancestor?"
  2. the fixed greedy heuristic  -- an absolute yardstick that never changes and
     never saturates, so numbers stay comparable across every run you ever do

Both are played as ISOLATED hands with match_scores=[0,0] and fresh bolt
counters, which is a different measurement from eval.py's matches-to-101 (there,
bolts persist and the running score is live in the observation). Isolated hands
give a lower-variance read on raw hand strength; matches give the number that
actually decides a game. To keep the two directly comparable this script also
reports the IMPLIED match win rate, obtained by resampling the observed per-hand
game-point pairs into matches to 101.

Reported per opponent:
  * per-hand win / tie / loss, avg point diff with 95% CI
  * implied match win rate (so it lines up with the training-loop eval)
  * decomposition by declaring side: share of hands, win rate, bolt rate
  * round-1 accept rate vs trump-suit length, and per-phase policy entropy

Caveat on the reference number: if REFERENCE is also in the training frozen pool,
the model is directly incentivised to exploit it, so a rising score against it
does not by itself prove absolute improvement. Read it alongside the heuristic
column -- that is what separates "getting stronger" from "learning this one
opponent".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from belot.env import BelotEnv
from belot.observation import build_observation
from belot.model import RecurrentMAPPOModel

HIDDEN = 512
NT_POWER = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 3: 6, 7: 7}
T_POWER  = {0: 0, 1: 1, 5: 2, 6: 3, 3: 4, 7: 5, 2: 6, 4: 7}
T_PTS    = {0: 0, 1: 0, 2: 14, 3: 10, 4: 20, 5: 3, 6: 4, 7: 11}
NT_PTS   = {0: 0, 1: 0, 2: 0, 3: 10, 4: 2, 5: 3, 6: 4, 7: 11}


def power(c, tr): return (1, T_POWER[c % 8]) if c // 8 == tr else (0, NT_POWER[c % 8])
def pts(c, tr):   return T_PTS[c % 8] if c // 8 == tr else NT_PTS[c % 8]


def zero_state(device):
    return (torch.zeros(1, 1, HIDDEN, device=device),
            torch.zeros(1, 1, HIDDEN, device=device))


# --------------------------------------------------------------------------
# Opponents: each exposes act(env, seat) and new_hand()
# --------------------------------------------------------------------------
class RandomOpponent:
    name = "uniform-random"

    def __init__(self, rng): self.rng = rng
    def new_hand(self): pass
    def act(self, env, seat):
        return int(self.rng.choice(np.flatnonzero(env.get_legal_actions())))


class HeuristicOpponent:
    name = "greedy heuristic"

    def new_hand(self): pass

    def act(self, env, seat):
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
            return max(cards, key=lambda c: power(c, tr)[1] -
                       (3 if c // 8 == tr and T_POWER[c % 8] < 6 else 0))
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


class NetOpponent:
    """A frozen checkpoint, with its own per-seat LSTM state."""

    def __init__(self, net, device, label="frozen reference"):
        self.net, self.device, self.name = net, device, label
        self.hc = {}

    def new_hand(self):
        self.hc = {s: zero_state(self.device) for s in range(4)}

    @torch.no_grad()
    def act(self, env, seat):
        local, glob, mask = build_observation(env, seat, [0, 0])
        lt = torch.from_numpy(local).unsqueeze(0).to(self.device)
        gt = torch.from_numpy(glob).unsqueeze(0).to(self.device)
        mt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(self.device)
        dist, _, self.hc[seat] = self.net(lt, gt, self.hc[seat], mt, is_sequence=False)
        return int(dist.probs.argmax(dim=-1).item())


# --------------------------------------------------------------------------
def implied_match_win_rate(gp_pairs, n_sims=20000, seed=0):
    """
    Resample the OBSERVED per-hand (gp0, gp1) outcomes into matches to 101.
    Uses the empirical distribution, so 16/6/26-point hands and -10 penalties are
    represented at their true frequency rather than assumed Gaussian.
    """
    rng = np.random.default_rng(seed)
    gp = np.asarray(gp_pairs, dtype=np.float64)
    if len(gp) == 0:
        return float('nan'), float('nan'), float('nan')
    wins = ties = 0
    lens = []
    for _ in range(n_sims):
        s0 = s1 = 0.0
        h = 0
        while s0 < 101 and s1 < 101 and h < 100:
            a, b = gp[rng.integers(len(gp))]
            s0 += a; s1 += b; h += 1
        wins += s0 > s1; ties += s0 == s1; lens.append(h)
    return wins / n_sims, ties / n_sims, float(np.mean(lens))


@torch.no_grad()
def run(model, opponent, n_games, device):
    wins = ties = 0
    diffs, gp_pairs = [], []
    decomp = {'we': [0, 0, 0], 'they': [0, 0, 0]}     # games, wins, declarer-bolts
    accept = {}
    ent = {'BIDDING': [], 'PLAYING': []}

    for g in range(n_games):
        env = BelotEnv(); env.dealer = g % 4; env.reset()
        hc = {s: zero_state(device) for s in range(4)}
        opponent.new_hand()
        info = {}
        while not env.done:
            s = env.current_player
            if s % 2 == 0:
                local, glob, mask = build_observation(env, s, [0, 0])
                lt = torch.from_numpy(local).unsqueeze(0).to(device)
                gt = torch.from_numpy(glob).unsqueeze(0).to(device)
                mt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
                dist, _, hc[s] = model(lt, gt, hc[s], mt, is_sequence=False)
                ent[env.phase].append(dist.entropy().item())
                n_tr = None
                if env.phase == "BIDDING" and mask[33]:
                    n_tr = sum(1 for c in env.hands[s] + [env.face_up_card]
                               if c // 8 == env.face_up_suit)
                    accept.setdefault(n_tr, [0, 0])[1] += 1
                a = int(dist.probs.argmax(dim=-1).item())
                if n_tr is not None and a == 33:
                    accept[n_tr][0] += 1
            else:
                a = opponent.act(env, s)
            _, _, _, info = env.step(a)

        gp = info["game_points"]
        gp_pairs.append((gp[0], gp[1]))
        diffs.append(gp[0] - gp[1])
        wins += gp[0] > gp[1]; ties += gp[0] == gp[1]
        key = 'we' if env.declaring_team == 0 else 'they'
        decomp[key][0] += 1
        decomp[key][1] += gp[0] > gp[1]
        decomp[key][2] += env.raw_points_by_team[env.declaring_team] <= 80

    n = n_games
    p = wins / n
    ci = 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
    d_ci = 1.96 * np.std(diffs) / np.sqrt(n)
    mwr, mtie, mlen = implied_match_win_rate(gp_pairs)

    print(f"\nvs {opponent.name}:")
    print(f"  per-hand : win {p:.1%} +-{ci:.1%} | tie {ties/n:.1%} | loss {1-p-ties/n:.1%} "
          f"| avg point diff {np.mean(diffs):+.2f} +-{d_ci:.2f}")
    print(f"  implied match win rate (to 101): {mwr:.1%} "
          f"(tie {mtie:.1%}, {mlen:.1f} hands/match) "
          f"<- compare with the training-loop eval")
    for k, label in [('we', 'model team declares'), ('they', 'opponent declares ')]:
        gN, gW, gB = decomp[k]
        if gN:
            print(f"    {label}: {gN/n:5.1%} of hands | win {gW/gN:5.1%} | declarer bolts {gB/gN:5.1%}")
    if accept:
        print("  round-1 accept rate by trump-suit length (hand + face-up):")
        for k in sorted(accept):
            a0, a1 = accept[k]
            print(f"    {k} trumps: {a0/max(a1,1):5.1%}  (n={a1})")
    for ph in ('BIDDING', 'PLAYING'):
        if ent[ph]:
            print(f"  mean policy entropy [{ph}]: {np.mean(ent[ph]):.3f}")
    return {"hand_win": p, "hand_diff": float(np.mean(diffs)), "match_win": mwr}


def load_net(path, device):
    net = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(device)
    ck = torch.load(path, map_location=device)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net, ck.get("epoch", "?")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest_model.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    if len(sys.argv) > 3:
        ref_path = sys.argv[3]
    else:
        ref_path = "checkpoints/reference_model.pt"
        if not os.path.exists(ref_path):
            ref_path = "checkpoints/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_net(path, device)
    print(f"checkpoint: {path} (epoch {epoch}) | {n} hands per opponent, dealer rotated")

    if os.path.exists(ref_path):
        ref_net, ref_epoch = load_net(ref_path, device)
        if os.path.samefile(ref_path, path):
            print(f"WARNING: reference is the same file as the probed model -- "
                  f"expect ~50% by construction")
        opp = NetOpponent(ref_net, device,
                          label=f"frozen reference [{ref_path}, epoch {ref_epoch}]")
        run(model, opp, n, device)
    else:
        print(f"\n(no reference checkpoint at {ref_path}; skipping the head-to-head)")

    run(model, HeuristicOpponent(), n, device)


if __name__ == "__main__":
    main()