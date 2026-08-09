"""
Full-match evaluation: play complete matches to 101 bile, not isolated hands.

Usage:
  python match_eval.py checkpoints/latest_model.pt checkpoints/best_model.pt 50
  python match_eval.py checkpoints/latest_model.pt heuristic 50
  python match_eval.py checkpoints/latest_model.pt random 50

Team 0 (seats 0&2) = first argument. Team 1 (seats 1&3) = second argument
(a checkpoint path, or the literal strings 'heuristic' / 'random').

Match semantics mirror training (vec_env.finish_and_reset):
  - game points accumulate across hands until a team reaches >= 101
  - env.reset() preserves bolt counters within the match; dealer auto-rotates
    hand to hand (env.step does this at terminal); first dealer rotates per match
  - the RUNNING match score is fed into build_observation, so score-aware
    features are live (exactly as during training, unlike per-hand eval)
  - LSTM state resets at each hand (training semantics)

Reported: match win rate with 95% CI, avg hands per match, per-hand win rate,
avg final scores, bolt counts. NOTE the CI: 50 matches -> +-~14% on the match
win rate. For a decisive latest-vs-best verdict prefer 150-200 matches.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')  # run from the project root
from env import BelotEnv
from observation import build_observation
from model import RecurrentMAPPOModel

HIDDEN = 512
MAX_HANDS_PER_MATCH = 100

NT_POWER = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 3: 6, 7: 7}
T_POWER  = {0: 0, 1: 1, 5: 2, 6: 3, 3: 4, 7: 5, 2: 6, 4: 7}
T_PTS    = {0: 0, 1: 0, 2: 14, 3: 10, 4: 20, 5: 3, 6: 4, 7: 11}
NT_PTS   = {0: 0, 1: 0, 2: 0, 3: 10, 4: 2, 5: 3, 6: 4, 7: 11}


def power(c, tr): return (1, T_POWER[c % 8]) if c // 8 == tr else (0, NT_POWER[c % 8])
def pts(c, tr):   return T_PTS[c % 8] if c // 8 == tr else NT_PTS[c % 8]


def heuristic_action(env, rng):
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


def zero_state(device):
    return (torch.zeros(1, 1, HIDDEN, device=device),
            torch.zeros(1, 1, HIDDEN, device=device))


def load_net(path, device):
    m = RecurrentMAPPOModel().to(device)
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


def make_policy(spec, device, rng):
    """Returns f(env, seat, hc, match_scores) -> (action, new_hc)."""
    if spec == "random":
        def f(env, seat, hc, ms):
            return int(rng.choice(np.flatnonzero(env.get_legal_actions()))), hc
        return f
    if spec == "heuristic":
        def f(env, seat, hc, ms):
            return heuristic_action(env, rng), hc
        return f
    net = load_net(spec, device)

    @torch.no_grad()
    def f(env, seat, hc, ms):
        local, glob, mask = build_observation(env, seat, ms)
        lt = torch.from_numpy(local).unsqueeze(0).to(device)
        gt = torch.from_numpy(glob).unsqueeze(0).to(device)
        mt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
        dist, _, new_hc = net(lt, gt, hc, mt, is_sequence=False)
        return int(dist.probs.argmax(dim=-1).item()), new_hc
    return f


def play_match(pol0, pol1, first_dealer, device):
    env = BelotEnv()
    env.dealer = first_dealer
    env.reset()                       # ctor already reset once; this applies the dealer
    ms = [0, 0]
    hands = hand_wins0 = 0
    bolts = [0, 0]
    while ms[0] < 101 and ms[1] < 101 and hands < MAX_HANDS_PER_MATCH:
        hc = {s: zero_state(device) for s in range(4)}
        info = {}
        while not env.done:
            s = env.current_player
            pol = pol0 if s % 2 == 0 else pol1
            a, hc[s] = pol(env, s, hc[s], ms)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        if env.raw_points_by_team and env.declaring_team is not None \
                and env.raw_points_by_team[env.declaring_team] <= 80:
            bolts[env.declaring_team] += 1
        ms[0] += gp[0]; ms[1] += gp[1]
        hands += 1
        hand_wins0 += gp[0] > gp[1]
        if ms[0] < 101 and ms[1] < 101:
            env.reset()               # keeps bolts, dealer already rotated by step()
    winner = 0 if ms[0] > ms[1] else (1 if ms[1] > ms[0] else -1)
    return winner, ms, hands, hand_wins0, bolts


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest_model.pt"
    b = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/best_model.pt"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    np.random.seed(0)
    pol0 = make_policy(a, device, rng)
    pol1 = make_policy(b, device, rng)

    w = t = 0
    all_hands, all_hw, finals, bolt_tot = [], [], [], [0, 0]
    for m in range(n):
        winner, ms, hands, hw0, bolts = play_match(pol0, pol1, m % 4, device)
        w += winner == 0; t += winner == -1
        all_hands.append(hands); all_hw.append(hw0 / hands); finals.append(ms)
        bolt_tot[0] += bolts[0]; bolt_tot[1] += bolts[1]
        print(f"match {m:3d} | {'WIN ' if winner == 0 else ('tie ' if winner == -1 else 'loss')} "
              f"{ms[0]:3d}-{ms[1]:3d} in {hands:2d} hands", flush=True)

    p = w / n
    ci = 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
    finals = np.array(finals)
    print(f"\nTEAM0 = {a}\nTEAM1 = {b}")
    print(f"match win rate : {p:.1%} +-{ci:.1%}   (ties {t}/{n})")
    print(f"hands per match: {np.mean(all_hands):.1f} | per-hand win rate {np.mean(all_hw):.1%}")
    print(f"avg final score: {finals[:,0].mean():.1f} - {finals[:,1].mean():.1f}")
    print(f"bolts conceded as declarer, total: team0 {bolt_tot[0]} | team1 {bolt_tot[1]}")
    if ci > 0.10:
        print(f"NOTE: +-{ci:.0%} CI is wide; rerun with 150-200 matches for a decisive verdict.")


if __name__ == "__main__":
    main()