"""
AUDIT 3 -- Belief matrix (obs feature 10, local dims [339:435]).

During PLAYING every unseen card is held by exactly one of the 3 opponents, so
for each unseen card the belief mass summed over the 3 opponent rows should be
exactly 1.0, and the belief should be probabilistically calibrated
(mean predicted p in a bucket == empirical holding frequency).

Two specific defects are probed:
  D1: a card KNOWN to be held by opponent X (the face-up card taken at bidding)
      keeps receiving mass from the other two rows, because `unseen` never
      excludes known cards -> column mass ~2 instead of 1.
  D2: single-pass row rescaling followed by np.clip distorts the distribution
      (columns no longer sum to 1 even for ordinary cards).
"""
import sys
import numpy as np

sys.path.insert(0, '.')  # run from the project root
from env import BelotEnv
from observation import build_observation

BELIEF_OFF = 339  # 32+32+5+5+3+108+6+4+144


def belief_rows(obs):
    return obs[BELIEF_OFF:BELIEF_OFF + 96].reshape(3, 32)


def main():
    np.random.seed(7)
    known_card_masses = []       # D1: column mass of the face-up card while known-held
    normal_masses = []           # D2: column mass of ordinary unseen cards
    cal_pred, cal_true = [], []  # calibration pairs (predicted p, actually holds)

    games = 0
    while games < 400:
        env = BelotEnv()
        recip = None
        while not env.done:
            if env.phase == "PLAYING" and recip is None:
                recip = env.declarer if env.bidding_round == 1 else env.dealer
            if env.phase == "PLAYING":
                me = env.current_player
                obs, gobs, mask = build_observation(env, me, [0, 0])
                W = belief_rows(obs)
                others = [(me + 1) % 4, (me + 2) % 4, (me + 3) % 4]

                seen = set(env.hands[me]) | set(env.graveyard) | \
                       {c for _, c in env.current_trick}
                for c in range(32):
                    if c in seen:
                        continue
                    col = W[:, c].sum()
                    holder_known = (recip is not None and recip != me
                                    and env.known_cards[recip, c])
                    if holder_known:
                        known_card_masses.append(col)
                    else:
                        normal_masses.append(col)
                    for i, p in enumerate(others):
                        cal_pred.append(W[i, c])
                        cal_true.append(float(c in env.hands[p]))

            legal = np.flatnonzero(env.get_legal_actions())
            env.step(int(np.random.choice(legal)))
        games += 1

    known = np.array(known_card_masses); normal = np.array(normal_masses)
    pred = np.array(cal_pred); true = np.array(cal_true)

    print(f"sampled states: {len(normal)+len(known)} card-columns over {games} games\n")
    print("D1  column mass for the KNOWN-held face-up card (ideal = 1.000):")
    print(f"    n={len(known)}  mean={known.mean():.3f}  max={known.max():.3f}  "
          f"frac>1.5: {(known>1.5).mean():.2%}")
    print("D2  column mass for ordinary unseen cards (ideal = 1.000):")
    print(f"    n={len(normal)}  mean={normal.mean():.3f}  "
          f"p05={np.percentile(normal,5):.3f}  p95={np.percentile(normal,95):.3f}\n")

    print("Calibration (predicted holding prob vs empirical frequency):")
    print(f"{'bucket':>12} {'n':>9} {'mean pred':>10} {'empirical':>10}")
    edges = [0.0, 0.101, 0.3, 0.5, 0.7, 0.9, 0.999, 1.001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum() == 0:
            continue
        print(f"[{lo:.2f},{hi:.2f}) {m.sum():>9} {pred[m].mean():>10.3f} {true[m].mean():>10.3f}")

    bad = known.mean() > 1.3
    print("\nRESULT:", "CONFIRMED BUG -- known-held cards are double/triple counted "
          "across opponents; the actor is fed contradictory card-location beliefs."
          if bad else "belief mass looks consistent")


if __name__ == "__main__":
    main()
