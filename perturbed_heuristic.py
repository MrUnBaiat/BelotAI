"""
A PERTURBED greedy heuristic -- a training opponent that is not the yardstick.

WHY THIS EXISTS. EXP-A measured a clearly resolvable gradient against the greedy
heuristic (ratio deviation 39.3%, cosine +0.00430 +- 0.00111, B_simple 29,660)
while the self-play regime is indistinguishable from pure noise (3.7%,
+0.00029 +- 0.00107, B_simple 437,395). So a real improvement direction exists
against non-descendant opponents -- which argues for putting a scripted opponent
in the training pool.

But the obvious choice, the exact greedy heuristic, cannot go in the pool:

  1. It is the project's cheap absolute yardstick (`Eval/HandDiffVsHeuristic`).
  2. More seriously, `pimc.py:166` and `pimc_model.py:143` delegate BIDDING to
     `_heuristic_action`. So PIMC -- the intended replacement yardstick -- is
     itself partly the greedy heuristic. Training against the exact heuristic
     would let the model learn to exploit heuristic BIDDING, and that exploitation
     would transfer straight into the PIMC yardstick. Neither yardstick would
     survive.

So the pool gets a PERTURBED variant instead, and both the exact heuristic and
PIMC stay held out.

THE PERTURBATION is confined to bidding, because bidding is exactly the phase PIMC
shares with the heuristic and therefore the phase whose exploitation would
contaminate the yardstick:
  * looser accept threshold  (n>=3 and p>=20)  ->  (n>=3 and p>=14)
  * looser round-2 naming threshold, same change
  * suit choice breaks ties by trump POINTS rather than by (length, points)
Card play is left identical to the greedy heuristic: it is already a reasonable
sparring policy, and changing it would make the opponent weaker for no benefit.

Verify with v7_02_perturbed_check.py that this policy actually differs from the
exact heuristic by a material margin -- a perturbation too small to change
behaviour would just be the exact heuristic wearing a hat, and would re-create the
contamination it exists to avoid.
"""
import numpy as np

from eval import _NT_POWER, _T_POWER, _T_PTS, _pt, _pw

# MEASURED CALIBRATION. A threshold change alone gave only 7.6% bidding divergence
# from the exact heuristic (v7_02, first pass). That is too mild for the job: the
# contamination being defended against is the model learning to exploit HEURISTIC
# BIDDING, and an opponent whose bidding is 92.4% identical to the heuristic's
# leaks almost all of that exploitation into the PIMC yardstick anyway. So the
# threshold change is combined with an epsilon-random bid.
#
# The RNG is PRIVATE (never numpy's global stream), for the same reason pimc.py's
# is: the global stream is what makes re-seeded deals reproducible across paired
# conditions, and this module must not touch it.
BID_EPS = 0.20
_RNG = np.random.default_rng(20260813)


def perturbed_heuristic_action(belot):
    """Greedy heuristic with a looser, differently-tie-broken, epsilon-random bid.
    Card play is identical to the exact heuristic by construction."""
    legal = np.flatnonzero(belot.get_legal_actions())
    hand = belot.hands[belot.current_player]

    if belot.phase == "BIDDING":
        if _RNG.random() < BID_EPS:
            return int(_RNG.choice(legal))
        def strength(s, extra=None):
            cards = [c for c in hand if c // 8 == s] + \
                    ([extra] if extra is not None and extra // 8 == s else [])
            return len(cards), sum(_T_PTS[c % 8] for c in cards)

        if 33 in legal:
            n, p = strength(belot.face_up_suit, extra=belot.face_up_card)
            # PERTURBATION: accept on 14 trump points instead of 20 -> bids more often
            return 33 if (n >= 3 and p >= 14) or n >= 4 else 32
        suits = [a - 34 for a in legal if a >= 34]
        # PERTURBATION: rank candidate suits by POINTS first, not (length, points)
        best = max(suits, key=lambda s: (strength(s)[1], strength(s)[0]))
        n, p = strength(best)
        return 32 if (32 in legal and not (n >= 3 and p >= 14)) else 34 + best

    # --- card play: identical to the greedy heuristic ---
    tr, trick, cards = belot.trump, belot.current_trick, list(legal)
    if not trick:
        return max(cards, key=lambda c: _pw(c, tr)[1]
                   - (3 if c // 8 == tr and _T_POWER[c % 8] < 6 else 0))
    cur = max(_pw(c, tr) for _, c in trick)
    cur_w = max(trick, key=lambda pc: _pw(pc[1], tr))[0]
    partner = (cur_w % 2) == (belot.current_player % 2)
    winners = [c for c in cards if _pw(c, tr) > cur]
    if partner and len(trick) >= 2:
        return (max(cards, key=lambda c: (_pt(c, tr), -_pw(c, tr)[1])) if len(trick) == 3
                else min(cards, key=lambda c: (_pt(c, tr), _pw(c, tr)[1])))
    if winners:
        return min(winners, key=lambda c: _pw(c, tr)[1]) if len(trick) == 3 \
            else max(winners, key=lambda c: _pw(c, tr)[1])
    return min(cards, key=lambda c: (_pt(c, tr), _pw(c, tr)[1]))
