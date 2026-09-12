# Results

Every figure below is a measurement with a 95% confidence interval and its sample
size. Where a number is a bound rather than a point estimate it says so. Where a
result is negative it is reported as a result, because most of them are.

The measuring instrument is described in §1; it matters more than any individual
number, because Belot's per-hand outcome swings by tens of game points on card luck
alone and a naive comparison of two decent policies is mostly noise.

---

## 1. How strength is measured

**pts/hand** — the mean per-hand game-point difference. A hand is worth 16 game
points to the winning team, so this is the natural scale. Win rate is *not* used:
the declaring team wins whenever it avoids a bolt, so win rate saturates and cannot
resolve small effects.

**Swap-paired ("duplicate bridge") deals.** Every deal is played twice with the seat
pairs exchanged, and the edge is `(dA − dB) / 2`. Two identical policies then cancel
**deal by deal** rather than on average, so the identical-policy control returns
exactly `0.000` with `max|edge|` exactly `0` — and if it does not, something is
leaking and no number from that run counts. Measured variance reduction on genuinely
different policies: **1.9× to 2.3× in standard deviation**, i.e. 3.7×–5.3× in
variance.

Run it yourself: `python tools/check_swap_control.py`.

**Stochastic opponents are reseeded per deal.** The search's determinization draw is
not cancelled by pairing. Letting one generator run continuously through both arms of
an evaluation once left **67% of a headline interval as uncancelled search noise**.

**Per-hand is not per-match.** A match to 101 is about 11.5 hands, so a small
per-hand edge compounds: +0.5 → 56% match win, +1.0 → 62%, +1.9 → 72%. Predicted
72.5% against a measured 72.0%.

---

## 2. Reference points

| | pts/hand |
|---|---|
| greedy heuristic vs uniform random | +5.56 |
| trained network vs uniform random | +5.79 ± 0.32 |
| trained network vs greedy heuristic | +1.9 to +2.3 |
| rollout PIMC (D=32) vs greedy heuristic | +3.015 ± 0.371 |
| rollout PIMC (D=32) vs the network | +0.752 ± 0.435 |

Reproduce the first row with `python tools/heuristic_baseline.py`.

---

## 3. The deliverable

**Model bidding + model card play at tricks 0–2 + exact-solve PIMC (D=128) from
trick 3.**

| | pts/hand | n |
|---|---|---|
| vs the bare network, **D=128** | **+1.270 ± 0.197** | 1000 |
| vs the bare network, D=8 | +0.835 ± 0.219 | 1000 |
| vs the bare network, D=8 (earlier run) | +0.974 ± 0.175 | 1500 |
| vs a held-out frozen network, D=8 | +0.936 ± 0.243 | 500 |

All with the identical-policy control at exactly zero.

**Scored in the simulator's game, which has no melds.** belot.md counts declared
combinations in the bolt test and in the stakes. Measured on the same deals scored both
ways, the conversion above has an edge over the network about 0.2 pts/hand smaller under
the platform's rule (−0.200 ± 0.171 at D=32, −0.172 ± 0.211 at D=128, n=800 and 600),
and the meld-aware conversion now deployed online does not lose it (+0.256 ± 0.181,
+0.483 ± 0.219). The offline player is unchanged — with no melds the two conversions are
identical by construction — so this table stands as a measurement of the simulator's
game. `research/v10_search/FINDINGS.md` §13 has the rule, its validation and the pricing.
Head to head, the corrected conversion is worth **+0.174 ± 0.122 pts/hand** over the
deployed one (n=2,000, three independent deal ranges, all six controls exactly zero) —
about a sixth of what the whole search from trick 3 is worth. The simulator can now score
the platform's game itself: `BelotEnv.melds = True`, or `swap_edges(..., melds=True)`
(`belot/melds.py`). Future offline numbers should use it.

**The determinization ladder was re-measured in that game** (`research/v10_search` x22,
n=600): the D=128-over-D=8 gain transfers (the meld-free-scoring shrinkage is
−0.013 ± 0.216), but the curve saturates by D=32 — D32 − D8 = +0.303 ± 0.264 against
D128 − D32 = +0.031 ± 0.264, at 3.55 vs 13.41 s per paired deal.

Every parameter is a measurement rather than a preference:

| choice | why |
|---|---|
| search starts at **trick 3** | extending it into tricks 0–2 measured **−0.348 ± 0.309** — significantly worse. At trick 0 there are 24 unseen cards, so one determinization's verdict is nearly uninformative and searching adds variance, not skill. |
| **D = 128** | worth **+0.435 ± 0.208** over D=8, swap-paired on identical deals and **replicated** on deals no earlier run had touched. See §3.1. |
| **exact** solve, not rollout | worth **+0.360 ± 0.204** from trick 3 on. The same upgrade at trick 2 is worth **+0.015 ± 0.339** — nothing. |
| bidding left to the **network** | see §6. |

**The base is worth the entire effect.** The identical search on a greedy-heuristic
base scores **−1.188**; on the network base, **+0.614**. The network's job is to hand
good positions to the search.

### 3.1 D = 128, and a claim this file used to make

This file previously said *"D=16 measured as equal strength at equal cost, so D=8 is
the cheaper of two equals."* That null is real. The generalisation drawn from it —
that the determinization axis is flat — was not measured, and is wrong.

`tools/dd_depth_sweep.py` plays both arms in **one interleaved loop**. Because
`swap_eval._play` seeds each deal from a fixed `BASE_SEED`, deal *d* is the same deal
in every run this project has done, so the arms are paired on **deals as well as
seats** and the per-deal difference cancels card luck a second time.

| run | deals | D=8 | D=128 | paired D128 − D8 |
|---|---|---|---|---|
| 1 | 0–499 | +0.992 ± 0.307 | +1.370 ± 0.287 | **+0.378 ± 0.291** |
| 2 (replication) | 500–999 | +0.678 ± 0.311 | +1.170 ± 0.271 | **+0.492 ± 0.297** |
| pooled | 1000 | +0.835 ± 0.219 | +1.270 ± 0.197 | **+0.435 ± 0.208** |

Run 2 used deals no measurement in this project had ever played. The two runs agree
(run2 − run1 = +0.114, z = +0.54). Pooled: bootstrap **[+0.226, +0.645]** over 20k
resamples; the arms diverged on **458/1000** deals, D=128 winning 272 and losing 186,
**sign test p = 0.00007**. All four identical-policy controls returned exactly
`+0.000 ± 0.000` with `max|edge|` exactly `0`.

The audit record already pointed this way and the summary had smoothed it away:
`audit_v9/07_BUILD.md` reports the rollout-PIMC teacher going **+0.003 (D=8) → +0.317
(D=16) → +0.483 (D=32)** against the model.

**Cost.** One exact solve, measured on real captured online positions:

| trick | mean per solve | trick | mean per solve |
|---|---|---|---|
| 1 | 1.468 s | 4 | 0.0051 s |
| 2 | 0.325 s | 5 | 0.0007 s |
| 3 | 0.031 s | 6 | 0.0001 s |

A whole decision at D=128 from trick 3 therefore runs **0.52 s median, 5.28 s at p99,
7.94 s worst**, against belot.md's 25 s turn clock — no decision was ever truncated.
Trick 0 is unreachable at any D: a *single* solve there costs 8.2 s median.

**Not measured:** where on the 8 → 128 curve the gain begins. D=32 costs 0.11 s
median per decision and may capture most of it.

**What replay cannot do.** Re-scoring recorded hands at a different D gives timing
and decision changes exactly, and strength not at all — once a different card is
played, the opponents' replies are a counterfactual that was never observed.

---

## 4. Why the agent plateaued

Training ran ~3,000 epochs and stopped improving. The cause is structural, not a
tuning problem.

**The policy-gradient update is exactly zero where the policy is confident.** The
score function `∂log π(a)/∂z_b = [b == a] − π(b)` vanishes for a decision taken with
probability 1.000. Measured **participation: 10.8%** — roughly one decision in nine
carries any gradient at all. 35% of decisions are forced outright, and bidding's
25th-percentile `π(a)` is **0.9989**.

That single fact explains the small gradient, the failure of every entropy
intervention, the one-off-shift-then-flat shape of each training arm, and why
successive arms move orthogonally (cosine 0.002–0.011 against a random-vector floor
of 0.00195) and never compound. Measured consequence: **660 epochs of training were
worth +0.089 ± 0.286 head-to-head**, on 692/1000 deals played differently — the
policy changes materially while its strength does not.

`python tools/participation_ratio.py` and `python tools/gradient_noise_scale.py`.

---

## 5. The ceiling, and why more search does not reach it

**Perfect information is worth +2.110 ± 0.349 pts/hand** above the deliverable
(n=500, control exactly zero) — the bound on everything the search family could ever
gain. It splits almost evenly: knowing the partner's hand +1.088 ± 0.324, knowing
both opponents' +1.022 ± 0.291. `python tools/information_headroom.py`.

Four independent axes of the card-play search were measured and all four are flat:

| axis | result |
|---|---|
| coverage | trick 2 null with both evaluators; tricks 0–2 null-to-negative |
| determinizations | D 8→16 null twice; D 32→128 +0.155 ± 0.322 |
| evaluator | exact beats rollout by +0.360 ± 0.204 late — **already deployed** |
| gating | every gate within ±0.25 of the baseline |

Determinization variance was attacked directly and does not yield: common random
numbers are already implemented, and stratifying the draws on the highest unseen
trump gave an effective-D multiplier of **0.88×** — slightly adverse.

---

## 6. Bidding

Bidding is the one component the policy gradient never trained. Its headroom looks
enormous and almost all of it is unreachable.

| bidder, card play identical on both sides | pts/hand |
|---|---|
| clairvoyant (knows the realised deal) | **+4.011 ± 0.273** |
| the network's own bidder | 0.000 by definition |
| greedy heuristic | −0.180 ± 0.246 |
| uniform random | −2.449 ± 0.277 |

`python tools/bidding_headroom.py` reproduces the first two rows.

**But a real bidder cannot see the future.** Averaging over 32 sampled worlds instead
of knowing the true one collapses the ceiling from +4.011 to **at most +0.255 ±
0.063** — a rigorous upper bound, since the estimate of a maximum is biased upward
and the true headroom is non-negative. So **≈94% of the apparent bidding ceiling is
clairvoyance over card luck, not bidding skill**, and the network's bidder already
captures **90.6%** of the random-to-perfect range.

A bidder that must estimate the value of each bid from a feasible number of sampled
worlds does worse than not choosing at all: cross-fitted, it is −1.534 with 1 world,
−0.542 with 8, and still **−0.263 with 16**.

---

## 7. What was tried and did not work

Reported because the negative results are most of the evidence, and each one is a
measurement rather than an opinion.

| attempt | result |
|---|---|
| learned evaluator replacing the rollout inside the search | regret 2.167 ± 0.305 vs the rollout's 1.591 ± 0.265 — significantly worse; in play −0.372 ± 0.325 |
| learned belief network feeding the determinization sampler | −0.0083 ± 0.0013 log-likelihood against the environment's own constraint tracker — worse |
| expert iteration on the base policy | structurally confined to tricks 0–2, where the best affordable teacher measured +0.167 ± 0.285 — not enough to teach anything |
| attributed play history added to the observation | −0.003 ± 0.126 raw points of held-out regret; its entire measured effect is undoing the capacity cost of its own 626 dimensions |
| privileged card locations annealed into the belief block | worth −0.080 ± 0.097 *even at train time*, so there is nothing to bootstrap from |
| Deep Monte Carlo on card play (1.85M hands) | composite **−0.916 ± 0.207** |
| Deep Monte Carlo on bidding (1.21M hands) | isolated bidder **−1.175 ± 0.215** |

### The common reason

The action-controllable part of a Belot outcome is a small fraction of its variance.
Measured on held-out states with exact double-dummy labels:

| | sd, raw points |
|---|---|
| across states — how much the *state's* value varies | **30.850** |
| within a state — how much the *card choice* moves it | **3.400** |

A ratio of 0.110, so the action accounts for about **1.2% of the variance** any
return-regression is fitting. A value function trained on realised returns therefore
learns which deal it is holding, not which card to play: its within-state
correlation with the exact values was **+0.043**, and it did not grow over a 23×
increase in training hands.

The methods that work here are the ones that **pair** the comparison — scoring every
candidate card against the *same* sampled world, so the state term cancels exactly.
That is what the search does, and it is why the search beats everything learned.

Bidding is the exception that confirms the shape: its action-to-state ratio is 0.816
rather than 0.110, and a learned bidder duly reached a within-state correlation of
**+0.547**. It still lost, because the ceiling above the network's bidder is ≤ +0.255
while the floor below is −2.449.

---

## 8. Known limitations

- **The belief block's IPF runs a fixed 6 rounds.** An early-terminating 24-round
  version converges more tightly. Every number on this page was measured with the
  file as it ships, so the fixed version stays as-is rather than silently changing
  the code the results describe.
- **The double-dummy solver does not model the zero-tricks (−10) rule**, which needs
  final trick counts it does not carry. From trick 3 a team that has taken a trick
  cannot trigger it. Stated rather than hidden.
- **Everything here was measured against the agent itself or a frozen copy.** Against
  human opponents the belief and opponent-modelling results would need re-measuring;
  a near-deterministic self-play corpus is the easiest possible case for a
  constraint tracker and the hardest possible case for learning to read an opponent.
