"""
AUDIT v4 / EXP-5 -- a hard confidence interval on the central claim.

EXP-4's headline is |G|^2 ~ 1.2e-5 and B_simple ~ 1.9e5 episodes, inferred by
subtracting two much larger numbers (at 16,640 episodes the noise term is still
~11x the signal term). A point estimate obtained that way needs an explicit CI
before anything is built on it, and the B=8192 row had only ONE disjoint pair, so
its "+-0.0000" is a degenerate interval, not a tight one.

This re-analyses the cached 130 chunk gradients with:
  1. Bootstrap over chunks -> CI on |G|^2, tr(Sigma) and B_simple.
  2. Held-out cosine over 500 RANDOM disjoint half-splits (65 vs 65 chunks) with
     a bootstrap CI. E[cos] > 0 iff |G|^2 > 0, so this is a direct sign test on
     "does a systematic direction exist at all".
  3. A permutation null: randomly flip the sign of each chunk gradient. That
     destroys any common direction while preserving the noise structure exactly,
     so the null distribution of the same statistic is the correct yardstick.

DECISION RULE. If the observed half-split cosine sits above the 99th percentile
of the sign-flip null, a systematic improvement direction exists and the plateau
is a signal-to-noise problem, not convergence.
"""
import numpy as np

CHUNK = 128
G = np.load("v4_chunk_grads.npy")        # (n_chunks, n_coords)
N = G.shape[0]
rng = np.random.default_rng(0)
print(f"cached gradients: {G.shape[0]} chunks x {G.shape[1]:,} coords "
      f"({CHUNK} episodes per chunk)")


def mccandlish(mat, b_small):
    """|G|^2, tr(Sigma), B_simple from a set of equal-sized disjoint batch grads."""
    n = mat.shape[0]
    Gm = mat.mean(0)
    gsq_s = float((mat ** 2).sum(1).mean())
    gsq_b = float((Gm ** 2).sum())
    bs, bb = float(b_small), float(b_small * n)
    g2 = (bb * gsq_b - bs * gsq_s) / (bb - bs)
    trS = (gsq_s - gsq_b) / (1.0 / bs - 1.0 / bb)
    return g2, trS, (trS / g2 if g2 > 0 else np.inf)


g2, trS, bsimple = mccandlish(G, CHUNK)
print(f"\npoint estimates over all {N} chunks ({N * CHUNK:,} episodes)")
print(f"  |G|^2      {g2:.4e}")
print(f"  tr(Sigma)  {trS:.4e}")
print(f"  B_simple   {bsimple:,.0f} episodes")

# ---------- 1. bootstrap over chunks ----------
B = 3000
vals = np.empty((B, 3))
for i in range(B):
    s = rng.integers(0, N, N)
    vals[i] = mccandlish(G[s], CHUNK)
lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
print(f"\nbootstrap ({B} resamples over chunks), 95% CI")
print(f"  |G|^2      [{lo[0]:.3e}, {hi[0]:.3e}]   "
      f"{'EXCLUDES 0 -> signal exists' if lo[0] > 0 else 'includes 0 -> unresolved'}")
print(f"  tr(Sigma)  [{lo[1]:.3e}, {hi[1]:.3e}]")
print(f"  B_simple   [{lo[2]:,.0f}, {hi[2]:,.0f}] episodes")

# ---------- 2. held-out cosine over random disjoint half-splits ----------
def half_split_cos(mat, n_splits, generator):
    out = np.empty(n_splits)
    h = mat.shape[0] // 2
    for i in range(n_splits):
        p = generator.permutation(mat.shape[0])
        a, b = mat[p[:h]].mean(0), mat[p[h:2 * h]].mean(0)
        out[i] = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return out


obs = half_split_cos(G, 500, rng)
m = obs.mean()
boot = np.array([obs[rng.integers(0, len(obs), len(obs))].mean() for _ in range(3000)])
clo, chi = np.percentile(boot, [2.5, 97.5])
print(f"\nheld-out cosine, 500 random disjoint 65-vs-65 splits "
      f"({65 * CHUNK:,} episodes per side)")
print(f"  mean cosine {m:+.5f}   95% CI [{clo:+.5f}, {chi:+.5f}]")

# ---------- 3. sign-flip permutation null ----------
null = np.empty(500)
for i in range(500):
    s = rng.choice([-1.0, 1.0], size=(N, 1))
    null[i] = half_split_cos(G * s, 1, rng)[0]
p99 = np.percentile(null, 99)
pval = float((null >= m).mean())
print(f"\nsign-flip null (destroys any common direction, preserves noise)")
print(f"  null mean {null.mean():+.5f}   null 99th pct {p99:+.5f}   "
      f"null max {null.max():+.5f}")
print(f"  observed {m:+.5f}  ->  p = {pval:.4f}")

print("\n" + "=" * 78)
if clo > 0 and pval < 0.01:
    print("CONFIRMED: a systematic improvement direction EXISTS but is ~"
          f"{bsimple / CHUNK:,.0f}x below")
    print(f"the noise at the 128-episode optimizer step. The plateau is a")
    print(f"signal-to-noise failure, not convergence to a local optimum.")
else:
    print("NOT RESOLVED: cannot reject the no-common-direction null.")
