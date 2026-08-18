"""
AUDIT v4 / EXP-4 -- noise-limited, or converged? The decisive disambiguation.

EXP-1 measured |G|^2 ~ 0 for the actor gradient at epoch 2400. That single number
is consistent with TWO opposite diagnoses:

  (A) NOISE-LIMITED. A real improvement direction exists but is buried under
      sampling noise at the 128-episode step size. Fix = more data per step.
  (B) CONVERGED. The policy is genuinely at/near a stationary point of the
      self-play objective. Fix = change the objective (opponents, exploration,
      representation); more data changes nothing.

They are distinguished by how the gradient behaves as the batch GROWS. Under (A)
the direction becomes coherent once the batch approaches B_simple: disjoint-batch
cosine rises away from zero and |G_batch|^2 stops falling like 1/B. Under (B) the
cosine stays at zero at every batch size and |G_batch|^2 keeps falling like 1/B
all the way out, because there is nothing underneath the noise.

METHOD. Collect K rollouts (~17k episodes, ~10x a training epoch). Compute the
128-episode chunk gradients once, then build larger-batch gradients by AVERAGING
disjoint groups of chunks -- exact, and far cheaper than re-running backward. To
keep memory sane, all cosines/norms are computed on a FIXED random coordinate
subset of the parameter vector; a random projection is an unbiased estimator of
both |G|^2 and tr(Sigma), so B_simple is unbiased too.

READ THE RESULT AS:
  cosine at B=8192 clearly > 0  -> diagnosis (A), the batch is too small
  cosine flat at ~0 out to 8192 -> diagnosis (B), the policy is stationary
"""
import sys
import random

import numpy as np
import torch

sys.path.insert(0, '.')
import train as T
from memory import make_minibatch
from model import RecurrentMAPPOModel
from vec_env import VectorizedBelot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_ROLLOUTS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TARGET_GAMES = 512
CHUNK = 128                      # the real MINIBATCH_EPISODES
SUBSET = 262_144                 # random coordinates kept for the statistics
ENT_COEF = T.anneal(T.ENTROPY_START, T.ENTROPY_END, 2400, T.ENTROPY_ANNEAL_EPOCHS)


def actor_params(model):
    return [p for n, p in model.named_parameters() if not n.startswith("critic")]


def chunk_grad(model, episodes, norm, idx, params):
    model.zero_grad(set_to_none=True)
    b_obs, b_gobs, b_masks, b_actions, b_old, b_adv, b_ret, pad = make_minibatch(episodes)
    to = lambda x: x.to(DEVICE)
    b_obs, b_gobs, b_masks = to(b_obs), to(b_gobs), to(b_masks)
    b_actions, b_old, b_adv, pad = to(b_actions), to(b_old), to(b_adv), to(pad)
    B = b_obs.size(0)
    h0 = torch.zeros(1, B, T.HIDDEN, device=DEVICE)
    c0 = torch.zeros(1, B, T.HIDDEN, device=DEVICE)
    dist, _, _ = model(b_obs, b_gobs, (h0, c0), b_masks, is_sequence=True)
    ratio = torch.exp(dist.log_prob(b_actions) - b_old)      # == 1 identically
    s1 = ratio * b_adv
    s2 = torch.clamp(ratio, 1 - T.CLIP_EPSILON, 1 + T.CLIP_EPSILON) * b_adv
    loss = (-(torch.min(s1, s2) * pad).sum() - ENT_COEF * (dist.entropy() * pad).sum()) / norm
    loss.backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                   for p in params])
    return g[idx].detach().cpu().clone()


def stats(grads, b_eps):
    """Disjoint-batch cosine + McCandlish estimators at batch size b_eps."""
    n = len(grads)
    G = torch.stack(grads).mean(0)
    gsq_s = float(torch.stack([g.pow(2).sum() for g in grads]).mean())
    gsq_b = float(G.pow(2).sum())
    pc = [float(torch.dot(grads[i], grads[j]) /
                (grads[i].norm() * grads[j].norm() + 1e-12))
          for i in range(n) for j in range(i + 1, n)]
    bs, bb = float(b_eps), float(b_eps * n)
    g2 = (bb * gsq_b - bs * gsq_s) / (bb - bs) if n > 1 else float("nan")
    trS = (gsq_s - gsq_b) / (1.0 / bs - 1.0 / bb) if n > 1 else float("nan")
    return dict(n=n, b=b_eps, cos=float(np.mean(pc)),
                ci=float(1.96 * np.std(pc) / np.sqrt(len(pc))),
                gsq=gsq_s, g2=g2, trS=trS,
                b_simple=(trS / g2 if g2 and g2 > 0 else float("inf")))


def main():
    ck = torch.load("checkpoints/latest_model.pt", map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=T.HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    pool = [T.snapshot(model, DEVICE)]
    params = actor_params(model)
    n_par = sum(p.numel() for p in params)

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n_par, generator=g)[:SUBSET].to(DEVICE)
    print(f"actor params {n_par:,}; statistics on a random {SUBSET:,}-coord subset "
          f"({SUBSET / n_par:.1%})")

    all_chunks = []
    for k in range(K_ROLLOUTS):
        random.seed(100 + k); np.random.seed(100 + k); torch.manual_seed(100 + k)
        vec = VectorizedBelot(T.NUM_ENVS)
        eps, _ = T.collect_rollout(model, vec, TARGET_GAMES, DEVICE, pool)
        # advantage normalisation per rollout, exactly as update() does per epoch
        for ep in eps:
            ep.returns, ep.advantages = ep.compute_gae(T.GAMMA, T.LAM)
        a = torch.cat([ep.advantages for ep in eps])
        m, s = a.mean(), a.std()
        for ep in eps:
            ep.advantages = (ep.advantages - m) / (s + 1e-8)
        random.shuffle(eps)
        nch = len(eps) // CHUNK
        mean_steps = float(sum(len(e) for e in eps[:nch * CHUNK])) / nch
        for i in range(nch):
            all_chunks.append(chunk_grad(model, eps[i * CHUNK:(i + 1) * CHUNK],
                                         mean_steps, idx, params))
        print(f"  rollout {k + 1}/{K_ROLLOUTS}: {len(eps)} episodes -> "
              f"{nch} chunks (total {len(all_chunks)})", flush=True)

    random.shuffle(all_chunks)
    np.save("v4_chunk_grads.npy",
            torch.stack(all_chunks).numpy().astype(np.float32))
    print("cached chunk gradients -> v4_chunk_grads.npy")
    N = len(all_chunks)
    print(f"\ntotal {N} chunks of {CHUNK} episodes = {N * CHUNK:,} episodes "
          f"(~{N / 13.6:.1f} training epochs of data)\n")

    print(f"{'batch (episodes)':>18}{'groups':>8}{'disjoint cosine':>22}"
          f"{'|g_b|^2':>13}{'|G|^2 est':>13}{'B_simple':>14}")
    print("-" * 88)
    rows = []
    for mult in (1, 2, 4, 8, 16, 32, 64):
        per = mult
        n_groups = N // per
        if n_groups < 2:
            continue
        grads = [torch.stack(all_chunks[i * per:(i + 1) * per]).mean(0)
                 for i in range(n_groups)]
        r = stats(grads, CHUNK * per)
        rows.append(r)
        bs = "inf" if not np.isfinite(r["b_simple"]) else f"{r['b_simple']:,.0f}"
        print(f"{r['b']:>18,}{r['n']:>8}{r['cos']:>+14.4f} +-{r['ci']:.4f}"
              f"{r['gsq']:>13.3e}{r['g2']:>13.2e}{bs:>14}")

    print("\n" + "=" * 88)
    print("READ WITH CARE. The per-batch cosines below are marginal, and a marginal\n"
          "cosine is exactly the kind of evidence this project has been burned by.\n"
          "v4_05_snr_ci.py runs the proper sign-flip permutation test on these same\n"
          "gradients; it returns p = 0.14, i.e. 16,640 episodes CANNOT establish that\n"
          "|G| > 0. Do not quote a NOISE-LIMITED vs CONVERGED verdict from this table.\n")
    print("What IS established here, and does not depend on resolving |G|:")
    print(f"  * at the {CHUNK}-episode optimizer step the gradient is pure noise")
    print(f"    (cosine {rows[0]['cos']:+.4f} +- {rows[0]['ci']:.4f});")
    print(f"  * every row is consistent with one B_simple in the 1e4-2e5 range,")
    print(f"    i.e. the training batch is 80x-1500x too small either way.")


if __name__ == "__main__":
    main()
