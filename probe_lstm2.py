"""
State-content probe (extends probe_lstm.py -- keep both files side by side).

Usage:  python probe_lstm2.py checkpoints/latest_model.pt [n_strength_hands]

Decision-level conditions, evaluated at the SAME states on baseline trajectories:
  base       true carried hidden state, full obs
  zero       h,c = 0            (training-familiar reset state)
  garbage    h,c ~ N(0, sigma)  per-unit sigma matched to real harvested states
  swap       real h,c harvested from a DIFFERENT game at the same trick depth
  no-bel+grv true state, obs belief block AND graveyard block zeroed

Metrics per decision, bucketed by trick number:
  masked KL(base||alt) and argmax agreement   -> legal-actions-only comparison
    (masked distributions carry ~0 mass on illegal actions, so this IS the
     "only the ones the action mask allows" comparison)
  raw-logit cosine similarity over all 38 pre-mask outputs -> representation
    drift, independent of the mask

Then paired strength runs (identical deals, deterministic opponents) for
zero / garbage / swap applied at EVERY decision, to tie decision drift to
game-point outcomes.

Interpretation matrix for the state conditions:
  zero~base, garbage/swap << base -> net READS the state pathway but the
      content adds no value (zero is just a safe learned default)
  zero~garbage~swap~base          -> downstream weights discount h/c entirely;
      the LSTM can be removed from the architecture at ~no cost
  swap ~ base but garbage << base -> net only needs the state to look
      statistically normal; content is interchangeable across games (memory
      stores style, not game facts)
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')  # run from the project root
from env import BelotEnv
from observation import build_observation
from model import RecurrentMAPPOModel
from probe_lstm import heuristic_action, zero_state, HIDDEN, BEL, GRV

HARVEST_GAMES = 250
DIVERGENCE_GAMES = 300


@torch.no_grad()
def forward_raw(model, env, seat, hc, device, zero_ranges=()):
    """Returns (raw_logits(38,), masked_probs(38,), new_hc)."""
    local, glob, mask = build_observation(env, seat, [0, 0])
    for a, b in zero_ranges:
        local[a:b] = 0.0
    lt = torch.from_numpy(local).unsqueeze(0).to(device)
    gt = torch.from_numpy(glob).unsqueeze(0).to(device)
    mt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
    dist_raw, _, new_hc = model(lt, gt, hc, None, is_sequence=False)   # no mask
    raw = dist_raw.logits.squeeze(0)
    masked = raw.masked_fill(~mt.squeeze(0).bool(), -1e9)
    probs = torch.softmax(masked, dim=-1)
    return raw, probs, new_hc


def bucket(env):
    return "BID" if env.phase == "BIDDING" else env.tricks_played


@torch.no_grad()
def harvest_states(model, device, seed):
    """Collect real (h,c) states by bucket, plus per-unit std for garbage noise."""
    np.random.seed(seed)
    pool = {}
    hs, cs = [], []
    for g in range(HARVEST_GAMES):
        env = BelotEnv(); env.dealer = g % 4; env.reset()
        hc = {s: zero_state(device) for s in (0, 2)}
        while not env.done:
            s = env.current_player
            if s % 2 == 0:
                pool.setdefault(bucket(env), []).append(
                    (hc[s][0].clone(), hc[s][1].clone()))
                hs.append(hc[s][0].flatten()); cs.append(hc[s][1].flatten())
                _, probs, hc[s] = forward_raw(model, env, s, hc[s], device)
                a = int(probs.argmax().item())
            else:
                a = heuristic_action(env)
            env.step(a)
    sigma_h = torch.stack(hs).std(dim=0).view(1, 1, HIDDEN).clamp_min(1e-4)
    sigma_c = torch.stack(cs).std(dim=0).view(1, 1, HIDDEN).clamp_min(1e-4)
    return pool, sigma_h, sigma_c


def make_alt_state(kind, true_hc, buck, pool, sig, device, trng):
    if kind == "zero":
        return zero_state(device)
    if kind == "garbage":
        sigma_h, sigma_c = sig
        return (torch.randn(1, 1, HIDDEN, generator=trng, device=device) * sigma_h,
                torch.randn(1, 1, HIDDEN, generator=trng, device=device) * sigma_c)
    if kind == "swap":
        cand = pool.get(buck) or pool.get("BID") or next(iter(pool.values()))
        # FIX: explicitly pass device=device to match the CUDA generator
        i = int(torch.randint(len(cand), (1,), generator=trng, device=device).item())
        return cand[i]
    return true_hc  # 'no-bel+grv' keeps the true state


ALTS = ["zero", "garbage", "swap", "no-bel+grv"]


@torch.no_grad()
def divergence(model, pool, sig, device, seed):
    np.random.seed(seed)
    trng = torch.Generator(device=device); trng.manual_seed(seed)
    S = {k: {"kl": [], "agree": [], "cos": [], "trick": []} for k in ALTS}
    for g in range(DIVERGENCE_GAMES):
        env = BelotEnv(); env.dealer = g % 4; env.reset()
        hc = {s: zero_state(device) for s in (0, 2)}
        while not env.done:
            s = env.current_player
            if s % 2 == 0:
                raw_b, p, new_hc = forward_raw(model, env, s, hc[s], device)
                a = int(p.argmax().item())
                if env.phase == "PLAYING":
                    for k in ALTS:
                        zr = [BEL, GRV] if k == "no-bel+grv" else ()
                        st = make_alt_state(k, hc[s], bucket(env), pool, sig, device, trng)
                        raw_a, q, _ = forward_raw(model, env, s, st, device, zr)
                        kl = float((p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum())
                        cos = float(torch.nn.functional.cosine_similarity(
                            raw_b.unsqueeze(0), raw_a.unsqueeze(0)).item())
                        S[k]["kl"].append(kl)
                        S[k]["agree"].append(int(q.argmax().item()) == a)
                        S[k]["cos"].append(cos)
                        S[k]["trick"].append(env.tricks_played)
                hc[s] = new_hc
            else:
                a = heuristic_action(env)
            env.step(a)
    return S


@torch.no_grad()
def strength(model, kind, pool, sig, n_hands, device, seed):
    """Paired strength run with the alt state applied at EVERY model decision."""
    np.random.seed(seed)
    trng = torch.Generator(device=device); trng.manual_seed(seed + 1)
    diffs = []
    for g in range(n_hands):
        env = BelotEnv(); env.dealer = g % 4; env.reset()
        hc = {s: zero_state(device) for s in (0, 2)}
        info = {}
        while not env.done:
            s = env.current_player
            if s % 2 == 0:
                st = hc[s] if kind == "base" else \
                    make_alt_state(kind, hc[s], bucket(env), pool, sig, device, trng)
                _, p, new_hc = forward_raw(model, env, s, st, device)
                if kind == "base":
                    hc[s] = new_hc     # alt conditions overwrite state anyway
                a = int(p.argmax().item())
            else:
                a = heuristic_action(env)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        diffs.append(gp[0] - gp[1])
    return np.array(diffs, dtype=np.float64)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest_model.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentMAPPOModel().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"checkpoint: {path}\nharvesting real hidden states ({HARVEST_GAMES} games)...")
    pool, sh, sc = harvest_states(model, device, seed=7)
    print(f"pooled states per bucket: " +
          ", ".join(f"{k}:{len(v)}" for k, v in sorted(pool.items(), key=lambda kv: str(kv[0]))))

    print(f"\nPer-decision comparison at identical states (PLAYING, {DIVERGENCE_GAMES} games)")
    S = divergence(model, pool, (sh, sc), device, seed=11)
    print(f"{'condition':12s} {'maskedKL':>9} {'agree':>7} {'rawCos38':>9}   agree by trick 0..7")
    for k in ALTS:
        kl = np.mean(S[k]["kl"]); ag = np.mean(S[k]["agree"]); co = np.mean(S[k]["cos"])
        tr = np.array(S[k]["trick"]); agv = np.array(S[k]["agree"], dtype=float)
        by = " ".join(f"{agv[tr == t].mean():.2f}" if (tr == t).any() else "  - "
                      for t in range(8))
        print(f"{k:12s} {kl:9.4f} {ag:6.1%} {co:9.4f}   {by}")

    print(f"\nPaired strength vs heuristic ({n} identical deals per condition)")
    base = strength(model, "base", pool, (sh, sc), n, device, seed=123)
    print(f"{'base':12s} avg point diff {base.mean():+6.2f}")
    for k in ["zero", "garbage", "swap"]:
        d = strength(model, k, pool, (sh, sc), n, device, seed=123)
        delta = d - base
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        print(f"{k:12s} avg point diff {d.mean():+6.2f} | delta {delta.mean():+5.2f} +-{1.96*se:.2f}")

    print("\nReading guide: see module docstring (zero vs garbage vs swap matrix).")


if __name__ == "__main__":
    main()