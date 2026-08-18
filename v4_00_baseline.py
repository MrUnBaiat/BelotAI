"""
AUDIT v4 / step 0 -- re-establish the headline number on the checkpoint I was given.

AUDIT_HANDOFF section 2 claims +1.89 +- 0.21 pts/hand vs the greedy heuristic at
epoch ~2200. Nothing downstream means anything if that does not reproduce on this
checkpoint, so measure it first, with a CI, before touching a hypothesis.

Also reports the checkpoint metadata (epoch, selection metric, stored best_metric)
and whether best_model.pt and latest_model.pt are actually the same weights.
"""
import hashlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from eval import evaluate_matches
from model import RecurrentMAPPOModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 250


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    net = RecurrentMAPPOModel(hidden_dim=512).to(DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, ckpt


def main():
    for p in ("checkpoints/latest_model.pt", "checkpoints/best_model.pt"):
        net, ck = load(p)
        print(f"{p}: epoch={ck.get('epoch')} "
              f"selection_metric={ck.get('selection_metric')!r} "
              f"best_metric={ck.get('best_metric')} "
              f"weights_sha={state_hash(ck['model_state_dict'])}")

    model, _ = load("checkpoints/latest_model.pt")

    for opp in ("heuristic", "random"):
        # Re-seed so every condition I ever compare sees the identical deals.
        np.random.seed(12345)
        torch.manual_seed(12345)
        t0 = time.time()
        r = evaluate_matches(model, num_matches=MATCHES, device=DEVICE,
                             opponent=opp, greedy=True)
        n_hands = r["avg_hands_per_match"] * MATCHES
        print(f"\nvs {opp}: {MATCHES} matches / ~{n_hands:.0f} hands "
              f"({time.time() - t0:.0f}s)")
        print(f"  avg_hand_diff   {r['avg_hand_diff']:+.3f} +- {r['hand_diff_ci95']:.3f}")
        print(f"  match_win_rate  {r['match_win_rate']:.3f} +- {r['match_win_ci95']:.3f}"
              f"   ties {r['match_tie_rate']:.3f}")
        print(f"  hand_win_rate   {r['hand_win_rate']:.3f}")
        print(f"  hands/match     {r['avg_hands_per_match']:.2f}")
        print(f"  declarer bolts  us={r['declarer_bolts'][0]} them={r['declarer_bolts'][1]}")


if __name__ == "__main__":
    main()
