"""
AUDIT v6 / EXP-0 -- verify the zero-LSTM-state assumption for THIS build.

EXP-1 threads no per-seat recurrent state through hypothetical rollouts: every
rollout decision is taken from a ZERO hidden state. That is only sound if the
carried state contributes nothing to strength. Handoff section 5 measured exactly
that (+0.27 +- 0.32, a null) -- but at epoch ~1775, on a different checkpoint,
before the C1 run. Inheriting it would be exactly the kind of assumption this
project has been burned by, so it is re-measured here on the checkpoint EXP-1 will
actually use.

FALSIFIED IF zeroing the state at every decision costs significantly more than
~0.2 pts/hand. In that case the rollout design must thread per-(world, seat)
hidden state, which is much more expensive, and EXP-1's timing numbers change.

Paired on identical deals, greedy policy, deterministic heuristic opponent.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from env import BelotEnv
from eval import _heuristic_action
from model import RecurrentMAPPOModel
from observation import build_observation
from v4_paired_eval import paired, summarise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CKPT = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/v4_exp/c1_latest.pt"
HIDDEN = 512
BASE = 770_000


@torch.no_grad()
def run(model, n_deals, zero_state):
    out = np.empty(n_deals)
    z = lambda: (torch.zeros(1, 1, HIDDEN, device=DEVICE),
                 torch.zeros(1, 1, HIDDEN, device=DEVICE))
    for d in range(n_deals):
        env = BelotEnv(); env.dealer = d % 4
        np.random.seed(BASE + d); env.reset(); env.bolts_by_team = [0, 0]
        hc = {s: z() for s in range(4)}
        info = {}
        while not env.done:
            seat = env.current_player
            if seat % 2 == 0:
                local, glob, mask = build_observation(env, seat, [0, 0])
                state = z() if zero_state else hc[seat]
                dist, _, new = model(
                    torch.from_numpy(local).unsqueeze(0).to(DEVICE),
                    torch.from_numpy(glob).unsqueeze(0).to(DEVICE), state,
                    torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(DEVICE),
                    is_sequence=False)
                hc[seat] = new
                a = int(dist.probs.argmax(-1).item())
            else:
                a = _heuristic_action(env)
            _, _, _, info = env.step(a)
        gp = info["game_points"]
        out[d] = gp[0] - gp[1]
    return out


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = RecurrentMAPPOModel(hidden_dim=HIDDEN).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print(f"checkpoint {CKPT} (epoch {ck.get('epoch')}), {N} paired deals\n")

    carried = run(model, N, zero_state=False)
    zeroed = run(model, N, zero_state=True)
    print("  " + summarise("carried LSTM state", carried))
    print("  " + summarise("zero state every decision", zeroed))
    print("\n  " + paired("zeroed - carried (same deals)", zeroed, carried))

    d = zeroed - carried
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n  handoff section 5 measured +0.27 +- 0.32 at epoch ~1775")
    if abs(d.mean()) < 0.2 or abs(d.mean()) <= ci:
        print("  -> ASSUMPTION HOLDS. Rollouts may use a zero hidden state, so EXP-1")
        print("     does not need per-(world, seat) recurrent threading.")
    else:
        print("  -> ASSUMPTION FAILS on this checkpoint. EXP-1 must thread hidden")
        print("     state through rollout worlds; re-cost the timing before running.")


if __name__ == "__main__":
    main()
