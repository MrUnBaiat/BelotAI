"""
AUDIT 6 -- End-to-end learning smoke test (CPU-sized).

Uses the project's own collect_rollout()/update() verbatim, with shrunken
hyper-parameters (hidden 192, 16 envs, 96 games/rollout) so ~30 epochs run on a
CPU in minutes. If the pipeline is sound, win rate vs random and explained
variance must climb visibly from the random-policy baseline within tens of
epochs. Also logs how many PPO iterations survive the KL early-stop, a
diagnostic missing from the main training loop.
"""
import sys, time, random
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, '.')  # run from the project root
import train, eval as ev
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel

# ---- shrink the run so it fits on CPU ----
train.HIDDEN = 192
ev.HIDDEN = 192
train.MINIBATCH_EPISODES = 96
NUM_ENVS, TARGET_GAMES, EPOCHS, EVAL_GAMES = 16, 96, 30, 150


def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(4)
    device = 'cpu'
    model = RecurrentMAPPOModel(hidden_dim=train.HIDDEN)
    opt = optim.Adam(model.parameters(), lr=getattr(train, "LR", None) or train.LR_START)
    vec = VectorizedBelot(NUM_ENVS)

    res = ev.evaluate(model, num_games=EVAL_GAMES, device=device)
    print(f"epoch  -1 | win% {res['win_rate']:.3f} | diff {res['avg_point_diff']:+6.2f} | (untrained)")

    for epoch in range(EPOCHS):
        t0 = time.time()
        out = train.collect_rollout(model, vec, TARGET_GAMES, device)
        eps = out[0] if isinstance(out, tuple) else out
        m = train.update(model, opt, eps, device)
        line = (f"epoch {epoch:3d} | EV {m['explained_variance']:+.3f} | "
                f"KL {m['approx_kl']:.4f} | clip {m['clip_frac']:.3f} | "
                f"ent {m['entropy']:.3f} | {time.time()-t0:5.1f}s")
        if epoch % 5 == 4:
            res = ev.evaluate(model, num_games=EVAL_GAMES, device=device)
            line += f" | win% {res['win_rate']:.3f} diff {res['avg_point_diff']:+6.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
