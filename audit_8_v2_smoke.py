"""Small-config end-to-end smoke test of the v2 stack (CPU)."""
import sys, time, random
import numpy as np, torch, torch.optim as optim
sys.path.insert(0, '.')
import train as T, eval as EV
from vec_env import VectorizedBelot
from model import RecurrentMAPPOModel

T.HIDDEN = 192; EV.HIDDEN = 192
T.MINIBATCH_EPISODES = 96
T.SNAPSHOT_EVERY = 8; T.FROZEN_POOL_MAX = 2
NUM_ENVS, TARGET_GAMES, EPOCHS, MATCHES = 16, 96, 24, 25

def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(4)
    dev='cpu'
    model = RecurrentMAPPOModel(hidden_dim=T.HIDDEN)
    opt = optim.Adam(model.parameters(), lr=T.LR_START)
    vec = VectorizedBelot(NUM_ENVS)
    pool = [T.snapshot(model, dev)]          # stand-in for best_model.pt
    ref = pool[0]
    r = EV.evaluate_matches(model, MATCHES, dev, "random")
    print(f"epoch  -1 | vs random match% {r['match_win_rate']:.2f} handdiff {r['avg_hand_diff']:+.2f} (untrained)")
    for ep in range(EPOCHS):
        lr = T.anneal(T.LR_START, T.LR_END, ep, T.LR_ANNEAL_EPOCHS)
        for g in opt.param_groups: g['lr']=lr
        ec = T.anneal(T.ENTROPY_START, T.ENTROPY_END, ep, T.ENTROPY_ANNEAL_EPOCHS)
        t0=time.time()
        eps, ri = T.collect_rollout(model, vec, TARGET_GAMES, dev, pool)
        m = T.update(model, opt, eps, dev, entropy_coef=ec)
        line=(f"epoch {ep:3d} | eps {m['episodes']:4d} | EV {m['explained_variance']:+.2f} "
              f"| KL {m['approx_kl']:.4f} | iters {m['iters_completed']}/{T.PPO_ITERS} "
              f"| entB {m['entropy_bidding']:.2f} entP {m['entropy_playing']:.2f} "
              f"| flush {ri['flush_steps']:2d} | {time.time()-t0:4.1f}s")
        if ep % 8 == 7:
            r = EV.evaluate_matches(model, MATCHES, dev, "random")
            rf = EV.evaluate_matches(model, MATCHES, dev, "model", frozen=ref)
            line += (f" | rand {r['match_win_rate']:.2f}/{r['avg_hand_diff']:+.2f}"
                     f" | ref {rf['match_win_rate']:.2f}/{rf['avg_hand_diff']:+.2f}")
        if T.SNAPSHOT_EVERY and ep>0 and ep%T.SNAPSHOT_EVERY==0:
            pool.append(T.snapshot(model, dev))
            if len(pool)>T.FROZEN_POOL_MAX: pool.pop(1)
        print(line, flush=True)

if __name__ == "__main__":
    main()
