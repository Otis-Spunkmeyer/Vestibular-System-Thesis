"""
Step 4 - Particle Swarm Optimization over the four SNS gains.

Hand-rolled numpy PSO. Particle evaluations run in parallel processes because
each frf_cost() is a full multi-second simulation.
Run from inside Controller/pso_tuning/:  python pso.py
"""
import numpy as np
import multiprocessing as mp
from cost import frf_cost
from model_frf import BASELINE_GAINS

# search bounds for [kp, kd, kc, kt]; upper kp respects the ~9 nA divergence limit
BOUNDS = np.array([[0.5, 9.0],
                   [0.5, 15.0],
                   [0.5, 15.0],
                   [0.0, 10.0]])


def pso(n=16, iters=25, w=0.7, c1=1.5, c2=1.5, seed=0, procs=None):
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]

    X = rng.uniform(lo, hi, size=(n, 4))                 # particle positions (gains)
    # Seed particle 0 at the hand-tuned baseline. Without this the swarm is
    # entirely random, gbest is saved unconditionally, and a full run can return
    # gains WORSE than the ones we started from. Since gbest_c only ever
    # decreases from an initial value <= c[0], seeding makes the result a hard
    # guarantee: never worse than baseline. Its velocity stays random, so the
    # particle still explores -- pbest[0] holds the baseline regardless.
    X[0] = np.clip(BASELINE_GAINS, lo, hi)               # clip: BOUNDS may be narrowed
    V = rng.uniform(-1, 1, size=(n, 4)) * (hi - lo) * 0.1  # initial velocities

    with mp.Pool(procs) as pool:
        c = np.array(pool.map(frf_cost, list(X)))        # evaluate swarm in parallel
        pbest, pbest_c = X.copy(), c.copy()              # each particle's best-so-far
        gi = c.argmin()
        gbest, gbest_c = X[gi].copy(), c[gi]             # global best

        for it in range(iters):
            r1, r2 = rng.random((n, 4)), rng.random((n, 4))
            V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
            X = np.clip(X + V, lo, hi)                    # move, keep inside bounds
            c = np.array(pool.map(frf_cost, list(X)))

            imp = c < pbest_c                            # improved personal bests
            pbest[imp], pbest_c[imp] = X[imp], c[imp]
            j = pbest_c.argmin()
            if pbest_c[j] < gbest_c:
                gbest, gbest_c = pbest[j].copy(), pbest_c[j]
            print(f"iter {it:3d}  gbest={gbest_c:.4f}  gains={np.round(gbest, 3)}",
                  flush=True)

    np.save("best_gains.npy", gbest)
    print("saved best_gains.npy:", np.round(gbest, 4), " cost", round(float(gbest_c), 4))
    return gbest, gbest_c


if __name__ == "__main__":          # required on Windows (multiprocessing spawn)
    pso()
