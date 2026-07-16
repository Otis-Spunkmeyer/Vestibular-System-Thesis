"""
Step 5 - inspect the PSO result.

Loads the best gains, prints baseline-vs-best cost, and overlays the best-fit
model FRF on the human target.  Run after pso.py:  python run_fit.py
"""
import numpy as np
import matplotlib.pyplot as plt
from model_frf import model_frf, BASELINE_GAINS
from target import interp_target
from cost import frf_cost

best = np.load("best_gains.npy")
base = np.array(BASELINE_GAINS)
print("baseline gains", np.round(base, 3), " cost", round(frf_cost(base), 4))
print("best     gains", np.round(best, 3), " cost", round(frf_cost(best), 4))

f, g, p = model_frf(best)
gt, pt, _ = interp_target(f)

fig, ax = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
ax[0].loglog(f, gt, "ko-", label="target (human)")
ax[0].loglog(f, g, "r.-", label="model (best)")
ax[0].set_ylabel("Gain"); ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
ax[1].semilogx(f, pt, "ko-")
ax[1].semilogx(f, p, "r.-")
ax[1].set_ylabel("Phase (deg)"); ax[1].set_xlabel("Frequency (Hz)")
ax[1].grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("frf_fit_result.png", dpi=110)
plt.show()
