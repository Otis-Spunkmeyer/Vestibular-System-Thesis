"""
Step 3 - the scalar objective the PSO minimizes.

Compares the model FRF to the digitized human target on a fixed set of in-band
frequencies, as a coherence-weighted mean of log-gain error and phase error.
"""
import numpy as np
from model_frf import model_frf, analysis_freqs, BASELINE_GAINS
from target import interp_target

# Computed once at import: the fixed analysis grid + the target sampled on it.
_F = analysis_freqs()                 # in-band PRTS harmonics (gain-independent)
_G_T, _P_T, _COH = interp_target(_F)  # human gain / phase / coherence on that grid


def _on_grid(f, v):
    """Interpolate a model quantity (defined at freqs f) onto the fixed grid _F."""
    o = np.argsort(f)
    return np.interp(np.log10(_F), np.log10(f[o]), v[o])


def frf_cost(gains, w_gain=1.0, w_phase=1.0, fail=1e6):
    """Scalar fit cost for one gain vector (lower = better)."""
    f, g, p = model_frf(gains)
    if f is None or len(f) == 0:
        return fail                              # blew up / no usable output
    gm = _on_grid(f, g)
    pm = _on_grid(f, p)
    eg = (np.log10(np.clip(gm, 1e-3, None)) - np.log10(_G_T)) ** 2
    ep = ((pm - _P_T) / 180.0) ** 2
    return float(np.sum(_COH * (w_gain * eg + w_phase * ep)) / _COH.sum())


if __name__ == "__main__":
    print("analysis freqs (Hz):", np.round(_F, 3))
    print("baseline cost:", frf_cost(BASELINE_GAINS))
