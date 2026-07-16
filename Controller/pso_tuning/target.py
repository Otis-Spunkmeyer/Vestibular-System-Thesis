"""
Step 1 - the digitized human target (Peterka Fig 5C).

Loads the three CSVs (each with its own, non-aligned frequency samples) and
resamples gain / phase / coherence onto ANY set of frequencies via
log-frequency interpolation.  Step 0 settled the gain units: linear ratio.
"""
import os
import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "digitized data")

# Step 0 (settled): the CSV column is LINEAR gain, not dB.
# log10(gain) lands on an exact 1/48-decade lattice, i.e. the digitizer read
# Fig 5C's log y-axis directly.  Reading it as dB flattens it to 1.02..1.42.
GAIN_IS_DB = False


def _load(name):
    """Read a 2-column 'freq, value' CSV -> (freq_array, value_array)."""
    a = np.loadtxt(os.path.join(_DATA, name), delimiter=",")
    return a[:, 0], a[:, 1]


def interp_target(freqs):
    """Return (gain, phase_deg, coherence) sampled at `freqs` (Hz)."""
    fg, g = _load("Peterka5c_GainvsFreq(hz).csv")
    fp, p = _load("Peterka5c_PhasevsFreq(Hz).csv")
    fc, c = _load("CoherencevsFreq(Hz).csv")

    if GAIN_IS_DB:
        g = 10 ** (g / 20)                      # dB -> linear ratio

    lf = np.log10(freqs)                        # interpolate in log-frequency
    gain  = np.interp(lf, np.log10(fg), g)
    phase = np.interp(lf, np.log10(fp), p)
    coh   = np.interp(lf, np.log10(fc), c)
    return gain, phase, coh
