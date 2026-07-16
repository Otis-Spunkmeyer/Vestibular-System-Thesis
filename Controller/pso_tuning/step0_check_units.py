"""
Step 0 - decide whether the digitized gain CSV is in linear ratio or decibels.
Plots it both ways; compare the overlay against the Peterka Fig 5C gain panel.
Run from inside Controller/pso_tuning/:  python step0_check_units.py

ANSWERED: the LINEAR trace matches Fig 5C, so target.py sets GAIN_IS_DB = False.
Confirmed independently of the eyeball match: every log10(gain) in the CSV is an
exact multiple of 1/48 of a decade (to 1e-14), which is what you get when a
digitizer samples a log-scaled y-axis and records the axis value.  A dB column
would instead be quantized uniformly in the value itself.  As linear the curve
spans 0.154..3.014 peaking at 0.69 Hz (the Fig 5C shape); read as dB it
collapses to a near-flat 1.018..1.415.  Kept for reproducibility.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "digitized data")
freq, gain = np.loadtxt(os.path.join(_DATA, "Peterka5c_GainvsFreq(hz).csv"), delimiter=",").T

fig, ax = plt.subplots()
ax.loglog(freq, gain,            "o-", label="values as LINEAR gain")
ax.loglog(freq, 10 ** (gain / 20), "s--", label="values as dB -> linear")
ax.axhline(1.0, ls=":", color="gray")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Gain")
ax.set_title("Gain CSV: linear vs dB (compare to Peterka 5C)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
