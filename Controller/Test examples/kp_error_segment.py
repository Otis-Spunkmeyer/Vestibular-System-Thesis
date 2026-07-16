"""
kp_error_segment.py -- kp*error bilateral SNS segment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sns_neurons import Neuron, Synapse

# ====================================
# STEP 1: SNS PARAMETERS
# ====================================

Er = -60.0 # mV -- resting potential: V when no imput
R  =  20.0 # mV -- operating range: V goes from Er(silent) to Er+R (Fully active)
Gm =   1.0 # uS -- membrane conductance
Cm =   5.0 # nF -- membrance capacitance

Elo = Er     # -60.0 mV -- lower clamp of PWL synapse window
Ehi = Er + R # -40.0 mV -- upper clamp

gs_exc = 0.115 # uS -- excitatory sysnapse conducance
Es_exc = 134.0 # mV -- excitatory reversal potential

gs_inh = 0.558  # uS -- inhibatory synapse conductance
Es_inh = -100.0 # mV -- inhibitory reversal potential

I_APP_MOD = R * Gm # 20.0 # 20.0 nA -- baseline current for modulator neurons

DEG_TO_NA = 1.0 # nA per degree of body tilt

# ====================
# Test block
# ====================

if __name__ == "__main__":
    print("=== Step 1: parameter check ===")
    print(f" tau = Cm/Gm = {Cm/Gm:.1f} ms")
    print(f" Elo/Ehi ={Elo}/{Ehi} mV")
    print(f" I_APP_MOD = {I_APP_MOD} nA")
    print(f" 1 deg -> {1.0 * DEG_TO_NA / Gm:.1f} mV activation")
    print(f" 20 deg -> {20.0 * DEG_TO_NA / Gm:.1f} mV = full range R")
