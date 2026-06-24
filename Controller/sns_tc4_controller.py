"""
sns_tc4_controller.py — TC4 SNS Controller
Hilts 2018 neuron/synapse baseline + McNeal & Hunt 2026 architecture

Biological basis:    Peterka 2002, Test Condition 4 (TC4)
Neuron/synapse params: Hilts 2018, Tables A.1 and A.2
Controller layout:   McNeal & Hunt 2026 (split-derivative + dual-channel sensory)

TC4 conditions (Peterka 2002, Table 2, 1-degree PRTS amplitude):
  Visual:        eyes closed    → visual channel absent
  Surface:       PRTS stimulus  → proprioceptive channel partially unreliable
  Graviceptive:  veridical      → vestibular channel active
  Wp = 0.70,  Wg = 0.30

Error signal:  e = theta_ref - Wg * BS - Wp * (BS - SS)
Torque:        Ta = (Kp + Kd * s + Kc * |s| + Kt * T) * e(t - tau_d)

Network layout (46 neurons, 64 synapses):
  Stage 1 — Bilateral input          (6  neurons)
  Stage 2 — TC4 sensory integration  (8  neurons)  [McNeal & Hunt 2026 TC4 addition]
  Stage 3 — Derivative estimation    (8  neurons)  [split into accel/decel]
  Stage 4 — Co-activation |d(t)|     (1  neuron)
  Stage 5 — Bias-gated gain stages   (20 neurons)  Kp, Kd, Kc, Kt
  Stage 6 — Type-Ib input node       (1  neuron)
  Stage 7 — Bilateral motor output   (2  neurons)
"""

import numpy as np
import matplotlib.pyplot as plt

from sns_neurons import Neuron, Synapse
from prts_generator import generate_prts


# ====================================
# STEP 1: SNS PARAMETERS
# Source: Hilts 2018, Tables A.1 and A.2
# ====================================

# --- Base neuron physiology (Hilts Table A.1) ---
Er  = -60.0  # mV  resting potential: voltage when neuron has no input
R   =  20.0  # mV  operating range: V goes from Er (silent) to Er+R (fully active)
Gm  =   1.0  # µS  membrane conductance (all neurons)

"""
WHY these three constants?
Er, R, Gm define the SNS operating space. All arithmetic (addition, subtraction,
multiplication) is expressed in terms of these three values. Keeping them fixed
means every synapse conductance can be derived analytically from the desired
arithmetic behavior -- there are no free parameters in the synapse design.
"""

# --- Membrane capacitances: sets each neuron's time constant τ = Cm/Gm ---
# Hilts Table A.1 gives τ directly; Cm = τ × Gm.
Cm_standard =  1.0       # nF  τ = 1 ms    (most neurons: error, prod, bias, mod, coact)
Cm_input    = 20.0       # nF  τ = 20 ms   (Hilts neurons 1,2 — input/reference)
Cm_fast     =  0.1       # nF  τ = 0.1 ms  (Hilts neurons 4,17 — fast derivative tracker)
Cm_slow     =  8.0       # nF  τ = 8 ms    (Hilts neurons 5,18 — slow derivative integrator)
Cm_output   = 20.0       # nF  τ = 20 ms   (Hilts neurons 14,22 — motor output)
Cm_kt_prod  = 30_000.0   # nF  τ = 30,000 ms = 30 s  (Hilts neurons 25,26 — Kt integrator)

"""
WHY does each stage get a different time constant?
  Cm_input (20 ms):  Input neurons low-pass-filter the sensory signal, mimicking
                     the dynamics of vestibular hair cells and proprioceptors.

  Cm_fast (0.1 ms):  Must respond nearly instantaneously to track the error signal.
                     The derivative is computed as fast − slow; if fast also lagged,
                     there would be no meaningful difference to extract.

  Cm_slow (8 ms):    Creates an 8 ms lag behind the fast neuron. When the error
                     is rising, fast > slow → d_accel fires. When falling,
                     slow > fast → d_decel fires.

  Cm_output (20 ms): Motor output neurons smooth the sum of Kp+Kd+Kc+Kt products,
                     preventing transient spikes from reaching the muscle.

  Cm_kt_prod (30 s): The Kt*T neurons are SLOW INTEGRATORS. They accumulate tension
                     feedback over tens of seconds, representing the body's memory
                     of sustained loading rather than instantaneous force. Biologically,
                     this mimics the slow adaptation of Golgi tendon organs.
                     A 30-second time constant means a sudden change takes ~30 s
                     to fully register, providing a very gentle long-term correction.
"""

Elo = Er      # -60.0 mV  lower clamp of the piecewise-linear (PWL) synapse window
Ehi = Er + R  # -40.0 mV  upper clamp

# --- Standard addition synapse (Hilts Table A.2, ΔE = +194 mV, gs = 0.115 µS) ---
gs_exc = 0.115   # µS  excitatory synapse max conductance
Es_exc = 134.0   # mV  excitatory reversal potential  (Es = Er + ΔE = -60 + 194 = 134)

"""
Biological analogue: AMPA receptors (glutamate, excitatory).
Mixed Na+/K+ permeability gives real neurons a reversal near 0 mV.
The SNS uses +134 mV so that the driving force (Es - V) stays nearly constant
across the 20 mV operating range (changes < 10%), keeping Eq. 13 linear.
ΔE = Es - Er = 134 - (-60) = +194 mV  ✓  (Hilts Table A.2)
"""

# --- Standard subtraction synapse (Hilts Table A.2, ΔE = -40 mV, gs = 0.55775 µS) ---
gs_inh = 0.55775  # µS  inhibitory synapse max conductance
Es_inh = -100.0   # mV  inhibitory reversal potential  (Es = Er + ΔE = -60 + (-40) = -100)

"""
Biological analogue: GABA receptors (γ-aminobutyric acid, inhibitory).
Opens Cl-/K+ channels. Reversal set by Nernst equation for intracellular K+.
Es = -100 mV is always below the operating range [-60, -40], so the driving
force (Es - V) is always negative — consistently inhibitory.
ΔE = Es - Er = -100 - (-60) = -40 mV  ✓  (Hilts Table A.2)
"""

# --- Kp multiplication synapse (Hilts Table A.2, synapses 5 & 22, gs = 2.2 µS) ---
gs_kp_mult = 2.2    # µS  proportional signal conductance
Es_kp_mult = 134.0  # mV  excitatory reversal (ΔE = +194)

"""
WHY gs = 2.2 µS for Kp (not the standard 0.115 µS)?
The standard gs_exc=0.115 µS provides unity gain: U_post ≈ U_pre at steady state.
The Kp signal synapse needs ~19× more conductance to push kp_prod above the
threshold where the gating synapse (below) can be overcome. Without the boost,
the error signal would be too weak relative to the gate's inhibition.
"""

# --- Kd multiplication synapse (Hilts Table A.2, synapses 10-11 & 27-28, gs = 54 µS) ---
gs_kd_mult = 54.0   # µS  derivative signal conductance
Es_kd_mult = 134.0  # mV  excitatory reversal (ΔE = +194)

"""
WHY gs = 54 µS for Kd (much larger than Kp's 2.2 µS)?
The derivative signal (fast − slow) is inherently small: both fast and slow
track similar average values, and their difference is only a fraction of R.
A conductance of 54 µS (≈470× standard) amplifies this tiny differential
into a motor command comparable in magnitude to the proportional term.
Biologically, this mirrors high-gain amplification on the derivative channel
in PD controllers — derivative action is precise but inherently low-signal.
"""

# --- Gating synapse / Mult. Syn 2 (Hilts Table A.2, ΔE = 0, gs = 20 µS) ---
# Used in ALL multiplication circuits: kp_mod→kp_prod, kd_mod→kd_prod, etc.
gs_gate = 20.0   # µS  gating conductance
Es_gate = -60.0  # mV  gating reversal potential  (Es = Er, ΔE = 0)

"""
WHY Es = Er = -60 mV for the gating synapse (ΔE = 0)?
This is Hilts 2018's key innovation over standard inhibitory synapses.

Standard inhibition (Es = -100): provides a constant-strength floor that
pushes V toward -100 mV regardless of the post-synaptic neuron's state.

Gating synapse (Es = Er = -60, ΔE = 0): the driving force is:
    I_gate = gs_gate × activation_pre × (Es_gate - V_post)
           = gs_gate × activation_pre × (Er - V_post)
           = gs_gate × activation_pre × (-U_post)

When V_post = Er (U=0, silent): driving force = 0 → no current.
When V_post > Er (U>0, active): driving force is negative → inhibitory.

The gate is SELF-CALIBRATING: it pushes back exactly in proportion to how
active the product neuron already is. This enables true multiplication because
the output stabilizes where signal excitation equals gate inhibition, giving
U_prod ∝ U_signal × U_mod / R rather than a clipped, saturated response.
"""

# --- Operating-point bias for modulator neurons ---
I_APP_MOD = R * Gm  # 20.0 nA  holds all k*_mod neurons at U = R (fully gating)

"""
WHY exactly R × Gm = 20 nA?
At steady state with no synaptic input:
    Gm × (Er - V) + I_app = 0  →  V = Er + I_app/Gm  →  U = I_app/Gm
Setting I_app = R × Gm makes U = R = 20 mV — the top of the operating range.
The mod neuron then sits FULLY ACTIVE, maximally inhibiting the prod neuron.
The bias neuron's inhibition then pulls mod DOWN from R toward 0, opening
the gate proportionally. Gain scales with bias current.
"""

# --- Hilts 2018 baseline bias currents (Table A.1, I_app column) ---
BP_DEFAULT = 12.5  # nA  Kp bias   (Hilts neuron 8;  McNeal & Hunt PSO: 4.26 nA)
BD_DEFAULT = 13.5  # nA  Kd bias   (Hilts neuron 11; McNeal & Hunt PSO: 5.01 nA)
BC_DEFAULT =  2.48 # nA  Kc bias   (no Hilts equivalent — McNeal & Hunt 2026 value retained)
BT_DEFAULT =  0.8  # nA  Kt bias   (Hilts neuron 23; McNeal & Hunt PSO: 5.42 nA)

"""
WHY do Kp/Kd/Kt biases differ from McNeal & Hunt PSO values?
The PSO was run with different synapse conductances (all standard gs_exc/gs_inh).
With Hilts 2018 multiplication conductances (gs_kp=2.2, gs_kd=54, gs_gate=20),
the gain-vs-bias relationship changes entirely. These Hilts baseline values are
the physiologically-motivated starting point BEFORE optimization.
Kc has no Hilts equivalent (split derivative is a McNeal & Hunt addition),
so the PSO value is retained as the starting point.
"""

# --- Angle encoding ---
DEG_TO_NA = 1.0  # nA per degree  (1 degree maps to 1 nA of applied current)

"""
WHY 1 nA/degree?
With Gm = 1 µS and I_app = theta_deg × DEG_TO_NA, the steady-state voltage is:
    V = Er + I_app/Gm = -60 + theta_deg
So U = V - Er = theta_deg (degrees = millivolts directly).
The operating range R = 20 mV then corresponds to ±10 degrees of tilt,
which covers the physiological range for quiet standing.
"""

# --- TC4 sensory weights (Peterka 2002, Table 2, 1-degree PRTS) ---
GRAVICEPTIVE_WEIGHT   = 0.30  # Wg
PROPRIOCEPTIVE_WEIGHT = 0.70  # Wp


def _gs_for_gain(target_gain):
    """
    Solve for gs_max so that U_post ≈ target_gain × U_pre at U_pre = R/2.

    At steady state with one excitatory synapse:
        0 = Gm*(Er-V) + gs*(V_pre-Elo)/R * (Es_exc-V)
    Substituting U = V - Er, U_pre = R/2:
        gs = target_gain * Gm * R / (ΔE_exc - target_gain * R/2)

    WHY evaluate at U_pre = R/2 (the midpoint)?
    The SNS synapse is piecewise-linear, so this midpoint gives the best
    average approximation of the gain across the full operating range.
    """
    delta_E = Es_exc - Er  # 194 mV
    return (target_gain * Gm * R) / (delta_E - target_gain * (R / 2.0))


GS_WG = _gs_for_gain(GRAVICEPTIVE_WEIGHT)    # ≈ 0.0314 µS  (gain = 0.30)
GS_WP = _gs_for_gain(PROPRIOCEPTIVE_WEIGHT)  # ≈ 0.0749 µS  (gain = 0.70)


# ====================================
# STEP 2: NEURONS
# 46 total — mapped from Hilts 2018 Table A.1 neuron numbering
# ====================================

def build_neurons():
    """
    Construct all 46 neurons of the TC4 SNS controller.

    Returns an ordered dict: name → Neuron object.
    Each neuron starts at V = Er (silent, U = 0).

    Hilts 2018 Table A.1 neuron → TC4 mapping:
        Neurons 1,2   → bs/ss/theta_ref input group  (Er=-60, Cm=20 nF)
        Neurons 4,17  → deriv_fast_ccw/cw             (Er=-60, Cm=0.1 nF)
        Neurons 5,18  → deriv_slow_ccw/cw             (Er=-60, Cm=8 nF)
        Neuron  8     → kp_bias                        (I_app=12.5 nA)
        Neurons 9,12,24 → kp/kd/kt_mod               (I_app=20 nA)
        Neuron  11    → kd_bias                        (I_app=13.5 nA)
        Neurons 14,22 → ta_ccw/ta_cw                  (Er=-60, Cm=20 nF)
        Neuron  23    → kt_bias                        (I_app=0.8 nA)
        Neurons 25,26 → kt_prod_ccw/cw                (Er=-60, Cm=30,000 nF)
    """

    # Local closure helpers — one per Cm variant from Hilts Table A.1.
    # WHY closures? They capture Er and Gm from the enclosing scope automatically.
    # 'N()' only makes sense here (inside build_neurons), so keeping it local
    # avoids polluting the module namespace with a function that has no
    # standalone meaning. This pattern is called a "factory closure."
    def N(name):        return Neuron(Cm_nF=Cm_standard, Gm_uS=Gm, Er_mV=Er, name=name)
    def N_in(name):     return Neuron(Cm_nF=Cm_input,    Gm_uS=Gm, Er_mV=Er, name=name)
    def N_fast(name):   return Neuron(Cm_nF=Cm_fast,     Gm_uS=Gm, Er_mV=Er, name=name)
    def N_slow(name):   return Neuron(Cm_nF=Cm_slow,     Gm_uS=Gm, Er_mV=Er, name=name)
    def N_out(name):    return Neuron(Cm_nF=Cm_output,   Gm_uS=Gm, Er_mV=Er, name=name)
    def N_ktp(name):    return Neuron(Cm_nF=Cm_kt_prod,  Gm_uS=Gm, Er_mV=Er, name=name)

    return {

        # -------- Stage 1: Bilateral input neurons (6) --------
        # WHY Cm_input (τ = 20 ms)?
        # Hilts Table A.1, neurons 1 and 2. The 20 ms time constant low-pass
        # filters sensory signals, mimicking transduction delays in vestibular
        # hair cells and ankle proprioceptors. It also smooths the transition
        # when the bilateral split switches from CCW-active to CW-active.
        #
        # WHY bilateral split (ccw + cw) instead of signed input?
        # SNS voltages are always ≥ Er. A single neuron cannot encode a
        # negative body sway. Splitting into CCW (positive half) and CW
        # (negative-magnitude half) keeps both non-negative.
        # Only one branch is active at any instant; the error circuit
        # downstream reconstructs the signed error from which branch fires.
        "bs_ccw":        N_in("bs_ccw"),        # body sway, CCW half
        "bs_cw":         N_in("bs_cw"),         # body sway, CW half
        "ss_ccw":        N_in("ss_ccw"),        # surface sway, CCW half
        "ss_cw":         N_in("ss_cw"),         # surface sway, CW half
        "theta_ref_ccw": N_in("theta_ref_ccw"), # reference angle, CCW half
        "theta_ref_cw":  N_in("theta_ref_cw"),  # reference angle, CW half

        # -------- Stage 2: TC4 dual-channel sensory integration (8) --------
        # McNeal & Hunt 2026 addition over Hilts 2018.
        # Hilts uses a single-channel error: e = theta_A - theta_ref.
        # TC4 replaces this with a two-channel weighted sum:
        #   e = theta_ref + Wg × BS  +  Wp × (BS - SS)
        # implemented neurally as three sub-stages:
        #   sub_diff = BS - SS              (subtraction subnetwork: EXC + INH)
        #   wg_node  = Wg × BS             (transmission synapse with gs=GS_WG)
        #   wp_node  = Wp × sub_diff       (transmission synapse with gs=GS_WP)
        #   error    = theta_ref + wg + wp  (three EXC inputs)
        #
        # WHY weighted sensory channels?
        # During platform rotation (TC4), the ankle senses rotation relative
        # to the surface — proprioception partially detects a "sway" that is
        # really just the platform moving. The graviceptive channel (Wg × BS)
        # uses vestibular information about body-in-space orientation, which is
        # unaffected by platform motion. Weighting Wg=0.30 and Wp=0.70 gives
        # the brain's best estimate of true body orientation (Peterka 2002).
        "sub_diff_ccw": N("sub_diff_ccw"),   # BS - SS, CCW (proprioceptive error)
        "sub_diff_cw":  N("sub_diff_cw"),    # BS - SS, CW
        "wg_ccw":       N("wg_ccw"),         # Wg × BS, CCW (graviceptive channel)
        "wg_cw":        N("wg_cw"),          # Wg × BS, CW
        "wp_ccw":      N("wp_ccw"),         # Wp × (BS-SS), CCW (proprioceptive channel)
        "wp_cw":        N("wp_cw"),          # Wp × (BS-SS), CW
        "error_ccw":    N("error_ccw"),      # total TC4 error, CCW
        "error_cw":     N("error_cw"),       # total TC4 error, CW

        # -------- Stage 3: Derivative estimation (8) --------
        # Hilts Table A.1: fast neurons (4,17) τ=0.1 ms, slow neurons (5,18) τ=8 ms.
        #
        # WHY two different Cm values?
        # Both fast and slow neurons receive the SAME excitatory input from error.
        # Their different time constants create a lag:
        #   fast tracks the input almost instantaneously (τ=0.1 ms)
        #   slow integrates, lagging ~8 ms behind the fast neuron
        # When error is RISING:  V_fast > V_slow  →  d_accel fires (fast - slow > 0)
        # When error is FALLING: V_slow > V_fast  →  d_decel fires (slow - fast > 0)
        #
        # WHY split into d_accel AND d_decel? (McNeal & Hunt 2026 extension)
        # A single signed derivative (fast - slow) goes below Er when the signal
        # is falling — and SNS voltages floor at Er, losing the deceleration signal.
        # Splitting preserves both halves as non-negative channels:
        #   d_accel → Kd excitatory: drives output when error growing (bad)
        #   d_decel → Kd inhibitory: suppresses output when error shrinking (returning)
        # Hilts 2018 does not include d_decel — that split is McNeal & Hunt's addition.
        "deriv_fast_ccw": N_fast("deriv_fast_ccw"),  # fast tracker, CCW  (Hilts neuron 4)
        "deriv_slow_ccw": N_slow("deriv_slow_ccw"),  # slow integrator, CCW (Hilts neuron 5)
        "d_accel_ccw":    N("d_accel_ccw"),           # fast-slow, CCW (accelerating)
        "d_decel_ccw":    N("d_decel_ccw"),           # slow-fast, CCW (decelerating)
        "deriv_fast_cw":  N_fast("deriv_fast_cw"),   # fast tracker, CW  (Hilts neuron 17)
        "deriv_slow_cw":  N_slow("deriv_slow_cw"),   # slow integrator, CW (Hilts neuron 18)
        "d_accel_cw":     N("d_accel_cw"),
        "d_decel_cw":     N("d_decel_cw"),

        # -------- Stage 4: Co-activation node (1) --------
        # WHY a separate co-activation node?
        # The Kd pathway uses a SIGNED derivative: it drives correction when error
        # is growing and suppresses it when error is shrinking. But even a shrinking
        # error requires joint stiffness to prevent overshoot and oscillation.
        # The Kc pathway receives |d| = d_accel + d_decel (all positive, unsigned)
        # and co-contracts BOTH muscles simultaneously, stiffening the ankle
        # regardless of which direction the error is moving.
        # This implements x_c(t) = |d(t)| from McNeal & Hunt Eq. 1.
        "coact_node": N("coact_node"),

        # -------- Stage 5: Bias-gated gain stages (20 neurons) --------
        # Each of the four gain pathways (Kp, Kd, Kc, Kt) has three layers:
        #
        #   k*_bias  — single shared neuron; receives I_app = b_alpha (tunable).
        #              Hilts Table A.1: kp_bias=12.5 nA, kd_bias=13.5 nA, kt_bias=0.8 nA.
        #
        #   k*_mod   — bilateral (ccw/cw); receives I_app = R×Gm = 20 nA (constant).
        #              Sits at U = R = 20 mV (fully active, fully gating prod).
        #              bias  inhibits mod, pulling it down from R. More bias = less mod.
        #
        #   k*_prod  — bilateral (ccw/cw); receives signal (EXC) + mod (GATE).
        #              Output: signal gets through proportionally to how much
        #              the bias has reduced mod. Net: prod ≈ Gain × signal.
        #
        # WHY double inhibition instead of direct multiplication?
        # Biological neurons cannot directly multiply two voltages.
        # The double-inhibition trick creates multiplication through:
        #   1. bias  → (inhibits) → mod    (mod drops as bias rises)
        #   2. mod   → (gates)   → prod   (prod rises as mod drops)
        # Combined: prod ∝ bias × signal — a gain-controlled amplifier.

        # Kp pathway: proportional to error (Hilts neuron 8 → 9 → 10/16)
        "kp_bias":    N("kp_bias"),     # Hilts neuron 8,  I_app=12.5 nA
        "kp_mod_ccw": N("kp_mod_ccw"), # Hilts neuron 9,  I_app=20 nA
        "kp_mod_cw":  N("kp_mod_cw"),
        "kp_prod_ccw": N("kp_prod_ccw"), # output: Kp × error_ccw
        "kp_prod_cw":  N("kp_prod_cw"),  # output: Kp × error_cw

        # Kd pathway: proportional to signed derivative (Hilts neuron 11 → 12 → 13/21)
        "kd_bias":    N("kd_bias"),     # Hilts neuron 11, I_app=13.5 nA
        "kd_mod_ccw": N("kd_mod_ccw"), # Hilts neuron 12, I_app=20 nA
        "kd_mod_cw":  N("kd_mod_cw"),
        "kd_prod_ccw": N("kd_prod_ccw"), # output: Kd × d_accel_ccw − Kd × d_decel_ccw
        "kd_prod_cw":  N("kd_prod_cw"),

        # Kc pathway: proportional to |d| (McNeal & Hunt addition; no Hilts equivalent)
        "kc_bias":    N("kc_bias"),     # I_app=BC_DEFAULT (McNeal & Hunt value retained)
        "kc_mod_ccw": N("kc_mod_ccw"), # I_app=20 nA
        "kc_mod_cw":  N("kc_mod_cw"),
        "kc_prod_ccw": N("kc_prod_ccw"), # output: Kc × |d|
        "kc_prod_cw":  N("kc_prod_cw"),

        # Kt pathway: proportional to muscle tension T (Hilts neuron 23 → 24 → 25/26)
        "kt_bias":    N("kt_bias"),     # Hilts neuron 23, I_app=0.8 nA
        "kt_mod_ccw": N("kt_mod_ccw"), # Hilts neuron 24, I_app=20 nA
        "kt_mod_cw":  N("kt_mod_cw"),
        "kt_prod_ccw": N_ktp("kt_prod_ccw"), # Hilts neuron 25, Cm=30,000 nF (30 s integrator)
        "kt_prod_cw":  N_ktp("kt_prod_cw"),  # Hilts neuron 26

        # -------- Stage 6: Type-Ib tension input (1) --------
        # Receives I_app from muscle tension feedback each timestep.
        # In dynamic simulation: I_Ib = clip(gIb × T_Hill, 0, 10 nA)
        # In standalone mode:    tension_to_ib_current_na(proxy)
        "ib_input": N("ib_input"),

        # -------- Stage 7: Bilateral motor output (2) --------
        # WHY Cm_output (τ = 20 ms)? Hilts Table A.1, neurons 14 and 22.
        # The output neurons integrate all four gain pathways (Kp+Kd+Kc+Kt).
        # A 20 ms time constant smooths transient spikes from the product neurons,
        # producing a clean muscle activation signal. Biologically, motor neuron
        # membranes have low-pass filter properties from their own capacitance.
        "ta_ccw": N_out("ta_ccw"),  # CCW (extensor) muscle command
        "ta_cw":  N_out("ta_cw"),   # CW  (flexor)   muscle command
    }


# ====================================
# STEP 3: SYNAPSES
# 64 total — numbered syn1…syn64 following Hilts 2018 Figure 3.1 convention
# adapted for McNeal & Hunt 2026 TC4 architecture
# ====================================

def build_synapses():
    """
    Construct all 64 synapses of the TC4 SNS controller.
    Every synapse is numbered; the diagram labels each arrow with its number.

    Returns a list of Synapse objects in syn1…syn64 order.

    Hilts 2018 Table A.2 synapse types used here:
        Addition     (ΔE=+194, gs=0.115):  most signal-routing connections
        Subtraction  (ΔE=-40,  gs=0.55775): inhibitory connections (BS-SS, d_accel, etc.)
        Mult. Kp     (ΔE=+194, gs=2.2):    error → kp_prod
        Mult. Kd     (ΔE=+194, gs=54):     d_accel → kd_prod
        Mult. Syn 2  (ΔE=0,    gs=20):     ALL mod → prod gating connections
    """

    # Local helpers — each encodes one Hilts Table A.2 synapse type.
    # 'n' is the synapse number; embedded in the name for diagram labeling.
    # WHY named synapses? The diagram function splits "syn5_..." on '_' to get
    # the label "syn5". Consistent naming here means every arrow is traceable
    # back to exactly one line of code.

    def Exc(pre, post, n):
        """Standard addition: gs=0.115, ΔE=+194 (Hilts Table A.2)."""
        return Synapse(gs_exc, Es_exc, Elo, Ehi,
                       name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post)

    def Inh(pre, post, n):
        """Standard subtraction: gs=0.55775, ΔE=-40 (Hilts Table A.2)."""
        return Synapse(gs_inh, Es_inh, Elo, Ehi,
                       name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post)

    def Trn(pre, post, n, gs):
        """Transmission with custom conductance (Wg or Wp gain, derived by _gs_for_gain)."""
        return Synapse(gs, Es_exc, Elo, Ehi,
                       name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post)

    def KpMult(pre, post, n):
        """Kp signal synapse: gs=2.2 µS, ΔE=+194 (Hilts Table A.2, synapses 5 & 22)."""
        return Synapse(gs_kp_mult, Es_kp_mult, Elo, Ehi,
                       name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post)

    def KdMult(pre, post, n):
        """Kd signal synapse: gs=54 µS, ΔE=+194 (Hilts Table A.2, synapses 10-11 & 27-28)."""
        return Synapse(gs_kd_mult, Es_kd_mult, Elo, Ehi,
                       name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post)

    def Gate(pre, post, n):
        """Gating / Mult. Syn 2: gs=20 µS, Es=-60 (ΔE=0) (Hilts Table A.2, syn 14-17, 31-34, 38)."""
        return Synapse(gs_gate, Es_gate, Elo, Ehi,
                       name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post)

    return [

        # ================================================================
        # Stage 1 → Stage 2: Sensory input to subtraction & transmission
        # (Hilts Figure 3.1 synapses 1-9 region)
        # ================================================================
        # WHY EXC + INH for subtraction?
        # SNS voltages can never go below Er — you cannot negate a signal directly.
        # Subtraction uses EXC(pre_A → post) + INH(pre_B → post):
        #   When pre_A > pre_B: net excitation drives post above Er (positive result).
        #   When pre_A < pre_B: net inhibition clamps post at Er (zero, not negative).
        # This is natural rectification: sub_diff only shows the positive part of (BS-SS).
        # The CW channel's sub_diff_cw shows the positive part of (SS-BS), symmetrically.
        Exc("bs_ccw", "sub_diff_ccw", 1),   # syn1:  BS excites sub_diff (adds BS)
        Inh("ss_ccw", "sub_diff_ccw", 2),   # syn2:  SS inhibits sub_diff (subtracts SS)
        Exc("bs_cw",  "sub_diff_cw",  3),   # syn3:  CW mirror
        Inh("ss_cw",  "sub_diff_cw",  4),   # syn4:  CW mirror

        # Graviceptive channel: wg = Wg × BS (gain = 0.30)
        # WHY GS_WG ≠ gs_exc?  Standard gs_exc gives unity gain: U_post ≈ U_pre.
        # We want U_wg ≈ 0.30 × U_bs. _gs_for_gain(0.30) solves for gs = 0.0314 µS.
        Trn("bs_ccw", "wg_ccw", 5, GS_WG),  # syn5:  Wg × BS_ccw
        Trn("bs_cw",  "wg_cw",  6, GS_WG),  # syn6:  Wg × BS_cw

        # Proprioceptive channel: wp = Wp × sub_diff (gain = 0.70)
        Trn("sub_diff_ccw", "wp_ccw", 7, GS_WP),  # syn7:  Wp × sub_diff_ccw
        Trn("sub_diff_cw",  "wp_cw",  8, GS_WP),  # syn8:  Wp × sub_diff_cw

        # ================================================================
        # Stage 2: Error node formation
        # Signed error: e = theta_ref - Wg*BS - Wp*(BS-SS)  (Peterka 2002)
        # Bilateral SNS implementation:
        #   error_ccw fires when (theta_ref - sensory_estimate) is CCW-positive
        #   error_cw  fires when (theta_ref - sensory_estimate) is CW-positive
        # ================================================================
        # WHY three EXC inputs to error?
        # In the bilateral architecture, theta_ref_ccw only has current when
        # theta_ref is positive (CCW). The sensory terms (wg_ccw, wp_ccw) only
        # have current when the BODY leans CCW. The error neuron fires when
        # theta_ref exceeds the sensory estimate — i.e., more reference than
        # sensory, which drives corrective CCW muscle output.
        # Within each bilateral channel all three inputs are excitatory because
        # each has already been split to carry only its non-negative half.
        # The three terms implement the CCW channel of: e = theta_ref - Wg*BS - Wp*(BS-SS)
        Exc("theta_ref_ccw", "error_ccw",  9),   # syn9:  adds theta_ref
        Exc("wg_ccw",        "error_ccw", 10),   # syn10: adds graviceptive component
        Exc("wp_ccw",        "error_ccw", 11),   # syn11: adds proprioceptive component
        Exc("theta_ref_cw",  "error_cw",  12),   # syn12
        Exc("wg_cw",         "error_cw",  13),   # syn13
        Exc("wp_cw",         "error_cw",  14),   # syn14

        # ================================================================
        # Stage 2 → Stage 3: Error to fast and slow derivative neurons
        # Both receive IDENTICAL excitatory input — Cm alone creates the lag.
        # ================================================================
        # WHY same synapse (gs_exc) for both fast and slow?
        # The derivative is computed from the VOLTAGE DIFFERENCE between the two
        # neurons, not from different input strengths. The time constant (Cm/Gm)
        # creates the lag: fast reaches steady state in ~0.1 ms, slow in ~8 ms.
        # During a fast change, fast voltage leads slow voltage → d_accel fires.
        Exc("error_ccw", "deriv_fast_ccw", 15),  # syn15
        Exc("error_ccw", "deriv_slow_ccw", 16),  # syn16
        Exc("error_cw",  "deriv_fast_cw",  17),  # syn17
        Exc("error_cw",  "deriv_slow_cw",  18),  # syn18

        # ================================================================
        # Stage 3: Derivative subtraction
        # d_accel = fast − slow  (positive when error ACCELERATING away from upright)
        # d_decel = slow − fast  (positive when error DECELERATING, returning)
        # Hilts Table A.2: subtraction synapses 7,8,18,26 → gs=0.55775, ΔE=-40
        # ================================================================
        Exc("deriv_fast_ccw", "d_accel_ccw", 19),  # syn19: +fast
        Inh("deriv_slow_ccw", "d_accel_ccw", 20),  # syn20: -slow  → accel = fast-slow
        Exc("deriv_slow_ccw", "d_decel_ccw", 21),  # syn21: +slow
        Inh("deriv_fast_ccw", "d_decel_ccw", 22),  # syn22: -fast  → decel = slow-fast
        Exc("deriv_fast_cw",  "d_accel_cw",  23),  # syn23
        Inh("deriv_slow_cw",  "d_accel_cw",  24),  # syn24
        Exc("deriv_slow_cw",  "d_decel_cw",  25),  # syn25
        Inh("deriv_fast_cw",  "d_decel_cw",  26),  # syn26

        # ================================================================
        # Stage 3 → Stage 4: Derivative outputs to co-activation node
        # coact = d_accel_ccw + d_accel_cw + d_decel_ccw + d_decel_cw ≈ |d(t)|
        # ================================================================
        # All four are non-negative (rectified half-waves), so their sum is the
        # unsigned derivative magnitude — the co-activation signal |d|.
        Exc("d_accel_ccw", "coact_node", 27),  # syn27
        Exc("d_accel_cw",  "coact_node", 28),  # syn28
        Exc("d_decel_ccw", "coact_node", 29),  # syn29
        Exc("d_decel_cw",  "coact_node", 30),  # syn30

        # ================================================================
        # Stage 5: Kp gain pathway (double-inhibition multiplication)
        #
        # Circuit:
        #   kp_bias -[INH]→ kp_mod -[GATE]→ kp_prod ←[KpMult]- error
        #
        # How it produces Kp × error:
        #   1. kp_bias inhibits kp_mod  → mod drops from R toward 0 as bias rises
        #   2. error excites kp_prod with gs=2.2 (strong drive)
        #   3. kp_mod gates kp_prod with ΔE=0 (voltage-dependent suppression)
        #   Net: kp_prod ≈ error × (1 - kp_mod/R) ∝ bias × error = Kp × error
        #
        # Hilts Table A.2: signal synapses 5,22 → gs=2.2 (Kp mult)
        #                   gate synapses 14,15 → gs=20, ΔE=0 (Mult. Syn 2)
        # ================================================================
        Inh("kp_bias",     "kp_mod_ccw",   31),  # syn31: bias inhibits mod (opens gate)
        Inh("kp_bias",     "kp_mod_cw",    32),  # syn32
        KpMult("error_ccw", "kp_prod_ccw", 33),  # syn33: error→prod  (Hilts gs=2.2)
        Gate("kp_mod_ccw",  "kp_prod_ccw", 34),  # syn34: mod gates prod (Hilts gs=20, ΔE=0)
        KpMult("error_cw",  "kp_prod_cw",  35),  # syn35
        Gate("kp_mod_cw",   "kp_prod_cw",  36),  # syn36

        # ================================================================
        # Stage 5: Kd gain pathway (directional derivative)
        #
        # Circuit:
        #   kd_bias -[INH]→ kd_mod -[GATE]→ kd_prod ←[KdMult]- d_accel
        #                                    kd_prod ←[INH]    - d_decel
        #
        # d_accel drives kd_prod UP (gs=54): error is GROWING, need more correction
        # d_decel drives kd_prod DOWN (gs_inh): error is SHRINKING, ease off
        # This creates a signed derivative: positive when accelerating, suppressed
        # when decelerating → prevents overshoot.
        #
        # WHY gs=54 for KdMult? See Step 1 comment. Short answer: the derivative
        # signal (fast-slow) is tiny; high conductance amplifies it to useful size.
        #
        # Hilts Table A.2: signal synapses 10-11,27-28 → gs=54 (Kd mult)
        #                   gate synapses 16,17 → gs=20, ΔE=0 (Mult. Syn 2)
        # ================================================================
        Inh("kd_bias",      "kd_mod_ccw",   37),  # syn37
        Inh("kd_bias",      "kd_mod_cw",    38),  # syn38
        KdMult("d_accel_ccw", "kd_prod_ccw", 39), # syn39: accel→prod (Hilts gs=54)
        Inh("d_decel_ccw",    "kd_prod_ccw", 40), # syn40: decel suppresses prod
        Gate("kd_mod_ccw",    "kd_prod_ccw", 41), # syn41: mod gates prod
        KdMult("d_accel_cw",  "kd_prod_cw",  42), # syn42
        Inh("d_decel_cw",     "kd_prod_cw",  43), # syn43
        Gate("kd_mod_cw",     "kd_prod_cw",  44), # syn44

        # ================================================================
        # Stage 5: Kc gain pathway (co-activation, unsigned |d|)
        # McNeal & Hunt 2026 addition — no Hilts 2018 equivalent.
        #
        # Uses KpMult conductance (gs=2.2) since coact_node has amplitude
        # similar to error. Uses same Gate (ΔE=0, gs=20) for consistency.
        # bc_default kept at McNeal & Hunt PSO value (2.48 nA) since there
        # is no Hilts Table A.1 entry for this pathway.
        # ================================================================
        Inh("kc_bias",      "kc_mod_ccw",   45),  # syn45
        Inh("kc_bias",      "kc_mod_cw",    46),  # syn46
        KpMult("coact_node", "kc_prod_ccw", 47),  # syn47: |d|→prod (Kp-style gs=2.2)
        Gate("kc_mod_ccw",   "kc_prod_ccw", 48),  # syn48
        KpMult("coact_node", "kc_prod_cw",  49),  # syn49
        Gate("kc_mod_cw",    "kc_prod_cw",  50),  # syn50

        # ================================================================
        # Stage 5: Kt gain pathway (Type-Ib tension feedback)
        #
        # Circuit:
        #   kt_bias -[INH]→ kt_mod -[GATE]→ kt_prod ←[EXC]- ib_input
        #
        # WHY standard Exc (gs=0.115) for ib_input → kt_prod, not KpMult?
        # In Hilts Table A.2, tension synapses are in the standard addition group
        # (gs=0.115). The slow integration of kt_prod (Cm=30,000 nF, τ=30 s)
        # provides the effective amplification instead of high conductance.
        # A standard synapse feeding a 30-second integrator accumulates the
        # signal over time to produce a sustained, slowly-varying tension signal.
        #
        # Hilts Table A.2: tension synapses 35,40 → standard addition gs=0.115
        #                   gate synapses 31-34,38 → gs=20, ΔE=0 (Mult. Syn 2)
        # ================================================================
        Inh("kt_bias",   "kt_mod_ccw",   51),  # syn51
        Inh("kt_bias",   "kt_mod_cw",    52),  # syn52
        Exc("ib_input",  "kt_prod_ccw",  53),  # syn53: tension→prod (standard gs=0.115)
        Gate("kt_mod_ccw", "kt_prod_ccw", 54), # syn54
        Exc("ib_input",  "kt_prod_cw",   55),  # syn55
        Gate("kt_mod_cw",  "kt_prod_cw",  56), # syn56

        # ================================================================
        # Stage 5 → Stage 7: All gain products to motor output
        # ta = kp_prod + kd_prod + kc_prod + kt_prod  (four EXC inputs summed)
        # ================================================================
        # WHY all excitatory?
        # Each product neuron only fires on its appropriate side (bilateral split).
        # kp_prod_ccw is active only when the body leans CCW — it should drive
        # the CCW (extensor) muscle. ta_ccw sums all four pathway contributions
        # for that side. No inhibitory connections needed because the wrong-side
        # product neurons are already silent (clamped at Er by the bilateral split).
        Exc("kp_prod_ccw", "ta_ccw", 57),  # syn57
        Exc("kd_prod_ccw", "ta_ccw", 58),  # syn58
        Exc("kc_prod_ccw", "ta_ccw", 59),  # syn59
        Exc("kt_prod_ccw", "ta_ccw", 60),  # syn60
        Exc("kp_prod_cw",  "ta_cw",  61),  # syn61
        Exc("kd_prod_cw",  "ta_cw",  62),  # syn62
        Exc("kc_prod_cw",  "ta_cw",  63),  # syn63
        Exc("kt_prod_cw",  "ta_cw",  64),  # syn64
    ]


# ====================================
# STEP 4: APPLIED CURRENTS (static analysis)
# ====================================

def build_iapp(bs_deg, ss_deg=0.0, theta_ref_deg=0.0,
               bp=BP_DEFAULT, bd=BD_DEFAULT,
               bc=BC_DEFAULT, bt=BT_DEFAULT,
               ib_na=0.0):
    """
    Build the applied-current dictionary for a static steady-state analysis.

    All angles are in degrees (converted to nA by DEG_TO_NA = 1 nA/deg).

    WHY bilateral split in build_iapp (not THETA_BIAS)?
    In kp_error_segment.py we added THETA_BIAS to keep single input neurons
    in the active range for negative angles. Here we split instead:
      bs_ccw = max(0, bs_deg) * DEG_TO_NA   (fires when body leans CCW)
      bs_cw  = max(0, -bs_deg) * DEG_TO_NA  (fires when body leans CW)
    Each input neuron only ever receives non-negative current.
    The error circuit downstream reconstructs the signed error from which
    branch is active. No offset correction needed for display — U = angle.
    """

    def split(deg):
        """Split a signed degree value into (ccw_nA, cw_nA), both non-negative."""
        i = deg * DEG_TO_NA
        return max(0.0, i), max(0.0, -i)

    bs_ccw_na,  bs_cw_na  = split(bs_deg)
    ss_ccw_na,  ss_cw_na  = split(ss_deg)
    ref_ccw_na, ref_cw_na = split(theta_ref_deg)

    return {
        # Stage 1: sensory inputs (bilateral split)
        "bs_ccw":        bs_ccw_na,
        "bs_cw":         bs_cw_na,
        "ss_ccw":        ss_ccw_na,
        "ss_cw":         ss_cw_na,
        "theta_ref_ccw": ref_ccw_na,
        "theta_ref_cw":  ref_cw_na,

        # Stage 6: Type-Ib tension feedback
        "ib_input": ib_na,

        # Stage 5 bias neurons: tunable gain parameters (b_alpha values)
        # These are the ONLY parameters tuned by PSO in the full simulation.
        "kp_bias": bp,  # Kp gain  (Hilts baseline: 12.5 nA)
        "kd_bias": bd,  # Kd gain  (Hilts baseline: 13.5 nA)
        "kc_bias": bc,  # Kc gain  (McNeal & Hunt:   2.48 nA)
        "kt_bias": bt,  # Kt gain  (Hilts baseline:  0.8 nA)

        # Stage 5 modulator neurons: constant at U=R (fully gating until bias pulls down)
        # WHY separate from bias? The mod constant is always R×Gm = 20 nA regardless
        # of gain setting. Changing bp only changes kp_bias I_app — mod is unchanged.
        # This decoupling makes PSO tuning cleaner: one parameter per pathway.
        "kp_mod_ccw": I_APP_MOD,
        "kp_mod_cw":  I_APP_MOD,
        "kd_mod_ccw": I_APP_MOD,
        "kd_mod_cw":  I_APP_MOD,
        "kc_mod_ccw": I_APP_MOD,
        "kc_mod_cw":  I_APP_MOD,
        "kt_mod_ccw": I_APP_MOD,
        "kt_mod_cw":  I_APP_MOD,
    }


# ====================================
# STEP 5: SIMULATION LOOP (static analysis)
# ====================================

def run_segment(neurons, synapses, I_app_dict, dt_ms=0.1, steps=2000):
    """
    Run the SNS network to steady state using forward Euler integration.

    Args:
        neurons:     dict from build_neurons() (modified in place)
        synapses:    list from build_synapses()
        I_app_dict:  dict from build_iapp() — fixed applied currents
        dt_ms:       timestep in milliseconds
        steps:       number of integration steps

    Returns:
        log: dict mapping neuron name → np.array of voltage over time

    NOTE: This is a STATIC analysis (fixed I_app). For dynamic simulation
          with time-varying inputs (PRTS stimulus, MuJoCo), use step_controller().
    """

    # Pre-allocate voltage log.
    # WHY np.zeros(steps) rather than appending to a list?
    # Appending copies the entire array each time — O(n²) total memory ops.
    # np.zeros reserves all memory upfront; writing by index is O(1) per step.
    log = {name: np.zeros(steps) for name in neurons}

    for step in range(steps):

        # ---- Phase A: compute ALL synaptic currents from CURRENT voltages ----
        # WHY compute all before updating any?
        # All neurons in biological circuits update simultaneously — they don't
        # "take turns." If we updated neuron A first, neuron B would see A's NEW
        # voltage when computing its own dV/dt, violating simultaneity.
        # Phase A READS all current voltages; Phase C WRITES all new voltages.
        # Never mix the two within a single timestep.
        I_syn = {name: 0.0 for name in neurons}
        for syn in synapses:
            I_syn[syn.post] += syn.current(neurons[syn.pre].V, neurons[syn.post].V)

        # ---- Phase B: compute dV/dt for ALL neurons (do not write V yet) ----
        dV = {}
        for name, nrn in neurons.items():
            I_app  = I_app_dict.get(name, 0.0)
            I_leak = nrn.Gm * (nrn.Er - nrn.V)
            # Kirchhoff current law:  Cm × dV/dt = I_leak + I_syn + I_app
            # I_leak = Gm × (Er - V): pulls voltage back to rest.
            #   When V > Er: (Er-V) < 0 → leak is negative (restoring).
            #   When V = Er: leak = 0 → neuron is at rest.
            dV[name] = (I_leak + I_syn[name] + I_app) / nrn.Cm

        # ---- Phase C: update ALL voltages simultaneously ----
        for name, nrn in neurons.items():
            nrn.V += dV[name] * dt_ms
            log[name][step] = nrn.V

    return log


# ====================================
# PHYSICAL PLANT  (Peterka 2002 / McNeal & Hunt 2026 Table I)
# ====================================

MOMENT_OF_INERTIA_KGM2          = 63.0
BODY_MASS_KG                    = 77.8
COM_HEIGHT_M                    = 0.9
GRAVITY_MS2                     = 9.81
DESTABILIZING_TORQUE_NM_PER_RAD = BODY_MASS_KG * GRAVITY_MS2 * COM_HEIGHT_M
JOINT_DAMPING_NMS_PER_RAD       = 351.0

ADAPTER_ANGLE_NA_PER_RAD  = 180.0 / np.pi  # gA ≈ 57.3 nA/rad (converts physical angle → SNS current)
SNS_CLIP_MAGNITUDE_NA     = 10.0            # ±10 nA symmetric saturation limit
ADAPTER_IB_SCALE_NA_PER_N = 10.0 / 1500.0  # |gIb|, Fmax=1500 N  (McNeal & Hunt Eq. 12)

SENSORY_DELAY_S    = 0.090   # tau_d — sensorimotor delay
TIMESTEP_S         = 2e-4    # simulation timestep
PRTS_AMPLITUDE_DEG = 1.0     # Peterka 2002 TC4: 1-degree peak-to-peak

HILL_FMAX_N       = 1500.0
HILL_MOMENT_ARM_M = 0.049


# ====================================
# MUSCLE MODEL PLACEHOLDER
# Replace with Hill-type model (McNeal & Hunt 2026, Section III-A-4)
# ====================================

def compute_tension_proxy_n(ta_ccw_activation, ta_cw_activation):
    """Placeholder — approximate muscle tension from SNS motor activations (nN)."""
    return abs(ta_ccw_activation - ta_cw_activation) * HILL_FMAX_N

def tension_to_ib_current_na(tension_n):
    """Placeholder — adapter from tension (N) to Type-Ib current (nA), clipped to [0,10]."""
    return min(ADAPTER_IB_SCALE_NA_PER_N * tension_n, SNS_CLIP_MAGNITUDE_NA)

def compute_net_torque_nm(ta_ccw_activation, ta_cw_activation):
    """Placeholder — net ankle torque (N·m) from bilateral activations."""
    return (ta_ccw_activation - ta_cw_activation) * HILL_FMAX_N * HILL_MOMENT_ARM_M


# ====================================
# ADAPTER FUNCTIONS (dynamic simulation)
# ====================================

def angle_to_current_na(angle_rad):
    """Map a physical angle (rad) to a clipped SNS input current (nA). McNeal & Hunt Eq. 11."""
    raw = ADAPTER_ANGLE_NA_PER_RAD * angle_rad
    return max(-SNS_CLIP_MAGNITUDE_NA, min(SNS_CLIP_MAGNITUDE_NA, raw))

def split_to_bilateral_na(signed_current_na):
    """Split a signed current into (ccw_nA, cw_nA), both non-negative. McNeal & Hunt Fig. 3."""
    return max(0.0, signed_current_na), max(0.0, -signed_current_na)

def motor_voltage_to_activation(v_sns_mv):
    """Map SNS motor voltage to muscle activation [0,1]. McNeal & Hunt Eq. 13."""
    return max(0.0, min(1.0, (v_sns_mv - Er) / R))


# ====================================
# DYNAMIC SIMULATION STEP
# ====================================

def step_controller(neurons, synapses,
                    body_sway_rad, surface_sway_rad, dt_ms,
                    theta_ref_rad=0.0, ib_current_na=0.0,
                    bp=BP_DEFAULT, bd=BD_DEFAULT,
                    bc=BC_DEFAULT, bt=BT_DEFAULT):
    """
    Advance the TC4 SNS controller by one timestep (dynamic simulation).

    Unlike run_segment() which uses a fixed I_app_dict, this function
    assembles applied currents from the current physical state on every call,
    allowing the network to respond to a time-varying body sway trajectory.

    Args:
        neurons, synapses: from build_neurons() and build_synapses()
        body_sway_rad:     current body sway angle (rad)
        surface_sway_rad:  current surface sway angle (rad)
        dt_ms:             simulation timestep in milliseconds
        theta_ref_rad:     desired body angle (rad)
        ib_current_na:     Type-Ib afferent current (nA), clipped to [0,10]
        bp, bd, bc, bt:    tunable bias parameters (nA)

    Returns:
        (ta_ccw_activation, ta_cw_activation): motor activations in [0,1]
    """
    bs_signed        = angle_to_current_na(body_sway_rad)
    ss_signed        = angle_to_current_na(surface_sway_rad)
    theta_ref_signed = angle_to_current_na(theta_ref_rad)

    bs_ccw_na,  bs_cw_na  = split_to_bilateral_na(bs_signed)
    ss_ccw_na,  ss_cw_na  = split_to_bilateral_na(ss_signed)
    ref_ccw_na, ref_cw_na = split_to_bilateral_na(theta_ref_signed)

    applied = {
        "bs_ccw": bs_ccw_na, "bs_cw": bs_cw_na,
        "ss_ccw": ss_ccw_na, "ss_cw": ss_cw_na,
        "theta_ref_ccw": ref_ccw_na, "theta_ref_cw": ref_cw_na,
        "ib_input": ib_current_na,
        "kp_bias": bp, "kd_bias": bd, "kc_bias": bc, "kt_bias": bt,
        "kp_mod_ccw": I_APP_MOD, "kp_mod_cw": I_APP_MOD,
        "kd_mod_ccw": I_APP_MOD, "kd_mod_cw": I_APP_MOD,
        "kc_mod_ccw": I_APP_MOD, "kc_mod_cw": I_APP_MOD,
        "kt_mod_ccw": I_APP_MOD, "kt_mod_cw": I_APP_MOD,
    }

    I_syn = {name: 0.0 for name in neurons}
    for syn in synapses:
        I_syn[syn.post] += syn.current(neurons[syn.pre].V, neurons[syn.post].V)

    for name, nrn in neurons.items():
        I_app  = applied.get(name, 0.0)
        I_leak = nrn.Gm * (nrn.Er - nrn.V)
        nrn.V += ((I_leak + I_syn[name] + I_app) / nrn.Cm) * dt_ms

    return motor_voltage_to_activation(neurons["ta_ccw"].V), \
           motor_voltage_to_activation(neurons["ta_cw"].V)


# ====================================
# BACKWARD COMPATIBILITY WRAPPER
# (visualize_tc4_network.py and other scripts that import build_tc4_network)
# ====================================

def build_tc4_network():
    """Thin wrapper — returns (neurons, synapses, bias_neuron_map) for legacy callers."""
    return build_neurons(), build_synapses(), {
        "bp": ["kp_bias"], "bd": ["kd_bias"],
        "bc": ["kc_bias"], "bt": ["kt_bias"],
    }


# ====================================
# STEP 7: ENTRY POINT — static analysis CLI
# ====================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TC4 SNS controller — Hilts 2018 baseline static analysis"
    )
    parser.add_argument("--bs",        type=float, default=5.0,        help="Body sway (degrees)")
    parser.add_argument("--ss",        type=float, default=0.0,        help="Surface sway (degrees)")
    parser.add_argument("--theta_ref", type=float, default=0.0,        help="Reference angle (degrees)")
    parser.add_argument("--bp",        type=float, default=BP_DEFAULT,  help="Kp bias current (nA)")
    parser.add_argument("--bd",        type=float, default=BD_DEFAULT,  help="Kd bias current (nA)")
    parser.add_argument("--bc",        type=float, default=BC_DEFAULT,  help="Kc bias current (nA)")
    parser.add_argument("--bt",        type=float, default=BT_DEFAULT,  help="Kt bias current (nA)")
    parser.add_argument("--steps",     type=int,   default=20000,       help="Simulation steps")
    parser.add_argument("--dt",        type=float, default=0.1,         help="Timestep (ms)")
    parser.add_argument("--no-plot",   action="store_true",             help="Skip network diagram")
    parser.add_argument("--simulate",  action="store_true",             help="Run full PRTS simulation")
    args = parser.parse_args()

    if args.simulate:
        # --- Full dynamic simulation with PRTS surface stimulus ---
        neurons, synapses, _ = build_tc4_network()
        time_s, surface_sway_rad = generate_prts(
            v_amp=np.deg2rad(PRTS_AMPLITUDE_DEG), dt=TIMESTEP_S
        )
        dt_ms = TIMESTEP_S * 1000.0
        delay_steps  = round(SENSORY_DELAY_S / TIMESTEP_S)
        delay_buffer = np.zeros(delay_steps)
        body_sway_log = np.zeros(len(time_s))
        ta_ccw_log    = np.zeros(len(time_s))
        ta_cw_log     = np.zeros(len(time_s))
        torque_log    = np.zeros(len(time_s))
        body_rad = 0.0
        body_vel = 0.0
        ta_ccw_prev = ta_cw_prev = 0.0

        for step in range(len(time_s)):
            ss_rad = surface_sway_rad[step]
            body_angle = body_rad + ss_rad
            delayed = delay_buffer[-1]
            delay_buffer = np.roll(delay_buffer, 1)
            delay_buffer[0] = body_angle
            ib_na = tension_to_ib_current_na(compute_tension_proxy_n(ta_ccw_prev, ta_cw_prev))
            ta_ccw_prev, ta_cw_prev = step_controller(
                neurons, synapses, delayed, ss_rad, dt_ms, ib_current_na=ib_na,
                bp=args.bp, bd=args.bd, bc=args.bc, bt=args.bt
            )
            ta_net = compute_net_torque_nm(ta_ccw_prev, ta_cw_prev)
            destab = DESTABILIZING_TORQUE_NM_PER_RAD * body_angle
            damp   = JOINT_DAMPING_NMS_PER_RAD * body_vel
            body_vel += (ta_net - damp + destab) / MOMENT_OF_INERTIA_KGM2 * TIMESTEP_S
            body_rad += body_vel * TIMESTEP_S
            body_sway_log[step] = np.rad2deg(body_angle)
            ta_ccw_log[step]    = ta_ccw_prev
            ta_cw_log[step]     = ta_cw_prev
            torque_log[step]    = ta_net

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(time_s, np.rad2deg(surface_sway_rad), label="Surface (deg)", color="black", lw=0.8)
        axes[0].plot(time_s, body_sway_log, label="Body (deg)", color="blue", lw=0.8)
        axes[0].set_ylabel("Angle (deg)"); axes[0].legend(fontsize=8)
        axes[0].set_title("TC4 SNS Controller — Hilts 2018 baseline + McNeal & Hunt 2026 architecture")
        axes[1].plot(time_s, torque_log, label="Torque (N·m)", color="red", lw=0.8)
        axes[1].set_ylabel("Torque (N·m)"); axes[1].legend(fontsize=8)
        axes[2].plot(time_s, ta_ccw_log, label="CCW activation", color="green", lw=0.8)
        axes[2].plot(time_s, ta_cw_log,  label="CW activation",  color="orange", lw=0.8)
        axes[2].set_ylabel("Activation [0,1]"); axes[2].set_xlabel("Time (s)"); axes[2].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("tc4_simulation_result.png", dpi=150)
        plt.show()
        print("Simulation complete. Saved to tc4_simulation_result.png")

    else:
        # --- Static steady-state analysis ---
        neurons  = build_neurons()
        synapses = build_synapses()
        iapp     = build_iapp(args.bs, args.ss, args.theta_ref,
                              args.bp, args.bd, args.bc, args.bt)
        log      = run_segment(neurons, synapses, iapp, dt_ms=args.dt, steps=args.steps)

        # Input neurons (bilateral split): U = applied current / Gm = angle in degrees
        # All other neurons: U = V - Er (standard activation above rest)
        INPUT_NEURONS = {"bs_ccw", "bs_cw", "ss_ccw", "ss_cw",
                         "theta_ref_ccw", "theta_ref_cw"}

        print(f"\n=== TC4 SNS Controller — Hilts 2018 baseline ===")
        print(f"  bs={args.bs} deg   ss={args.ss} deg   theta_ref={args.theta_ref} deg")
        print(f"  bp={args.bp} nA   bd={args.bd} nA   bc={args.bc} nA   bt={args.bt} nA\n")

        # ── Neurons ──────────────────────────────────────────────────────────
        # U = V - Er is the membrane potential above rest (the "activation level")
        # I_app column shows the externally applied current driving this neuron
        print(f"  {'Neuron':<20}  {'U (mV)':>9}  {'I_app (nA)':>11}  {'tau(ms)':>8}")
        print(f"  {'-'*55}")
        for name, nrn in neurons.items():
            U       = log[name][-1] - Er
            tau     = nrn.Cm / nrn.Gm
            i_app   = iapp.get(name, 0.0)
            print(f"  {name:<20}  {U:>+9.3f}  {i_app:>+11.3f}  {tau:>8.1f}")

        # ── Synapses ─────────────────────────────────────────────────────────
        # g_s  = actual conductance at the final simulation step (µS)
        #        g_s = 0 when V_pre < Elo (synapse is silenced)
        #        g_s = gs_max when V_pre > Ehi (synapse is fully open)
        # I_s  = g_s * (Es - V_post) — current delivered to the post-neuron (nA)
        #        positive I_s drives V_post toward Es (excitatory if Es > V_post,
        #        inhibitory if Es < V_post)
        # Act% = g_s / gs_max × 100 — how open the synapse is (0 = silent, 100 = saturated)
        print(f"\n  {'Synapse':<38}  {'Pre':>16} -> {'Post':<18}  {'g_s(uS)':>8}  {'I_s(nA)':>9}  {'Act%':>6}  Type")
        print(f"  {'-'*110}")

        def _syn_type(syn):
            if syn.Es == Er:    return "GATE"   # ΔE=0 gating synapse (Hilts Mult. Syn 2)
            if syn.Es < 0:      return "INH"    # standard inhibitory (Es=-100)
            return "EXC"                         # excitatory (Es=+134 or Wg/Wp variants)

        # Sort by the numeric synapse index so syn1…syn64 print in order
        def _syn_index(syn):
            try: return int(syn.name.split("_")[0].replace("syn", ""))
            except ValueError: return 9999

        for syn in sorted(synapses, key=_syn_index):
            V_pre  = log[syn.pre][-1]  if syn.pre  in log else Er
            V_post = log[syn.post][-1] if syn.post in log else Er
            g_s    = syn.conductance(V_pre)
            I_s    = syn.current(V_pre, V_post)
            act    = (g_s / syn.gs_max * 100) if syn.gs_max > 0 else 0.0
            stype  = _syn_type(syn)
            print(f"  {syn.name:<38}  {syn.pre:>16} -> {syn.post:<18}  {g_s:>8.4f}  {I_s:>+9.4f}  {act:>5.1f}%  {stype}")

        print(f"\n  Total: {len(neurons)} neurons, {len(synapses)} synapses")

        if not args.no_plot:
            try:
                from visualize_tc4_network import draw_network
                draw_network(neurons, synapses, log=log)
            except ImportError:
                print("visualize_tc4_network.py not found — skipping diagram.")
