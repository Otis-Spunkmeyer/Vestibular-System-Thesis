"""
TC4 SNS Controller — Peterka 2002, Test Condition 4
Extends the McNeal & Hunt 2026 split-derivative architecture with
dual-channel sensory integration (proprioceptive + graviceptive).

Reference: McNeal & Hunt 2026, "Split-derivative control and tension
feedback in a neuromechanical model of human balance"

TC4 conditions (Peterka 2002, Table 2, 1-degree amplitude):
  Visual:        eyes closed    → visual channel absent
  Surface:       PRTS stimulus  → proprioceptive channel active
  Graviceptive:  veridical      → vestibular channel active
  Wp = 0.70,  Wg = 0.30

Error signal:  e = -Wg * BS - Wp * (BS - SS)
Torque:        Ta = (Kp + Kd * s) * e(t - tau_d)   [PD only, no integral]

Network layout (38 neurons, 46 synapses):
  Stage 1 — Bilateral input          (4  neurons)
  Stage 2 — TC4 sensory integration  (8  neurons)   <-- TC4 addition over McNeal & Hunt
  Stage 3 — Derivative estimation    (6  neurons)
  Stage 4 — Co-activation |d(t)|     (1  neuron)
  Stage 5 — Bias-gated gain stages   (16 neurons)   Kp, Kd, Kc, Kt
  Stage 6 — Type-Ib input node       (1  neuron)
  Stage 7 — Bilateral motor output   (2  neurons)
"""

import numpy as np
import matplotlib.pyplot as plt

from sns_neurons import Neuron, Synapse
from sns_subnetworks import (
    Er, R, Elo, Ehi,
    gs_exc, Es_exc,
    gs_inh, Es_inh,
    GM_DEFAULT,
    make_neuron,
)
from prts_generator import generate_prts

# =====================================================================
# Physical plant parameters  (Peterka 2002 / McNeal & Hunt 2026 Table I)
# =====================================================================
MOMENT_OF_INERTIA_KGM2         = 63.0
BODY_MASS_KG                   = 77.8
COM_HEIGHT_M                   = 0.9
GRAVITY_MS2                    = 9.81
DESTABILIZING_TORQUE_NM_PER_RAD = BODY_MASS_KG * GRAVITY_MS2 * COM_HEIGHT_M
JOINT_DAMPING_NMS_PER_RAD      = 351.0

# =====================================================================
# TC4 sensory weights  (Peterka 2002, Table 2, 1-degree PRTS amplitude)
# =====================================================================
PROPRIOCEPTIVE_WEIGHT = 0.70   # Wp
GRAVICEPTIVE_WEIGHT   = 0.30   # Wg

# =====================================================================
# Bias parameters  (McNeal & Hunt 2026, Table IV, Split + kt on)
# These are the starting values from the best-performing condition.
# Tunable via particle swarm optimization to match human FRF.
# =====================================================================
BIAS_PROPORTIONAL_NA    = 4.26   # bp  — proportional pathway
BIAS_DIRECTIONAL_DER_NA = 5.01   # bd  — directional derivative pathway
BIAS_COACTIVATION_NA    = 2.48   # bc  — co-activation derivative pathway
BIAS_TYPE_IB_NA         = 5.42   # bt  — Type-Ib tension feedback pathway

# =====================================================================
# Adapter constants  (McNeal & Hunt 2026, Section III-A-3)
# These convert between physical units and SNS neural units.
# =====================================================================
ADAPTER_ANGLE_NA_PER_RAD     = 180.0 / np.pi   # gA ≈ 57.3 nA/rad
SNS_CLIP_MAGNITUDE_NA        = 10.0             # ±10 nA operating limit
ADAPTER_IB_SCALE_NA_PER_N    = 10.0 / 1500.0   # |gIb|, Fmax = 1500 N

# =====================================================================
# Simulation parameters
# =====================================================================
SENSORY_DELAY_S    = 0.090    # tau_d — sensorimotor delay
TIMESTEP_S         = 2e-4     # delta_t — simulation step
PRTS_AMPLITUDE_DEG = 1.0      # Peterka 2002 TC4: 1-degree peak-to-peak

# =====================================================================
# MUSCLE MODEL PLACEHOLDER
# =====================================================================
# TODO [REPLACE — Hill-type muscle model]:
#   The functions below substitute a net-torque proxy for the full
#   Hill-type model described in McNeal & Hunt 2026, Section III-A-4.
#
#   When replacing, implement:
#     activation dynamics:  tau_act(a, u) * da/dt = u - a          (Eq. 16)
#     force production:     F(a, l, l_dot) = Fmax * a * fl * fv + fpe  (Eq. 17)
#     muscle kinematics:    l(theta_A)     = l0 ∓ Lmoment * theta_A    (Eq. 18)
#     net torque:           Ta = Lmoment * (F_agonist - F_antagonist)   (Eq. 15)
#
#   Required constants (McNeal & Hunt 2026, Table I):
#     HILL_FMAX_N        = 1500 N
#     HILL_LENGTH0_M     = 0.41 m
#     HILL_MOMENT_ARM_M  = 0.049 m
#     HILL_LENGTH_RANGE  = (0.394, 0.423) m
#     FLV_PARAMS         = range=0.75, vmax=1.5, fpmax=1.3, fvmax=1.2
#
#   The tension T returned from the Hill model feeds back via:
#     I_Ib = clip(gIb * T, 0, SNS_CLIP_MAGNITUDE_NA)   (Eq. 12)
#   where gIb = -10 / Fmax nA/N.
# =====================================================================
HILL_FMAX_N       = 1500.0    # placeholder — used only for proxy scaling
HILL_MOMENT_ARM_M = 0.049     # placeholder — used only for proxy torque

def compute_tension_proxy_n(ta_ccw_activation, ta_cw_activation):
    """
    PLACEHOLDER — estimates muscle tension from SNS motor activations.
    Returns a non-negative tension in Newtons (approximate).
    TODO: Replace with Hill-type F(a, l, l_dot) using actual muscle state.
    """
    net_activation = abs(ta_ccw_activation - ta_cw_activation)
    return net_activation * HILL_FMAX_N

def tension_to_ib_current_na(tension_n):
    """
    PLACEHOLDER — adapter from muscle tension to Type-Ib input current.
    Returns a value in [0, SNS_CLIP_MAGNITUDE_NA] nA.
    TODO: Replace with: I_Ib = clip(gIb * T_from_Hill_model, 0, 10 nA)
    """
    raw = ADAPTER_IB_SCALE_NA_PER_N * tension_n
    return min(raw, SNS_CLIP_MAGNITUDE_NA)

def compute_net_torque_nm(ta_ccw_activation, ta_cw_activation):
    """
    PLACEHOLDER — converts bilateral SNS activations to net ankle torque.
    Returns net torque in N·m (positive = CCW / forward restoring direction).
    TODO: Replace with: Ta = Lmoment * (F_ccw_Hill - F_cw_Hill)
    """
    return (ta_ccw_activation - ta_cw_activation) * HILL_FMAX_N * HILL_MOMENT_ARM_M


# =====================================================================
# Adapter functions
# =====================================================================

def angle_to_current_na(angle_rad):
    """
    Map a physical angle (rad) to a clipped SNS input current (nA).
    Symmetric clip at ±10 nA keeps the SNS in its linear range (McNeal & Hunt Eq. 11).
    """
    raw = ADAPTER_ANGLE_NA_PER_RAD * angle_rad
    return max(-SNS_CLIP_MAGNITUDE_NA, min(SNS_CLIP_MAGNITUDE_NA, raw))

def split_to_bilateral_na(signed_current_na):
    """
    Split a signed current into a CCW (positive) and CW (negative-magnitude) component.
    Each component is non-negative; only one is active at any instant.
    This is how bilateral SNS architectures represent signed signals (McNeal & Hunt Fig. 3).
    Returns (ccw_current_na, cw_current_na).
    """
    ccw = max(0.0,  signed_current_na)
    cw  = max(0.0, -signed_current_na)
    return ccw, cw

def motor_voltage_to_activation(v_sns_mv):
    """
    Map SNS motor output voltage to muscle activation command in [0, 1].
    McNeal & Hunt 2026, Eq. 13:  A_MJ = clip((A_SNS + 60) / R, 0, 1)
    TODO: This activation feeds into the Hill-type muscle model.
    """
    return max(0.0, min(1.0, (v_sns_mv + 60.0) / R))


# =====================================================================
# Transmission synapse conductance formula  (Hilts 2018 / FSA design)
# gs_max is chosen so that U_post ≈ target_gain × U_pre at U_pre = R/2.
# Derivation:
#   At steady state:  U_post = (gs/R × U_pre × (Es-Er)) / (Gm + gs/R × U_pre)
#   Solve at U_pre = R/2:  gs = target_gain × Gm × R / ((Es-Er) - target_gain × R/2)
# =====================================================================
def _gs_for_gain(target_gain):
    driving_force_mv = Es_exc - Er          # 194 mV
    return (target_gain * GM_DEFAULT * R) / (driving_force_mv - target_gain * (R / 2.0))

GS_WG = _gs_for_gain(GRAVICEPTIVE_WEIGHT)    # ≈ 0.0314 µS  (gain = 0.30)
GS_WP = _gs_for_gain(PROPRIOCEPTIVE_WEIGHT)  # ≈ 0.0749 µS  (gain = 0.70)


# =====================================================================
# Network construction
# =====================================================================

def build_tc4_network():
    """
    Construct the full TC4 SNS network as described in the module docstring.

    Returns:
        neurons (dict):        name → Neuron object
        synapses (list):       list of Synapse objects
        bias_neuron_map (dict): pathway name → list of neuron names that receive
                                the corresponding bias current as I_applied
    """
    neurons  = {}
    synapses = []

    def add_neuron(n):
        neurons[n.name] = n

    def add_synapse(pre, post, gs, Es_reversal, syn_name=None):
        name = syn_name or f"{pre}_to_{post}"
        synapses.append(Synapse(gs, Es_reversal, Elo, Ehi, name=name, pre=pre, post=post))

    def new_standard_neuron(name):
        return Neuron(Cm_nF=5.0, Gm_uS=GM_DEFAULT, Er_mV=Er, name=name)

    # ----------------------------------------------------------------
    # Stage 1 — Bilateral input neurons
    # Each physical signal (body sway BS, surface sway SS) is split into
    # a CCW (positive) half and a CW (negative-magnitude) half before
    # entering the network.  Only one half is active at any instant.
    # ----------------------------------------------------------------
    for name in ["bs_ccw", "bs_cw", "ss_ccw", "ss_cw"]:
        add_neuron(new_standard_neuron(name))

    # ----------------------------------------------------------------
    # Stage 2 — TC4 dual-channel sensory integration  (BILATERAL)
    #
    # This stage is the TC4-specific addition over McNeal & Hunt 2026.
    # McNeal & Hunt use a single-channel error: e = theta_A - theta_ref.
    # TC4 replaces that with a two-channel error:
    #   e = -Wg * BS  -  Wp * (BS - SS)
    # implemented neurally as:
    #   sub_diff  = BS - SS                (subtraction subnetwork)
    #   wg_node   = Wg × BS               (transmission synapse, gain = 0.30)
    #   wp_node   = Wp × (BS - SS)        (transmission synapse, gain = 0.70)
    #   error     = wg_node + wp_node      (addition subnetwork)
    #
    # The negation of the full expression is handled by the motor-output
    # convention: the CCW pool drives restoring torque when the CCW error
    # is positive, and vice versa for the CW pool.
    #
    # Both CCW and CW sides are constructed identically (mirrored).
    # ----------------------------------------------------------------
    for side in ["ccw", "cw"]:

        # Subtraction: BS - SS  →  proprioceptive error (ankle angle)
        add_neuron(new_standard_neuron(f"sub_diff_{side}"))
        add_synapse(f"bs_{side}",   f"sub_diff_{side}", gs_exc, Es_exc)  # excitatory
        add_synapse(f"ss_{side}",   f"sub_diff_{side}", gs_inh, Es_inh)  # inhibitory

        # Transmission: Wg × BS  →  graviceptive channel
        add_neuron(new_standard_neuron(f"wg_{side}"))
        add_synapse(f"bs_{side}",       f"wg_{side}", GS_WG, Es_exc)

        # Transmission: Wp × (BS - SS)  →  proprioceptive channel
        add_neuron(new_standard_neuron(f"wp_{side}"))
        add_synapse(f"sub_diff_{side}", f"wp_{side}", GS_WP, Es_exc)

        # Addition: wg + wp  →  combined TC4 error
        add_neuron(new_standard_neuron(f"error_{side}"))
        add_synapse(f"wg_{side}", f"error_{side}", gs_exc, Es_exc)
        add_synapse(f"wp_{side}", f"error_{side}", gs_exc, Es_exc)

    # ----------------------------------------------------------------
    # Stage 3 — Derivative estimation  (McNeal & Hunt Eqs. 8-10)
    # d(t) = v_fast - v_slow
    # tau_fast = Cm_fast / Gm = 2 ms   (responds quickly to error changes)
    # tau_slow = Cm_slow / Gm = 10 ms  (lags, so the difference is derivative)
    # Both fast and slow neurons receive the same error signal.
    # The output neuron computes the difference via excitatory/inhibitory synapses.
    # ----------------------------------------------------------------
    for side in ["ccw", "cw"]:
        add_neuron(Neuron(Cm_nF=2.0,  Gm_uS=GM_DEFAULT, Er_mV=Er, name=f"deriv_fast_{side}"))
        add_neuron(Neuron(Cm_nF=10.0, Gm_uS=GM_DEFAULT, Er_mV=Er, name=f"deriv_slow_{side}"))
        add_neuron(new_standard_neuron(f"deriv_out_{side}"))

        add_synapse(f"error_{side}",      f"deriv_fast_{side}", gs_exc, Es_exc)
        add_synapse(f"error_{side}",      f"deriv_slow_{side}", gs_exc, Es_exc)
        add_synapse(f"deriv_fast_{side}", f"deriv_out_{side}",  gs_exc, Es_exc)
        add_synapse(f"deriv_slow_{side}", f"deriv_out_{side}",  gs_inh, Es_inh)

    # ----------------------------------------------------------------
    # Stage 4 — Co-activation node:  |d(t)| ≈ deriv_out_ccw + deriv_out_cw
    #
    # In the bilateral architecture each half-wave is naturally rectified:
    #   deriv_out_ccw is active only when d > 0 in the CCW direction.
    #   deriv_out_cw  is active only when d > 0 in the CW  direction.
    # Summing both gives the unsigned derivative magnitude.
    # This implements the co-activation pathway from McNeal & Hunt Eq. 1:
    #   x_c(t) = |d(t)|
    # ----------------------------------------------------------------
    add_neuron(new_standard_neuron("coact_node"))
    add_synapse("deriv_out_ccw", "coact_node", gs_exc, Es_exc)
    add_synapse("deriv_out_cw",  "coact_node", gs_exc, Es_exc)

    # ----------------------------------------------------------------
    # Stage 5 — Bias-gated gain stages  (McNeal & Hunt, Section III-A-2a)
    #
    # Each control pathway has:
    #   mod_neuron  — receives bias applied current I_app = b_alpha
    #   prod_neuron — receives the signal from upstream + modulation
    #
    # The effective pathway gain scales with the bias parameter b_alpha.
    # These are the ONLY tunable parameters (all synapse weights are fixed).
    # In McNeal & Hunt, bias parameters are optimized via PSO.
    # Starting values from Table IV (Split + kt on): bp=4.26, bd=5.01, bc=2.48, bt=5.42 nA
    # ----------------------------------------------------------------
    for side in ["ccw", "cw"]:

        # Kp: proportional gain  (signal = error)
        add_neuron(new_standard_neuron(f"kp_mod_{side}"))   # receives I_app = bp
        add_neuron(new_standard_neuron(f"kp_prod_{side}"))
        add_synapse(f"error_{side}",  f"kp_prod_{side}", gs_exc, Es_exc)
        add_synapse(f"kp_mod_{side}", f"kp_prod_{side}", gs_exc, Es_exc)

        # Kd: directional derivative gain  (signal = deriv_out, signed)
        add_neuron(new_standard_neuron(f"kd_mod_{side}"))   # receives I_app = bd
        add_neuron(new_standard_neuron(f"kd_prod_{side}"))
        add_synapse(f"deriv_out_{side}", f"kd_prod_{side}", gs_exc, Es_exc)
        add_synapse(f"kd_mod_{side}",    f"kd_prod_{side}", gs_exc, Es_exc)

        # Kc: co-activation derivative gain  (signal = coact_node = |d|, unsigned)
        # Both CCW and CW sides receive the same coact_node — symmetric stiffening.
        add_neuron(new_standard_neuron(f"kc_mod_{side}"))   # receives I_app = bc
        add_neuron(new_standard_neuron(f"kc_prod_{side}"))
        add_synapse("coact_node",      f"kc_prod_{side}", gs_exc, Es_exc)
        add_synapse(f"kc_mod_{side}", f"kc_prod_{side}", gs_exc, Es_exc)

        # Kt: Type-Ib tension feedback gain  (signal = ib_input)
        add_neuron(new_standard_neuron(f"kt_mod_{side}"))   # receives I_app = bt
        add_neuron(new_standard_neuron(f"kt_prod_{side}"))
        add_synapse("ib_input",        f"kt_prod_{side}", gs_exc, Es_exc)
        add_synapse(f"kt_mod_{side}", f"kt_prod_{side}", gs_exc, Es_exc)

    # ----------------------------------------------------------------
    # Stage 6 — Type-Ib tension input node
    # Receives I_applied = tension_to_ib_current_na(tension) each timestep.
    # TODO [REPLACE — Hill-type muscle model]:
    #   The applied current to this neuron currently comes from
    #   compute_tension_proxy_n(), which is a rough approximation.
    #   Replace with actual Hill-model muscle tension (McNeal & Hunt Eq. 12):
    #     I_Ib = clip(gIb * T_hill, 0, 10 nA)  where gIb = -10/Fmax nA/N
    # ----------------------------------------------------------------
    add_neuron(new_standard_neuron("ib_input"))

    # ----------------------------------------------------------------
    # Stage 7 — Bilateral motor output
    # Each motor neuron sums all four pathway contributions for its side.
    # ta_ccw drives the CCW (extensor) muscle; ta_cw drives the CW (flexor).
    # TODO [REPLACE — Hill-type muscle model]:
    #   Currently motor_voltage_to_activation() maps SNS voltage → [0,1],
    #   then compute_net_torque_nm() uses a linear proxy.
    #   Replace with Hill-type force computation:
    #     A_MJ = motor_voltage_to_activation(ta_ccw.V or ta_cw.V)
    #     F    = Fmax * A_MJ * fl(l) * fv(l_dot) + fpe(l)   (Eq. 17)
    #     Ta   = Lmoment * (F_ccw - F_cw)                    (Eq. 15)
    # ----------------------------------------------------------------
    for side in ["ccw", "cw"]:
        add_neuron(new_standard_neuron(f"ta_{side}"))
        add_synapse(f"kp_prod_{side}", f"ta_{side}", gs_exc, Es_exc)
        add_synapse(f"kd_prod_{side}", f"ta_{side}", gs_exc, Es_exc)
        add_synapse(f"kc_prod_{side}", f"ta_{side}", gs_exc, Es_exc)
        add_synapse(f"kt_prod_{side}", f"ta_{side}", gs_exc, Es_exc)

    # ----------------------------------------------------------------
    # Bias neuron map
    # Maps each bias parameter name to the list of modulator neurons that
    # receive it as I_applied.  Both CCW and CW sides share the same bias.
    # ----------------------------------------------------------------
    bias_neuron_map = {
        "bp": ["kp_mod_ccw", "kp_mod_cw"],
        "bd": ["kd_mod_ccw", "kd_mod_cw"],
        "bc": ["kc_mod_ccw", "kc_mod_cw"],
        "bt": ["kt_mod_ccw", "kt_mod_cw"],
    }

    return neurons, synapses, bias_neuron_map


# =====================================================================
# Simulation step
# =====================================================================

def step_controller(neurons, synapses, bias_neuron_map,
                    body_sway_rad, surface_sway_rad,
                    ta_ccw_activation_prev, ta_cw_activation_prev,
                    dt_ms):
    """
    Advance the TC4 SNS controller by one timestep.

    Args:
        neurons:                  dict of Neuron objects (modified in place)
        synapses:                 list of Synapse objects
        bias_neuron_map:          dict from build_tc4_network()
        body_sway_rad:            current body sway angle (rad), positive = CCW
        surface_sway_rad:         current surface sway angle (rad), positive = CCW
        ta_ccw_activation_prev:   CCW motor activation from previous step [0,1]
        ta_cw_activation_prev:    CW  motor activation from previous step [0,1]
        dt_ms:                    simulation timestep in milliseconds

    Returns:
        ta_net_nm:           net active ankle torque (N·m), positive = CCW
        ta_ccw_activation:   updated CCW motor activation [0,1]
        ta_cw_activation:    updated CW  motor activation [0,1]
    """
    # Convert angles to bilateral SNS currents
    bs_signed = angle_to_current_na(body_sway_rad)
    ss_signed = angle_to_current_na(surface_sway_rad)
    bs_ccw_na, bs_cw_na = split_to_bilateral_na(bs_signed)
    ss_ccw_na, ss_cw_na = split_to_bilateral_na(ss_signed)

    # Compute Type-Ib input current from muscle tension proxy
    # TODO [REPLACE — Hill-type muscle model]: see tension_to_ib_current_na()
    tension_proxy_n = compute_tension_proxy_n(ta_ccw_activation_prev, ta_cw_activation_prev)
    ib_current_na   = tension_to_ib_current_na(tension_proxy_n)

    # Assemble applied current dictionary
    applied_currents = {
        "bs_ccw":   bs_ccw_na,
        "bs_cw":    bs_cw_na,
        "ss_ccw":   ss_ccw_na,
        "ss_cw":    ss_cw_na,
        "ib_input": ib_current_na,
    }

    # Apply bias currents to modulator neurons
    bias_values = {
        "bp": BIAS_PROPORTIONAL_NA,
        "bd": BIAS_DIRECTIONAL_DER_NA,
        "bc": BIAS_COACTIVATION_NA,
        "bt": BIAS_TYPE_IB_NA,
    }
    for pathway_name, neuron_names in bias_neuron_map.items():
        for neuron_name in neuron_names:
            applied_currents[neuron_name] = bias_values[pathway_name]

    # Compute all synaptic currents from last-step voltages (forward Euler)
    synaptic_currents = {name: 0.0 for name in neurons}
    for syn in synapses:
        synaptic_currents[syn.post] += syn.current(neurons[syn.pre].V, neurons[syn.post].V)

    # Update all neuron voltages simultaneously
    for name, neuron in neurons.items():
        i_app   = applied_currents.get(name, 0.0)
        i_leak  = neuron.Gm * (neuron.Er - neuron.V)
        dv_dt   = (i_leak + synaptic_currents[name] + i_app) / neuron.Cm
        neuron.V += dv_dt * dt_ms

    # Read motor output voltages and convert to activations
    ta_ccw_activation = motor_voltage_to_activation(neurons["ta_ccw"].V)
    ta_cw_activation  = motor_voltage_to_activation(neurons["ta_cw"].V)

    # Compute net torque
    # TODO [REPLACE — Hill-type muscle model]: see compute_net_torque_nm()
    ta_net_nm = compute_net_torque_nm(ta_ccw_activation, ta_cw_activation)

    return ta_net_nm, ta_ccw_activation, ta_cw_activation


# =====================================================================
# Main simulation loop
# =====================================================================

if __name__ == "__main__":

    # Build the network
    neurons, synapses, bias_neuron_map = build_tc4_network()
    print(f"Network built: {len(neurons)} neurons, {len(synapses)} synapses")

    # Generate PRTS surface stimulus (TC4: 1-degree amplitude)
    time_s, surface_sway_rad = generate_prts(
        v_amp=np.deg2rad(PRTS_AMPLITUDE_DEG),
        dt=TIMESTEP_S
    )
    num_steps = len(time_s)
    dt_ms     = TIMESTEP_S * 1000.0   # convert to milliseconds for neuron update

    # Delay buffer for sensory feedback (pure time delay on body sway signal)
    delay_steps  = round(SENSORY_DELAY_S / TIMESTEP_S)
    delay_buffer = np.zeros(delay_steps)

    # Storage arrays
    body_sway_log        = np.zeros(num_steps)
    active_torque_log    = np.zeros(num_steps)
    surface_sway_log     = np.zeros(num_steps)
    ta_ccw_log           = np.zeros(num_steps)
    ta_cw_log            = np.zeros(num_steps)

    # Initial motor activations (start at rest)
    ta_ccw_prev = 0.0
    ta_cw_prev  = 0.0

    # Inverted pendulum state
    body_sway_rad_state   = 0.0
    body_sway_vel_rad_s   = 0.0

    for step in range(num_steps):
        current_surface_sway = surface_sway_rad[step]

        # Body angle as seen in space: theta_B = theta_platform + theta_ankle
        # (In this simplified loop, theta_ankle ≈ body_sway_rad_state directly.)
        body_angle_rad = body_sway_rad_state + current_surface_sway

        # Apply sensory delay to body sway angle
        delayed_body_sway = delay_buffer[-1]
        delay_buffer       = np.roll(delay_buffer, 1)
        delay_buffer[0]    = body_angle_rad

        # Advance SNS controller one step
        ta_net_nm, ta_ccw_prev, ta_cw_prev = step_controller(
            neurons, synapses, bias_neuron_map,
            body_sway_rad    = delayed_body_sway,
            surface_sway_rad = current_surface_sway,
            ta_ccw_activation_prev = ta_ccw_prev,
            ta_cw_activation_prev  = ta_cw_prev,
            dt_ms = dt_ms
        )

        # Inverted pendulum dynamics:  J * theta_ddot + B * theta_dot - mgh * theta_B = Ta
        destab_torque  = DESTABILIZING_TORQUE_NM_PER_RAD * body_angle_rad
        damping_torque = JOINT_DAMPING_NMS_PER_RAD * body_sway_vel_rad_s
        body_sway_acc  = (ta_net_nm - damping_torque + destab_torque) / MOMENT_OF_INERTIA_KGM2

        body_sway_vel_rad_s += body_sway_acc  * TIMESTEP_S
        body_sway_rad_state += body_sway_vel_rad_s * TIMESTEP_S

        # Log
        body_sway_log[step]     = np.rad2deg(body_angle_rad)
        active_torque_log[step] = ta_net_nm
        surface_sway_log[step]  = np.rad2deg(current_surface_sway)
        ta_ccw_log[step]        = ta_ccw_prev
        ta_cw_log[step]         = ta_cw_prev

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time_s, surface_sway_log, label="Surface sway θ_P (deg)", color="black", linewidth=0.8)
    axes[0].plot(time_s, body_sway_log,    label="Body sway θ_B (deg)",    color="blue",  linewidth=0.8)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("TC4 SNS Controller — Peterka 2002 Test Condition 4")

    axes[1].plot(time_s, active_torque_log, label="Active torque Ta (N·m)", color="red", linewidth=0.8)
    axes[1].set_ylabel("Torque (N·m)")
    axes[1].legend(fontsize=8)

    axes[2].plot(time_s, ta_ccw_log, label="CCW activation", color="green",  linewidth=0.8)
    axes[2].plot(time_s, ta_cw_log,  label="CW  activation", color="orange", linewidth=0.8)
    axes[2].set_ylabel("Muscle activation [0,1]")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("tc4_simulation_result.png", dpi=150)
    plt.show()
    print("Simulation complete. Plot saved to tc4_simulation_result.png")
