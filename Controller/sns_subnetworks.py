from sns_neurons import Neuron, Synapse

Er      = -60.0   # mV
R       =  20.0   # mV  operating range
Ehi     = Er + R  # -40.0 mV
Elo     = Er      # -60.0 mV

gs_exc  =  0.115  # µS  (≈ unity gain at mid-range)
Es_exc  =  134.0  # mV

gs_inh  =  0.558  # µS
Es_inh  = -100.0  # mV

GM_DEFAULT = 1.0  # µS, membrane conductance used by make_neuron

def make_neuron(name):
    return Neuron(Cm_nF=5.0, Gm_uS=GM_DEFAULT, Er_mV=Er, name=name)

# Transmission subnetwork
# Two-neuron path where U_post ≈ target_gain × U_pre at steady state.
# gs_max is derived analytically using the FSA design formula (Hilts 2018):
#   At steady state with U_pre as the presynaptic activation:
#   U_post = (gs_max/R × U_pre × (Es-Er)) / (Gm + gs_max/R × U_pre)
#   Solving for gs_max at the mid-range operating point U_pre = R/2:
#   gs_max = target_gain × Gm × R / ((Es - Er) - target_gain × (R/2))
# Valid for target_gain ∈ (0, 1).
def make_transmission_network(target_gain, name_prefix="trans"):
    """
    Build a two-neuron transmission subnetwork.

    Args:
        target_gain:  Desired steady-state gain U_post / U_pre (between 0 and 1).
        name_prefix:  String prefix for neuron and synapse names.

    Returns:
        trans_pre, trans_post, syn
    """
    driving_force_mv = Es_exc - Er                  # 194 mV
    gs_transmission  = (target_gain * GM_DEFAULT * R) / (driving_force_mv - target_gain * (R / 2.0))

    trans_pre  = make_neuron(f"{name_prefix}_pre")
    trans_post = make_neuron(f"{name_prefix}_post")
    syn = Synapse(
        gs_transmission, Es_exc, Elo, Ehi,
        name=f"{name_prefix}_pre_to_post",
        pre=f"{name_prefix}_pre",
        post=f"{name_prefix}_post"
    )
    return trans_pre, trans_post, syn

# Addition subnetwork
# takes two input signals a and b encoded as applied currents
# and produces an output signal c = a + b encoded as the voltage of the output neuron
def make_addition_network():
    # create neurons and synapses
    # return add_pre1, add_pre2, add_sum, syn1, syn2
    add_pre1 = make_neuron("add_pre1") # neuron for input a
    add_pre2 = make_neuron("add_pre2") # neuron for input b
    add_sum = make_neuron("add_sum") # neuron for output c
    syn1 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="add_pre1_to_sum", pre="add_pre1", post="add_sum")
    syn2 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="add_pre2_to_sum", pre="add_pre2", post="add_sum")
    return add_pre1, add_pre2, add_sum, syn1, syn2

# Subtraction subnetwork
# takes two input signals a and b encoded as applied currents
# and produces an output signal c = a - b encoded as the voltage of the output neuron
def make_subtraction_network():
    # create neurons and synapses
    # return sub_pre1, sub_pre2, sub_diff, syn1, syn2
    sub_pre1 = make_neuron("sub_pre1") # neuron for input a
    sub_pre2 = make_neuron("sub_pre2") # neuron for input b
    sub_diff = make_neuron("sub_diff") # neuron for output c
    syn1 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="sub_pre1_to_diff", pre="sub_pre1", post="sub_diff")
    syn2 = Synapse(gs_inh, Es_inh, Elo, Ehi, name="sub_pre2_to_diff", pre="sub_pre2", post="sub_diff")
    return sub_pre1, sub_pre2, sub_diff, syn1, syn2

# Multiplication subnetwork
# requires a third neuron that acts as a modulator to control the gain of the synapse from u1 to u_prod based on the voltage of u2
def make_multiplication_network():
    # create neurons and synapses
    # return u1, u2, u3, u_prod, syn1, syn2, syn3
    mul_pre1 = make_neuron("mul_pre1") # neuron for input a
    mul_pre2 = make_neuron("mul_pre2") # neuron for input b
    mul_mod = make_neuron("mul_mod") # intermediate modulator neuron
    mul_prod = make_neuron("mul_prod") # neuron for output c
    syn1 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="mul_pre1_to_prod", pre="mul_pre1", post="mul_prod") # synapse from pre1 to mul_prod (excitatory)
    syn2 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="mul_pre2_to_mod", pre="mul_pre2", post="mul_mod") # synapse from pre2 to mul_mod (excitatory)
    syn3 = Synapse(gs_inh, Es_inh, Elo, Ehi, name="mul_mod_to_prod", pre="mul_mod", post="mul_prod") # inhibitory per FSA multiplication (Szczecinski 2017, Fig. 4)
    return mul_pre1, mul_pre2, mul_mod, mul_prod, syn1, syn2, syn3

# Division subnetwork
# also requires a third neuron that acts as a modulator to control the reduces the gain of the synapse from u1 to u_quot proportionally based on the voltage of u2
def make_division_network():
    # create neurons and synapses
    # return div_pre1, div_pre2, div_mod, div_quot, syn1, syn2, syn3
    div_pre1 = make_neuron("div_pre1") # neuron for input a
    div_pre2 = make_neuron("div_pre2") # neuron for input b
    div_mod = make_neuron("div_mod") # intermediate modulator neuron
    div_quot = make_neuron("div_quot") # neuron for output c
    syn1 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="div_pre1_to_quot", pre="div_pre1", post="div_quot") # synapse from div_pre1 to div_quot (excitatory)
    syn2 = Synapse(gs_inh, Es_inh, Elo, Ehi, name="div_pre2_to_mod", pre="div_pre2", post="div_mod") # synapse from div_pre2 to div_mod (inhibitory)
    syn3 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="div_mod_to_quot", pre="div_mod", post="div_quot") # synapse from div_mod to div_quot (modulatory inhibition)
    return div_pre1, div_pre2, div_mod, div_quot, syn1, syn2, syn3

# Derivative subnetwork
def make_derivative_network():
    # create neurons and synapses
    deriv_fast = Neuron(Cm_nF=2.0,  Gm_uS=1.0, Er_mV=Er, name="deriv_fast") # fast neuron that responds quickly to changes in input
    deriv_slow = Neuron(Cm_nF=10.0, Gm_uS=1.0, Er_mV=Er, name="deriv_slow") # slow neuron that integrates the input over time
    deriv_out = Neuron(Cm_nF=5.0, Gm_uS=1.0, Er_mV=Er, name="deriv_out") # neuron for the derivative output
    syn1 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="deriv_fast_to_out", pre="deriv_fast", post="deriv_out")
    syn2 = Synapse(gs_inh, Es_inh, Elo, Ehi, name="deriv_slow_to_out", pre="deriv_slow", post="deriv_out")
    return deriv_fast, deriv_slow, deriv_out, syn1, syn2

# Integral subnetwork
def make_integral_network():
    # create neurons and synapses
    int_u1  = Neuron(Cm_nF=5.0, Gm_uS=1.0, Er_mV=Er, name="int_u1")
    int_u2  = Neuron(Cm_nF=5.0, Gm_uS=1.0, Er_mV=Er, name="int_u2")
    syn1 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="int_u1_to_u2", pre="int_u1", post="int_u2")
    syn2 = Synapse(gs_exc, Es_exc, Elo, Ehi, name="int_u2_to_u1", pre="int_u2", post="int_u1")
    return int_u1, int_u2, syn1, syn2