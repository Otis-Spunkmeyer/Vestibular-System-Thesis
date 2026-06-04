import numpy as np
import matplotlib.pyplot as plt
from sns_subnetworks import make_addition_network, Er, gs_exc, R, Es_exc
from simulate import simulate

# Simulation parameters
TIMESTEP_MS              = 0.1    # ms per simulation step
NUM_STEPS                = 1000   # 1000 steps = 100 ms total, well past tau=5 ms
INPUT_CURRENT_PRE1_NA    = 1.0    # nA applied to first input neuron
INPUT_CURRENT_PRE2_NA    = 0.5    # nA applied to second input neuron

# Build the addition network and unpack its components
input_neuron_1, input_neuron_2, sum_neuron, synapse_1, synapse_2 = make_addition_network()

# Assemble into the dict format that simulate() expects
neuron_dictionary = {
    "add_pre1": input_neuron_1,
    "add_pre2": input_neuron_2,
    "add_sum":  sum_neuron
}

synapse_list = [synapse_1, synapse_2]

# Only input neurons receive external drive; sum neuron gets no applied current
applied_current_dictionary = {
    "add_pre1": INPUT_CURRENT_PRE1_NA,
    "add_pre2": INPUT_CURRENT_PRE2_NA
}

# Run the synchronized simulation
voltage_log = simulate(
    neuron_dictionary,
    synapse_list,
    applied_current_dictionary,
    dt_ms=TIMESTEP_MS,
    steps=NUM_STEPS
)

# Analytical steady-state check (functional subnetwork Eq. 13)
# At steady state each input neuron's U = I_app, because Gm = 1 µS and no synaptic input
# So U_pre = I_app / Gm = I_app numerically (units: nA / µS = mV)
excitatory_driving_force_mv = Es_exc - Er        # 194 mV — how far reversal is from rest
activation_pre1             = INPUT_CURRENT_PRE1_NA
activation_pre2             = INPUT_CURRENT_PRE2_NA
total_input_activation      = activation_pre1 + activation_pre2
gain                        = gs_exc / R

analytical_steady_state_mv = (
    gain * excitatory_driving_force_mv * total_input_activation
) / (1 + gain * total_input_activation)

# Convert final simulated voltage to activation above rest
simulated_final_activation_mv = voltage_log["add_sum"][-1] - Er

print(f"Analytical steady-state U_sum : {analytical_steady_state_mv:.4f} mV")
print(f"Simulated  final        U_sum : {simulated_final_activation_mv:.4f} mV")

# Plot activation traces (U = V - Er) over time
time_axis_ms = np.arange(NUM_STEPS) * TIMESTEP_MS

plt.plot(time_axis_ms, voltage_log["add_pre1"] - Er, label="Input neuron 1")
plt.plot(time_axis_ms, voltage_log["add_pre2"] - Er, label="Input neuron 2")
plt.plot(time_axis_ms, voltage_log["add_sum"]  - Er, label="Sum neuron")
plt.axhline(analytical_steady_state_mv, color="black", linestyle="--",
            label="Analytical steady state")

plt.xlabel("Time (ms)")
plt.ylabel("Activation U (mV)")
plt.title("Addition Network: Simulated vs Analytical Steady State")
plt.legend()
plt.tight_layout()
plt.show()
