import numpy as np

def simulate(neurons, synapses, I_app_dict, dt_ms=0.1, steps=500):
    # neurons is a list of Neuron objects
    # synapses is a list of Synapse objects
    # I_app_dict is a dictionary mapping neuron names to applied currents (in nA)
    # dt_ms is the time step for the simulation in milliseconds
    # steps is the total number of time steps to simulate

    # create a dictionary to store voltage traces for each neuron

    log = {name: np.zeros(steps) for name in neurons}

    for step in range(steps):

        # Compute all synaptic currents from LAST timestep's voltages
        I_syn = {name: 0.0 for name in neurons}
        for syn in synapses:
            V_pre  = neurons[syn.pre].V
            V_post = neurons[syn.post].V
            I_syn[syn.post] += syn.current(V_pre, V_post)

        # Compute all dV/dt 
        dV = {}
        for name, nrn in neurons.items():
            I_app  = I_app_dict.get(name, 0.0)
            I_leak = nrn.Gm * (nrn.Er - nrn.V)
            dV[name] = (I_leak + I_syn[name] + I_app) / nrn.Cm

        #  update ALL voltages simultaneously 
        for name, nrn in neurons.items():
            nrn.V += dV[name] * dt_ms
            log[name][step] = nrn.V

    return log