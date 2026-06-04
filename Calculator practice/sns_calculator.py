import tkinter as tk

# defining class for a single neuron
class Neuron:
    def __init__(self, Cm, Gm, Er):
        self.Cm = Cm
        self.Gm = Gm
        self.Er = Er
        self.V = Er # voltage starts at resting potential
    
    def update(self, dt, I_app, I_syn):
        # based on I = C * dV/dt
        # rearranged from Cm * dV/dt = I_total (Hilts eq. 8)
        I_leak = self.Gm * (self.Er - self.V)
        dv_dt = (I_app + I_syn + I_leak) / self.Cm

        #Eulers step (numerically solves an ODR using incremental time steps and rate of change)
        self.V = self.V + dv_dt * dt

# Defining the synapse class
class Synapse:
    def __init__(self, gs_max, Es, Elo, Ehi):
        self.gs_max = gs_max
        self.Es = Es
        self.Elo = Elo
        self.Ehi = Ehi
    
    def get_conductance(self, V_pre):
        # sigmoid function to determine conductance based on presynaptic voltage
        if V_pre < self.Elo:
            return 0.0
        elif self.Elo <= V_pre <= self.Ehi:
            return self.gs_max * (V_pre - self.Elo) / (self.Ehi - self.Elo)
        else:
            return self.gs_max

    def get_current(self, V_pre, V_post):
        Gs = self.get_conductance(V_pre)
        # G = 1/R, so I = G*R
        # Es-V_post is the difference b/w the reveserse potential and the postsynaptic voltage, which drives the current flow
        return Gs * (self.Es - V_post)

# Addition subnetwork
# takes two input signals a and b encoded as applied currents
# and produces an output signal c = a + b encoded as the voltage of the output neuron
def make_addition_network():
    # create neurons and synapses
    # return u1, u2, u_sum, syn1, syn2
    u1 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input a
    u2 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input b
    u_sum = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for output c
    syn1 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u1 to u_sum
    syn2 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u2 to u_sum
    return u1, u2, u_sum, syn1, syn2

# Subtraction subnetwork
# takes two input signals a and b encoded as applied currents
# and produces an output signal c = a - b encoded as the voltage of the output neuron
def make_subtraction_network():
    # create neurons and synapses
    # return u1, u2, u_diff, syn1, syn2
    u1 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input a
    u2 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input b
    u_diff = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for output c
    syn1 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u1 to u_diff (excitatory)
    syn2 = Synapse(gs_max=1e-6, Es=-0.1, Elo=-60e-3, Ehi=-40e-3) # synapse from u2 to u_diff (inhibitory)
    return u1, u2, u_diff, syn1, syn2

# Multiplication subnetwork
# requires a third neuron that acts as a modulator to control the gain of the synapse from u1 to u_prod based on the voltage of u2
def make_multiplication_network():
    # create neurons and synapses
    # return u1, u2, u3, u_prod, syn1, syn2, syn3
    u1 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input a
    u2 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input b 
    u3 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # # intermediate modulator neuron
    u_prod = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for output c
    syn1 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u1 to u_prod (excitatory)
    syn2 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u2 to u3 (excitatory)
    syn3 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u3 to u_prod (modulatory)
    return u1, u2, u3, u_prod, syn1, syn2, syn3

# Division subnetwork
# also requires a third neuron that acts as a modulator to control the reduces the gain of the synapse from u1 to u_quot proportionally based on the voltage of u2
def make_division_network():
    # create neurons and synapses
    # return u1, u2, u3, u_quot, syn1, syn2, syn3
    u1 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input a
    u2 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for input b 
    u3 = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # intermediate modulator neuron
    u_quot = Neuron(Cm=5e-9, Gm=1e-6, Er=-60e-3) # neuron for output c
    syn1 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u1 to u_quot (excitatory)
    syn2 = Synapse(gs_max=1e-6, Es=-0.1, Elo=-60e-3, Ehi=-40e-3) # synapse from u2 to u3 (inhibitory)
    syn3 = Synapse(gs_max=1e-6, Es=0, Elo=-60e-3, Ehi=-40e-3) # synapse from u3 to u_quot (modulatory inhibition)
    return u1, u2, u3, u_quot, syn1, syn2, syn3

# Network Simulator
def run(network_type, a, b, dt=1e-4, steps=1000):
    if network_type == 'addition':
        u1, u2, u_sum, syn1, syn2 = make_addition_network()
        u3 = None;syn3 = None
        u_out = u_sum
    elif network_type == 'subtraction':
        u1, u2, u_diff, syn1, syn2 = make_subtraction_network()
        u3 = None; syn3 = None
        u_out = u_diff
    elif network_type == 'multiplication':
        u1, u2, u3, u_prod, syn1, syn2, syn3 = make_multiplication_network()
        u_out = u_prod
    elif network_type == 'division':
        u1, u2, u3, u_quot, syn1, syn2, syn3 = make_division_network()
        u_out = u_quot
    else:
        raise ValueError('Invalid network type')

    # apply input currents to the input neurons
    for step in range(steps):
        I_app_u1 = a # input signal a as applied current to neuron 1
        I_app_u2 = b # input signal b as applied current to neuron 2

        # update presynaptic neurons
        u1.update(dt, I_app_u1, 0)
        u2.update(dt, I_app_u2, 0)

        # calculate synaptic currents based on presynaptic voltages
        if network_type in ['addition', 'subtraction']:
            I_syn = syn1.get_current(u1.V, u_out.V) + syn2.get_current(u2.V, u_out.V)
            if network_type == 'addition':
                u_sum.update(dt, 0, I_syn)
            else:
                u_diff.update(dt, 0, I_syn)
        
        elif network_type in ['multiplication', 'division']:
            I_syn = syn1.get_current(u1.V, u_out.V) + syn3.get_current(u3.V, u_out.V)
            I_syn_u_modulator = syn2.get_current(u2.V, u3.V)
            if network_type == 'multiplication':
                u_prod.update(dt, 0, I_syn)
                u3.update(dt, 0, I_syn_u_modulator)
            else:
                u_quot.update(dt, 0, I_syn)
                u3.update(dt, 0, I_syn_u_modulator)
    return u_out.V

#Calculator GUI
# caluclator GUI using tkinter to allow the user to input two numbers and select an operation
# then runs the corresponding network and displays the result
def build_gui():
    root = tk.Tk()
    root.title("SNS Calculator")

    entry_a = tk.Entry(root)
    entry_a.pack()

    entry_b = tk.Entry(root)
    entry_b.pack()
    result_label = tk.Label(root, text="Result: ")
    result_label.pack()

    def calculate(op):
        a = float(entry_a.get())*1e-9 # convert from nA to A
        b = float(entry_b.get())*1e-9 # convert from nA to A
        V_out = run(op, a, b)
        result_label.config(text=f"Result: {V_out:.6f} V")

    tk.Button(root, text="+", command=lambda: calculate('addition')).pack()
    tk.Button(root, text="-", command=lambda: calculate('subtraction')).pack()
    tk.Button(root, text="*", command=lambda: calculate('multiplication')).pack()
    tk.Button(root, text="/", command=lambda: calculate('division')).pack()

    root.mainloop()

build_gui()