# defining class for a single neuron
class Neuron:
    def __init__(self, Cm_nF, Gm_uS, Er_mV, name="neuron"):
        self.name = name
        self.Cm = Cm_nF
        self.Gm = Gm_uS
        self.Er = Er_mV
        self.V = Er_mV # voltage starts at resting potential

        # computing the time constant (tau) of the neuron, which is the time it takes for the voltage to reach about 63% of its final value in response to a step input current
        self.tau = self.Cm / self.Gm

    # U is the activation above the resting potential, which is used in the sigmoid function to determine synaptic conductance
    @property
    def U(self):
        return self.V - self.Er
    
    def update(self, dt_ms, I_app_nA, I_syn_nA):
        # based on I = C * dV/dt
        # rearranged from Cm * dV/dt = I_total (Hilts eq. 8)
        I_leak = self.Gm * (self.Er - self.V)
        dv_dt = (I_app_nA + I_syn_nA + I_leak) / self.Cm

        #Eulers step (numerically solves an ODR using incremental time steps and rate of change)
        self.V = self.V + dv_dt * dt_ms

# Defining the synapse class
class Synapse:
    def __init__(self, gs_max_uS, Es_mV, Elo_mV, Ehi_mV, name="syn", pre=None, post=None):
        self.gs_max = gs_max_uS
        self.Es = Es_mV
        self.Elo = Elo_mV
        self.Ehi = Ehi_mV
        self.name = name
        self.pre = pre
        self.post = post
        self.R = Ehi_mV - Elo_mV # range of presynaptic voltage over which the synapse is active
    
    def conductance(self, V_pre_mV):
        # sigmoid function to determine conductance based on presynaptic voltage
        if V_pre_mV < self.Elo:
            return 0.0
        elif self.Elo <= V_pre_mV <= self.Ehi:
            return self.gs_max * (V_pre_mV - self.Elo) / (self.R)
        else:
            return self.gs_max

    def current(self, V_pre_mV, V_post_mV):
        Gs = self.conductance(V_pre_mV)
        # G = 1/R, so I = G*R
        # Es-V_post is the difference b/w the reveserse potential and the postsynaptic voltage, which drives the current flow
        return Gs * (self.Es - V_post_mV)