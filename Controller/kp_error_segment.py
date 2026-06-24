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

"""
Biological analogue: AMPA receptors (glutamate, excitatory). 
Mixed Na+/K+ permeability gives a real-neuron reversal near 0 mV. 
The SNS uses +134 mV so the driving force (Es_exc - V) stays nearly constant across the operating range 
(changes only ~10%, from 194 mV to 174 mV), which keeps Eq. 13 linear.
"""

gs_inh = 0.558  # uS -- inhibatory synapse conductance
Es_inh = -100.0 # mV -- inhibitory reversal potential

"""
Biological analogue: GABA receptors (gamma-aminobutyric acid, inhibitory). 
Opens K+ channels. K+ reversal potential is -90 to -100mV.
set by the Nernst equation for intracellular K+.
Consistantly below the operating range thus the driving force is always negative and inhibitory.
"""

I_APP_MOD = R * Gm # 20.0 nA -- baseline current for modulator neurons

DEG_TO_NA = 1.0 # nA per degree of body tilt


    
# ====================================
# STEP 2: NEURONS
# ====================================
# Each nueron holds exactly one signal. The SNS has no digital memory --
# a neuron's voltage is the signal. You need one neuron per value you want
# to represent simultaneously in the circuit.

def build_neurons():
    # Helper: create one standard neuron -- avoids repeating Cm/Gm/Er every line
    # WHY define N() inside build neurons()?
    # It's a local helper -- it only makes sense here, so we keep it here.
    # This pattern is called a "closure" N() captures Er, Gm, Cm from the 
    # enclosing scope automatically
    def N(name):
        return Neuron(Cm_nF=Cm, Gm_uS=Gm, Er_mV=Er, name=name)
    
    # Return a dictionary: keys are string names, values are Neuron objects.
    # WHY a dict and not a list?
    # The simulation will need to look up "give me the neuron cakked theta_in"
    # by name. Dict lookup is 0(1) (instant). Searching a list for a name
    # takes O(n) time, which is inefficient for large lists.
    return {
        # ---- inputs (driven by I_app from the user) ----
        "theta_in":  N("theta_in"), # actual body angle
        "theta_ref": N("theta_ref"), # desired body angle

        # --- error layer subtraction
        # WHY two error neurons?
        # SNS voltages are always >=0. A single neuron cannot hold a negative error
        # Split so ccw fires when theta > theta_ref, ccw fires when 
        # theta < theta_ref. Together they represebt a signed error signal.
        "error_ccw": N("error_ccw"),
        "error_cw":  N("error_cw"),

        # --- Kp gain stage (double inhibition multiplication
        "kp_bias":     N("kp_bias"), # encodes the Kp gain value via I_app
        "kp_mod_ccw":  N("kp_mod_ccw"), # modulator -- sits at R when no gain input
        "kp_mod_cw":   N("kp_mod_cw"),
        "kp_prod_ccw": N("kp_prod_ccw"), # OUTPUT: kp * error_ccw
        "kp_prod_cw":  N("kp_prod_cw"), # OUTPUT: kp * error_cw
    }
# ====================
# STEP 3: SYNAPSES
# ====================
# why 10 synapses?
# Each synapse is one directed connection. The polarity -- excitatory or inhibatory
# determines the arithmetic at the postsynaptic neuron.
# Excitatory adds to the signal, while inhibitory subtracts from it.

def build_synapses():
    # Local helper to avoid  repeating all parameters every line.
    # n is the synapse number -- used in the name string for the diagram.
    def Exc(pre, post, n):
        return Synapse(gs_exc, Es_exc, Elo, Ehi, name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post) 
    
    def Inh(pre, post, n):
        return Synapse(gs_inh, Es_inh, Elo, Ehi, name=f"syn{n}_{pre}_to_{post}", pre=pre, post=post)
    
    return [
        # --- ccw error = theta_in - theta_ref ---
        # syn 1: theta_in excites error_ccw (adds_theta)
        Exc("theta_in", "error_ccw", 1),

        # syn 2: theta_ref inhibits error_ccw (subtracts_theta_ref)
        # Why inhib? we want error_ccw = theta -theta_ref.
        # Subtractionin SNS = one EXC + one INH to the same output neuron.
        Inh("theta_ref", "error_ccw", 2),

        # --- CW error = theta_ref - theta_in ---
        #swapped polarity: now theta_ref is EXC and theta_in is INH.
        # WHY? CW channel is active when theta < theta_ref, which means
        # theta_ref - theta > 0. Swapping hich inmput is EXC/INH flips the sign.
        Exc("theta_ref", "error_cw", 3),
        Inh("theta_in", "error_cw",  4),

        # --- kp gain: double-inhibition multiplication ---
        # Syn 5 & 6: kp_bias inhits both mod neurons.
        # WHY INH? kp_mod starts at U=R (fully blocking prod).
        # an inhibitory kp_biuas reduces mod below R, which REDUCES mod's
        # inhibition of prod, which INCREASES the output. More bias = more gain.
        Inh("kp_bias", "kp_mod_ccw",  5),
        Inh("kp_bias", "kp_mod_cw",  6),

        # Syn 7 and 9: error signal excites the product neurons (the signal to scale).
        Exc("error_ccw", "kp_prod_ccw", 7),
        Exc("error_cw", "kp_prod_cw",  9),

        # Syn 8 and 10: kp_mod inhibits kp_prod (the gate that controls how much of the error is passed through).
        Inh("kp_mod_ccw", "kp_prod_ccw", 8),
        Inh("kp_mod_cw", "kp_prod_cw",  10),
    ]
    
# ====================
# STEP 4: SIMULTANEOUS UPDATE LOOP
# ====================

def run_segment(neurons, synapses, I_app_dict, dt_ms=0.1, steps=2000):
    # Pre-allocate one array per neuron to record voltage over time.
    # WHY pre-allocate with np.zeros instead of appending in a loop
    # Appending to a list inside a loop copies the whole list each time (slow).
    # np.zeros(steps) reserves all the memory upfront -- writing by index is fast
    log = {name: np.zeros(steps) for name in neurons}

    for step in range(steps):

        # --- Phase A: compute ALL synaptic currents from CURRENT voltage ---
        # Start everyone at zero -- each synapse will add its contribution.
        I_syn = {name: 0.0 for name in neurons}
        for syn in synapses:
            I_syn[syn.post] += syn.current(neurons[syn.pre].V, neurons[syn.post].V)

        # --- Phase B: compute dV/dt for EVERY neuron (do not update V yet) ---
        dV = {}
        for name, nrn in neurons.items():
            I_app = I_app_dict.get(name, 0.0)
            I_leak = nrn.Gm * (nrn.Er - nrn.V)
            dV[name] = (I_leak + I_syn[name] + I_app) / nrn.Cm

        # --- Phase C: update ALL voltages at once ---
        for name, nrn in neurons.items():
            nrn.V += dV[name] * dt_ms
            log[name][step] = nrn.V

    return log

# ====================
# Step 5: Angle Encoding
# ====================

THETA_BIAS = (R / 2) * Gm  # 10.0 nA -- centers ±10 deg in the operating range

def build_iapp(theta_deg, theta_ref_deg=0.0, bp=4.26):
    # WHY default arguments?
    # theta_ref_deg=0.0 means "if you don't specify refernce angle,
    # # assume upright (0 degrees)". Default arguments let you call the
    # function as build_iapps(5.0) and still get sensible behavior.
    # same for bp=4.26 -- the tuned kp bias from TC4 PSO fit.
    #
    # WHY THETA_BIAS?
    # SNS synapses only transmit when V_pre is between Elo and Ehi.
    # A negative theta drives theta_in below Elo, silencing its synapses.
    # Adding THETA_BIAS to both inputs keeps them in the active range.
    # The offset cancels in the error subtraction:
    #   error_cw = (theta_ref + bias) - (theta_in + bias) = theta_ref - theta_in
    return {
        "theta_in":  theta_deg     * DEG_TO_NA + THETA_BIAS,
        "theta_ref": theta_ref_deg * DEG_TO_NA + THETA_BIAS,
        "kp_mod_ccw": I_APP_MOD, # holds kp_mod at U=R (fully blocking)
        "kp_mod_cw":  I_APP_MOD, # same for CW channel
        "kp_bias": bp,           # encodes the KP gain value
    }

#=====================
# STEP 7: DIagram
# ====================

# Node positions (x, y) in figure-coordinate units
NODE_POS = {
    "theta_in":     (1, 1.5),
    "theta_ref":    (1, -1.5),
    "error_ccw":    (4, 1.5),
    "error_cw":     (4, -1.5),
    "kp_bias":      (7, 0.0),
    "kp_mod_ccw":   (10, 1.5),
    "kp_mod_cw":    (10, -1.5),
    "kp_prod_ccw":  (13, 1.5),
    "kp_prod_cw":   (13, -1.5)
}

# Display label and color per neuron
NODE_LABELS = {
    "theta_in":    "theta_in",
    "theta_ref":   "theta_ref",
    "error_ccw":   "error_ccw",
    "error_cw":    "error_cw",
    "kp_bias":     "kp_bias",
    "kp_mod_ccw":  "kp_mod_ccw",
    "kp_mod_cw":   "kp_mod_cw",
    "kp_prod_ccw": "kp_prod_ccw",
    "kp_prod_cw":  "kp_prod_cw"
}

NODE_COLORS = {
    "theta_in":     "#F39C12", # Orange -- input
    "theta_ref":    "#F39C12", # Orange -- input
    "error_ccw":    "#F1948A", # Salmon -- error
    "error_cw":     "#F1948A", # Salmon -- error
    "kp_bias":      "#3498DB", # Blue -- bias
    "kp_mod_ccw":   "#2ECC71", # Green -- modulation
    "kp_mod_cw":    "#2ECC71", # Green -- modulation
    "kp_prod_ccw":  "#9B59B6", # Purple -- output
    "kp_prod_cw":   "#9B59B6"  # Purple -- output
}

def draw_diagram(synapses, log=None, theta_deg=0.0, theta_ref_deg=0.0, bp=4.26):
    fig, ax = plt.subplots(figsize=(16,6))
    ax.set_facecolor("white")
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    NW, NH = 1.6, 0.55 # node width and height

    # Draw each neuron as a rounded rectangle with a label
    for name, (x, y) in NODE_POS.items():
        # fancyBboxPatch draws a rounded rectangle centered at (x, y)
        box = FancyBboxPatch(
            (x - NW/2, y - NH/2), NW, NH,
            boxstyle="round,pad=0.05",
            facecolor=NODE_COLORS[name],
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(box)

        # Neuron label -- show U value if a log was passed in
        label = NODE_LABELS[name]
        if log is not None:
            U = log[name][-1] - Er
            if name in ("theta_in", "theta_ref"):
                U -= THETA_BIAS
            label += f"\nU={U:+.2f}"

        ax.text(x, y, label, ha="center", va="center",
                fontsize=6.5, fontweight="bold", zorder=4)
            
    # Draw each synapse as a numbered arrow
    for syn in synapses:
        x1, y1 = NODE_POS[syn.pre]
        x2, y2 = NODE_POS[syn.post]
    
        # Green for excitatory, red for inhibatory
        color = "#27AE60" if syn.Es > 0 else "#c0392b"
        lw = 1.4 if syn.Es > 0 else 1.8

        # WHY shrinkA/ShrinkB? these push the arrow endpoints away from the 
        # box centers so the arrowhead lands on the box edge, not the center.
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    connectionstyle="arc3,rad=0.0", shrinkA=18,
                                    shrinkB=18,),
                                zorder=2,
                    )
        
        # Syanpse number label at arrow midpoint
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        num = syn.name.split("_")[0]   # "syn5_..." -> "syn5"
        ax.text(mx, my + 0.15, num, ha="center", va="bottom",
                fontsize=6, color=color, zorder=5)

    ax.set_title(
        f"Kp x error Bilateral SNS Segment\n"
        f"theta={theta_deg}deg  theta_ref={theta_ref_deg}deg  bp={bp} nA",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig("kp_error_diagram.png", dpi=150, bbox_inches="tight")
    plt.show()
            
# ====================
# Test block
# ====================

'''
if __name__ == "__main__":
    print("=== Step 1: parameter check ===")
    print(f" tau = Cm/Gm = {Cm/Gm:.1f} ms")
    print(f" Elo/Ehi ={Elo}/{Ehi} mV")
    print(f" I_APP_MOD = {I_APP_MOD} nA")
    print(f" 1 deg -> {1.0 * DEG_TO_NA / Gm:.1f} mV activation")
    print(f" 20 deg -> {20.0 * DEG_TO_NA / Gm:.1f} mV = full range R")
    
    print("=== Step 2: neuron check ===")
    neurons = build_neurons()

    # WHY loop with .items()?
    # dict.items() gives you (key, value) pairs one at a time.
    # name is the string key, nrn is Neuron object.
    for name, nrn in neurons.items():
        U = nrn.V - Er # activation above rest; should be 0.0 at startup
        print(f" {name:<14} v = {nrn.V:.1f} mV U = {U:.1f} mV")
    
    print("=== Step 3: synapse check ===")
    synapses = build_synapses()

    for syn in synapses: 
        kind = "Exc" if syn.Es > 0 else "INH"
        # Extract the synapse number from the name (e.g "syn3_" "3")
        num = syn.name.split("_")[0]
        print(f" {num:<6} {kind} {syn.pre:<14} -> {syn.post}")
    
    print("=== Step 4: zero-input stability check ===")
    neurons = build_neurons()
    synapses = build_synapses()
    log = run_segment(neurons, synapses, I_app_dict={}, steps=500)

    # Check that all neurons stayed at rest (V = er = -60mV)
    for name in neurons:
        v_final = log[name][-1]
        drift = abs(v_final - Er)
        status = "OK" if drift < 0.001 else "DRIFT!"
        print(f" {name:<14} V_final = {v_final:.4f} mV [{status}]")

    print("=== Step 5: angle encoding ===")

    # WHY rebuild neurons each time?
    # run_segment() writes into nrn.V directly. If we reused the same 
    # neuron objects, this run would start from wherever Step 4 left off.
    # Fresh neurons always start at V = Er = -60mV (clean slate).

    neurons = build_neurons()
    synapses = build_synapses()

    theta_deg = 5.0
    theta_ref_deg = 0.0
    iapp = build_iapp(theta_deg, theta_ref_deg)
    log = run_segment(neurons, synapses, iapp, steps=2000)

    print(f"  theta = {theta_deg} deg,  theta_ref = {theta_ref_deg} deg,  bp = {iapp['kp_bias']} nA")
    print()
    for name, nrn in neurons.items():
        U = log[name][-1] - Er
        print(f"  {name:<14} U = {U:+.3f} mV")

    neurons  = build_neurons()
    iapp     = build_iapp(5.0, 0.0)
    log      = run_segment(neurons, synapses, iapp, steps=2000)

    print("=== Diagnostic: all neuron activations ===")
    for name in neurons:
        U = log[name][-1] - Er
        print(f"  {name:<14} U = {U:+.3f} mV")


    print()
    print("=== Step 6: Kp * error sweep ===")
    print(f"  {'theta':>6}   {'error_ccw':>10}   {'kp_prod_ccw':>12}")
    for theta in [0.0, 5.0, 10.0, 15.0]:
        neurons  = build_neurons()           # fresh each time
        iapp     = build_iapp(theta, 0.0)
        log      = run_segment(neurons, synapses, iapp, steps=2000)
        U_err    = log["error_ccw"][-1] - Er
        U_prod   = log["kp_prod_ccw"][-1] - Er
        print(f"  {theta:>6.1f}   {U_err:>10.3f}   {U_prod:>12.3f}")

    # Draw diagram
    draw_diagram(synapses, log=log, theta_deg=5.0, theta_ref_deg=0.0, bp=4.26)
'''

if __name__ == "__main__":
    import argparse

    # argparse builds a command-line interface automatically.
    # Each add_argument() defines one flag the user can pass.
    # type=float converts the string the user types into a Python float.
    # default= is what you get if the flag is omitted.
    parser = argparse.ArgumentParser(description="Kp x error bilateral SNS segment")
    parser.add_argument("--theta",     type=float, default=5.0,  help="Body angle (degrees)")
    parser.add_argument("--theta_ref", type=float, default=0.0,  help="Reference angle (degrees)")
    parser.add_argument("--bp",        type=float, default=4.26, help="Kp bias current (nA)")
    parser.add_argument("--steps",     type=int,   default=2000, help="Simulation steps")
    parser.add_argument("--no-plot",   action="store_true",      help="Skip diagram")
    args = parser.parse_args()

    neurons  = build_neurons()
    synapses = build_synapses()
    iapp     = build_iapp(args.theta, args.theta_ref, args.bp)
    log      = run_segment(neurons, synapses, iapp, dt_ms=0.1, steps=args.steps)

    print(f"\n=== Kp x error segment ===")
    print(f"  theta={args.theta} deg   theta_ref={args.theta_ref} deg   bp={args.bp} nA\n")
    for name, nrn in neurons.items():
        U = log[name][-1] - Er
        if name in ("theta_in", "theta_ref"):
            U -= THETA_BIAS
        print(f"  {name:<14} U = {U:+.3f} mV")

    if not args.no_plot:
        draw_diagram(synapses, log=log,
                     theta_deg=args.theta,
                     theta_ref_deg=args.theta_ref,
                     bp=args.bp)


