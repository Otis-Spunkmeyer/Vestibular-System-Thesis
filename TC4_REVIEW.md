# TC4 SNS Controller — Comprehensive Design Review

**Project:** Vestibular balance thesis — SNS replication of Peterka 2002 Test Condition 4  
**Base architecture:** McNeal & Hunt 2026 (split-derivative SNS controller)  
**TC4 extension:** Dual-channel sensory integration with Peterka 2002 weights  
**All code lives in:** `Controller/`

---

## 0. Quick Start

### Run everything from scratch

```bash
cd Controller

# Generate all reference/proof figures (17 PNG files)
python draw_reference_figures.py

# Run the TC4 simulation and view output plot
python sns_tc4_controller.py

# View network topology diagram
python visualize_tc4_network.py
```

### Expected output files

| Command | Output files |
|---------|-------------|
| `draw_reference_figures.py` | `reference_seg1_neuron.png` through `reference_proof4_final.png` (17 files) |
| `sns_tc4_controller.py` | `tc4_simulation_result.png` (3-panel: angle, torque, muscle activation vs. time) |
| `visualize_tc4_network.py` | `tc4_network_diagram.png` (full network topology) |

### "I'm lost — read in this order"

1. **Section 1** — What SNS is and why it's used here  
2. **Section 4** — What Peterka TC4 conditions mean physically  
3. **Section 3** — How FSA turns an equation into a synapse conductance  
4. **Section 5** — The full network stage by stage  
5. **Section 7** — How the simulation loop works  
6. **Section 10** — What's not done yet and why it matters  

---

## 1. Background: Why SNS?

### What is a Synthetic Nervous System?

A Synthetic Nervous System (SNS) is a network of artificial neurons whose dynamics are modeled after biological non-spiking interneurons — neurons that do not fire action potentials but instead maintain graded membrane potentials. Each neuron integrates input currents and settles to a steady-state voltage that encodes the computed output signal.

The key difference from a rate-coded neural network or a trained deep network:  
**The synapse conductances are analytically designed from target equations, not learned from data.**

This means you start with the computation you want (e.g., "this neuron should output 0.70 times its input") and derive the exact conductance values that make that happen. The method for doing this is called Functional Subnetwork Analysis (FSA), described in Section 3.

### Why non-spiking, conductance-based neurons?

Non-spiking neurons are appropriate here because:
- **They implement smooth, graded computations** — the steady-state output is an algebraic function of the inputs, not a firing rate approximation
- **They are analytically tractable** — setting dV/dt = 0 gives a closed-form expression (Eq. 13) that can be inverted to find the required synapse conductance
- **They match biological interneurons** — postural control interneurons in the spinal cord and brainstem are non-spiking integrators

### The operating range R = 20 mV

All neurons operate in a voltage window:
- Resting potential: `Er = −60 mV`
- Lower bound: `Elo = Er = −60 mV` (silent)
- Upper bound: `Ehi = Er + R = −40 mV` (fully active)
- Operating range: `R = 20 mV`

A neuron's **activation** is `U = V − Er ∈ [0, 20] mV`. This maps to muscle activation as `A_MJ = U / R ∈ [0, 1]`, which feeds into the Hill-type muscle model. The 20 mV window is not an approximation — it is the design space. All FSA equations are derived assuming inputs and outputs lie within `[0, R]`.

### Why this approach is powerful

Because the network computation is analytically designed, you can:
- Guarantee a neuron computes exactly the right function at its design point
- Add or remove pathways by calculating the new required conductances
- Tune the overall gain by changing only the 4 bias currents (Section 6)
- Trace any output back through the network to the input equations algebraically

---

## 2. SNS Neuron and Synapse Math

### Neuron ODE

Every neuron follows the conductance-based leaky integrator:

```
Cm · dV/dt = I_leak + I_syn + I_app
```

Where each term means:

| Term | Formula | Meaning |
|------|---------|---------|
| `Cm · dV/dt` | — | Rate of voltage change scaled by membrane capacitance |
| `I_leak` | `Gm · (Er − V)` | Passive leak current pulling V back toward Er |
| `I_syn` | `Σ Gs,i · (Es,i − V)` | Sum of all synaptic currents (see below) |
| `I_app` | given externally | Applied current from sensor inputs or bias |

**Parameters:**

| Symbol | Default | Units | Role |
|--------|---------|-------|------|
| `Cm` | 5 nF | nanofarads | Membrane capacitance → sets time constant τ = Cm/Gm |
| `Gm` | 1 µS | microsiemens | Membrane (leak) conductance |
| `Er` | −60 mV | millivolts | Resting potential |
| `τ` | 5 ms | milliseconds | Time constant = Cm/Gm |

Special cases: `deriv_fast` uses `Cm = 2 nF` → `τ = 2 ms`; `deriv_slow` uses `Cm = 10 nF` → `τ = 10 ms`.

### Steady-state: Eq. 13

At steady state, `dV/dt = 0`. Solving for `U* = V* − Er`:

```
         Σ(gs,i/R · Upre,i · ΔEs,i) + Iapp
U*  =  ─────────────────────────────────────
          1 + Σ(gs,i/R · Upre,i)
```

where `ΔEs,i = Es,i − Er` (driving force) and `Upre,i = Vpre,i − Elo` (presynaptic activation).

This is the master equation for FSA design. Given target inputs and a target output, you choose `gs` values to satisfy it.

**Worked example:** Standard neuron, no synapses, `I_app = 5 nA`:
```
U* = 5 nA / (1 µS) = 5 mV          (25% of R)
V* = Er + U* = −55 mV
```

**Worked example:** Standard neuron, one excitatory synapse, `Upre = 10 mV` (mid-range), `gs = 0.115 µS`:
```
U* = (0.115/20 · 10 · 194) / (1 + 0.115/20 · 10)
   = (11.155) / (1.0575)
   ≈ 10.5 mV    ≈ Upre    (transmission at mid-range ✓)
```

### Piecewise-linear activation function

The synaptic conductance is not a sigmoid — it is a piecewise-linear (PWL) ramp:

```
         0                          if Vpre < Elo
Gs(Vpre) = gs_max · (Vpre − Elo)/R   if Elo ≤ Vpre ≤ Ehi
         gs_max                     if Vpre > Ehi
```

This is linear in the operating range `[Elo, Ehi]`, zero below it, and saturated above it. The linearity inside the operating range is what makes Eq. 13 a clean algebraic equation rather than a nonlinear transcendental equation.

### Synaptic current

```
I_syn,i = Gs,i(Vpre) · (Es,i − Vpost)
```

- **Excitatory synapse** (`Es = +134 mV`): `Es − Vpost ≈ +194 mV` (large positive driving force → inward positive current → depolarizes post-synaptic neuron)
- **Inhibitory synapse** (`Es = −100 mV`): `Es − Vpost ≈ −40 mV` (negative driving force → hyperpolarizes post-synaptic neuron)

---

## 3. FSA Design — How gs is Chosen

Functional Subnetwork Analysis (Szczecinski & Quinn 2017) is the method used to design every synapse conductance in this network. The idea is:

1. Write down what you want the neuron to compute (e.g., `U_out = k · U_in`)
2. Substitute into Eq. 13 and solve for `gs`
3. The result is an exact conductance value — no training, no approximation

### Unity-gain excitatory synapse (gs_exc = 0.115 µS)

Goal: `U_out = U_in` at the mid-range design point `U_in = R/2 = 10 mV`.

From Eq. 13 with one excitatory synapse:
```
k · R/2 = (ge/R · R/2 · ΔEe) / (Gm + ge/R · R/2)
```

Solving for `ge` (with `k = 1`, `ΔEe = 194 mV`, `Gm = 1 µS`, `R = 20 mV`):
```
ge = Gm · R / (ΔEe − R)
   = 1 · 20 / (194 − 20)
   = 20 / 174
   ≈ 0.115 µS
```

This value (`gs_exc = 0.115 µS`) is used for all addition subnetworks and all standard excitatory connections.

### Inhibitory synapse for cancellation (gs_inh = 0.558 µS)

Goal: `U_out = U_A − U_B` (subtraction). The trick is to set the output to zero when both inputs are equal — i.e., the numerator of Eq. 13 is zero when `U_A = U_B`.

Numerator with one exc + one inh synapse:
```
ge · ΔEe · U_A + ginh · ΔEinh · U_B = 0   when U_A = U_B
```

Since `ΔEe > 0` and `ΔEinh < 0`:
```
ginh = ge · ΔEe / |ΔEinh|
     = 0.115 · 194 / 40
     = 22.31 / 40
     ≈ 0.558 µS
```

This ensures exact cancellation. The result, `U_out = U_A − U_B`, follows from scaling analysis of Eq. 13.

### Transmission synapse for fractional gain (GS_WG, GS_WP)

Goal: `U_out = k · U_in` for arbitrary `0 < k < 1`. Solve Eq. 13 at the design point `U_in = R/2`:

```
gs = k · Gm · R / (ΔEe − k · R/2)
```

For `k = 0.30` (graviceptive weight Wg):
```
GS_WG = 0.30 · 1 · 20 / (194 − 0.30 · 10)
       = 6 / 191
       ≈ 0.031 µS
```

For `k = 0.70` (proprioceptive weight Wp):
```
GS_WP = 0.70 · 1 · 20 / (194 − 0.70 · 10)
       = 14 / 187
       ≈ 0.075 µS
```

These two values implement the Peterka 2002 TC4 sensory re-weighting analytically.

### Why you can't just change gs to tune the overall gain

Synapse conductances are fixed by the computation. Changing `gs_exc` from 0.115 would break the unity-gain addition property. The only free tuning parameters are the **bias currents** `b_α` applied to the modulator neurons in Stage 5 (Section 6).

---

## 4. Peterka 2002 TC4 — What the Conditions Mean

### The six test conditions (TC1–TC6)

Peterka 2002 studies human postural control under 6 conditions formed by combining sensory availability:

| TC | Vision | Surface | Dominant sense |
|----|--------|---------|---------------|
| TC1 | Normal | Fixed | All 3 senses |
| TC2 | Eyes closed | Fixed | Somatosensory + Vestibular |
| TC3 | Normal | Sway-referenced | Visual + Vestibular |
| TC4 | Eyes closed | Sway-referenced | **Vestibular only** |
| TC5 | Sway-referenced visual | Fixed | Somatosensory + Vestibular |
| TC6 | Sway-referenced visual | Sway-referenced | Vestibular only (all conflict) |

**Sway-referenced surface** means the platform tilts to track the subject's ankle angle in real time, making the ankle stretch receptor (proprioceptive) signal uninformative — the ankle always feels like it's at a neutral position regardless of body sway.

### TC4 specifically

- Eyes closed: no visual input
- Sway-referenced surface: proprioceptive signal is unreliable
- **Only the vestibular (graviceptive) system provides reliable body orientation**
- The brain must rely on vestibular signals weighted against a noisy proprioceptive background

Despite this, the brain does not completely ignore the proprioceptive channel. Instead it **re-weights** the two channels:
- Graviceptive weight: `Wg = 0.30`
- Proprioceptive weight: `Wp = 0.70`

The TC4 error signal is:
```
e = −Wg · θ_body − Wp · (θ_body − θ_surface)
  = −(Wg + Wp) · θ_body + Wp · θ_surface
  = −θ_body + 0.70 · θ_surface
```

In the SNS, the sign convention is flipped (restoring torque is positive), giving:
```
U_err = θ_body − 0.70 · θ_surface
```

### What the PRTS stimulus is

The PRTS (Pseudo-Random Ternary Sequence) is a deterministic surface perturbation signal that:
- Takes values in {−A, 0, +A} with `A = 1°` for TC4 (1-degree peak-to-peak)
- Is generated by a 5-stage shift register with modulo-3 feedback
- Has 3^5 − 1 = 242 states → period = 242 × 0.25 s = 60.5 s
- Contains energy at discrete frequencies from ~0.017 Hz to ~2 Hz

The PRTS is used because it allows frequency response function (FRF) identification: you can compute `H(f) = G_xy(f) / G_xx(f)` from the cross-spectrum of surface perturbation and body sway, giving gain (dB) and phase (deg) at every test frequency simultaneously from a single 60-second trial.

### What the target FRF looks like

For TC4, the body sway FRF (body angle / surface angle) shows:
- Gain: approximately 0–6 dB across 0.05–2 Hz (body sway exceeds surface sway by a small amount)
- Phase: negative (body lags surface), increasing in magnitude with frequency
- Resonance peak near 0.1–0.3 Hz (natural frequency of the inverted pendulum + controller)

Matching this FRF quantitatively is the main validation target once the Hill muscle model and PSO are implemented (see TODOs 1 and 2).

---

## 5. Network Architecture

### Signal flow overview (CCW side, one half of bilateral pair)

```
Physical inputs:
  θ_body (rad) ──→ angle_to_current_na() ──→ split_to_bilateral_na()
  θ_surface (rad) ──→ same adapters
  T_muscle (N) ──→ tension_to_ib_current_na()

Stage 1 — Bilateral inputs (4 neurons):
  bs_ccw, bs_cw     ← θ_body split
  ss_ccw, ss_cw     ← θ_surface split

Stage 2 — TC4 Sensory Integration (8 neurons, CCW side shown):
  sub_diff_ccw  = bs_ccw − ss_ccw      [exc: ge=0.115, inh: ginh=0.558]
  wg_ccw        = 0.30 × bs_ccw        [trans: GS_WG=0.031]
  wp_ccw        = 0.70 × sub_diff_ccw  [trans: GS_WP=0.075]
  error_ccw     = wg_ccw + wp_ccw      [add: ge=0.115 each]
  ─────────────────────────────────────────────────────────────────
  → U_error = θ_body − 0.70·θ_surface = e(t)

Stage 3 — Derivative Estimation (6 neurons, CCW side shown):
  deriv_fast_ccw ← error_ccw  [Cm=2nF,  τ=2ms]
  deriv_slow_ccw ← error_ccw  [Cm=10nF, τ=10ms]
  deriv_out_ccw  = fast − slow [exc + inh, same gs as sub_diff]
  ─────────────────────────────────────────────────────────────────
  → D(s) ≈ 8ms·s·E(s)  for ω ≪ 100 rad/s  →  d(t) ≈ 8ms·ė(t)

Stage 4 — Co-activation (1 neuron, shared):
  coact_node = deriv_out_ccw + deriv_out_cw  [add: ge=0.115 each]
  ─────────────────────────────────────────────────────────────────
  → |d(t)| (unsigned derivative magnitude; CCW+CW are half-rectified)

Stage 5 — Bias-gated Gain (16 neurons = 4 pathways × 2 sides):
  kp: kp_mod_ccw [I_app=bp] + kp_prod_ccw ← error_ccw + kp_mod_ccw
  kd: kd_mod_ccw [I_app=bd] + kd_prod_ccw ← deriv_out_ccw + kd_mod_ccw
  kc: kc_mod_ccw [I_app=bc] + kc_prod_ccw ← coact_node + kc_mod_ccw
  kt: kt_mod_ccw [I_app=bt] + kt_prod_ccw ← ib_input + kt_mod_ccw
  ─────────────────────────────────────────────────────────────────
  → Kp·e, Kd·d, Kc·|d|, Kt·Ib  (gains proportional to bp,bd,bc,bt)

Stage 6 — Type-Ib input (1 neuron, shared):
  ib_input  [I_app = tension_to_ib_current_na(T)]
  ─────────────────────────────────────────────────────────────────
  → Ib(t) current from Golgi tendon organ tension feedback

Stage 7 — Motor Output (2 neurons):
  ta_ccw = sum(kp_prod_ccw, kd_prod_ccw, kc_prod_ccw, kt_prod_ccw)
  ta_cw  = sum(kp_prod_cw,  kd_prod_cw,  kc_prod_cw,  kt_prod_cw)
  ─────────────────────────────────────────────────────────────────
  → motor_voltage_to_activation(ta_ccw.V) → A_MJ ∈ [0,1]
  → compute_net_torque_nm(A_ccw, A_cw)   → Ta (N·m)
```

### Stage-by-stage biological rationale

**Stage 1 — Bilateral split:**  
Neurons can only carry non-negative currents. Signed signals (body sway can be positive or negative) are represented by two half-wave rectified neurons. At any instant, only one of the pair (CCW or CW) carries current; the other is silent. This mirrors biological agonist/antagonist muscle pairs.

**Stage 2 — TC4 Sensory Integration:**  
This stage is the primary addition to McNeal & Hunt's base architecture. In humans, the brain does not simply use raw ankle angle for balance — it fuses vestibular (graviceptive) and proprioceptive information with condition-dependent weights. The subtraction subnetwork removes the surface motion component to extract the proprioceptive ankle error, then the two weighted channels are summed to form the combined error signal.

**Stage 3 — Derivative Estimation:**  
The derivative of the error is needed for the damping term in PD control. Direct differentiation is noise-amplifying; instead, two neurons with different time constants both receive the error signal, then their difference approximates the derivative. At steady state both outputs are equal; transiently, the fast neuron leads the slow one, and the difference is proportional to rate-of-change. This is analogous to a band-pass filter applied to the error.

Transfer function of `deriv_out`:
```
D(s) = E(s) · [1/(1+τf·s) − 1/(1+τs·s)]
     = E(s) · (τs−τf)·s / [(1+τf·s)(1+τs·s)]
     ≈ (τs−τf)·s · E(s)   for  ω ≪ 1/τs = 100 rad/s
     = 8ms · s · E(s)
```

**Stage 4 — Co-activation:**  
The `|d(t)|` term stiffens the ankle joint proportionally to the rate of sway, regardless of direction. The bilateral architecture provides this naturally: `deriv_out_ccw ≥ 0` when sway is accelerating CCW, `deriv_out_cw ≥ 0` when accelerating CW. Summing both gives the unsigned magnitude. This implements the co-contraction reflex seen in human balance under perturbation.

**Stage 5 — Bias-gated Gain:**  
Each control pathway (Kp, Kd, Kc, Kt) needs an independently adjustable gain. This cannot be done by changing synapse weights (those are fixed by the FSA computation). Instead, a modulator neuron converts an applied bias current into a DC voltage offset that shifts the operating point of the product neuron. The product neuron then integrates both the signal and the bias offset, making its output a scaled version of the signal where the scale factor depends on the bias. See Section 6 for full math.

**Stage 6 — Type-Ib Feedback:**  
Type-Ib Golgi tendon organs sense muscle tension and provide inhibitory feedback. In the SNS, this is modeled as an applied current to `ib_input` proportional to muscle tension. This pathway provides load-dependent stiffness modulation — as the muscle produces more force, the Ib feedback increases, slightly opposing further activation. This improves stability under load.

**Stage 7 — Motor Output:**  
The motor neuron sums all four pathway contributions via excitatory synapses. The output voltage maps linearly to muscle activation `A_MJ = clip((V + 60)/20, 0, 1)`. Two motor neurons (`ta_ccw`, `ta_cw`) drive opposing muscle groups; their difference produces the net ankle torque.

---

## 6. Bias-Gated Gain — How the Tunable Gains Work

### Why gains can't be set by changing gs

All synapse conductances are analytically derived from the computations they implement (FSA design). Changing `gs` would change the computation itself, not just scale the output. The only legitimate tunable degrees of freedom are the four bias currents.

### The mod + prod neuron pair

Each pathway consists of two neurons:

**Modulator neuron (mod):**
- No synaptic inputs
- Receives only `I_app = b_α` (the bias current)
- From Eq. 13 with zero synapses: `U*_mod = I_app / Gm = b_α / 1 µS`
- Output is a fixed DC voltage proportional to the bias. For `b_α = 4.26 nA`: `U*_mod = 4.26 mV`

**Product neuron (prod):**
- Receives two excitatory synapses: one from the signal neuron, one from the mod neuron
- Same conductance `ge = 0.115 µS` on both inputs (addition subnetwork design)
- From Eq. 13 (two exc. inputs):

```
U*_prod = K · (U_signal + U_mod)

where K = (ge/R) · ΔEe / (Gm + (ge/R) · (U_signal + U_mod)) ≈ const < 1
```

Substituting `U_mod = b_α / Gm`:
```
U*_prod = K · U_signal + K · b_α / Gm
        = K · U_signal + constant_offset
```

The constant offset `K · b_α / Gm` shifts the neuron's operating point. The gain on the signal term `K` is approximately proportional to `b_α` because `b_α` sets where on the nonlinear activation curve the neuron operates. Larger `b_α` → higher operating point → larger effective gain on the signal.

### Transistor analogy

This is exactly analogous to biasing a transistor into its active region:
- Bias current `b_α` → base bias voltage
- Signal `U_signal` → small-signal AC input
- `U*_prod` → amplified output

Change the base bias → change the transconductance → change the effective signal gain. In both cases, the "gain" is not a fixed property of the device — it depends on the operating point set by the bias.

### The 4 tunable parameters

| Bias | Current (nA) | Pathway | Signal it scales |
|------|-------------|---------|-----------------|
| `bp = 4.26` | proportional | Kp | error `e(t)` |
| `bd = 5.01` | directional derivative | Kd | derivative `d(t)` |
| `bc = 2.48` | co-activation | Kc | unsigned `|d(t)|` |
| `bt = 5.42` | Type-Ib | Kt | tension `Ib(t)` |

These values come from McNeal & Hunt 2026, Table IV (Split + kt condition). They must be re-optimized for TC4 conditions once the Hill muscle model is in place (see TODO-2).

---

## 7. Simulation Loop Mechanics

### Forward Euler integration

The simulation uses explicit (forward) Euler integration:
```
V(t + Δt) = V(t) + (dV/dt) · Δt
```

**Stability condition:** Δt must be much smaller than the shortest time constant in the network. With `τ_min = Cm_fast / Gm = 2 ms`:
```
Δt = 0.2 ms  →  Δt / τ_min = 0.1   (10× safety margin — stable)
```

If Δt were increased to 2ms or greater, the fast neurons would become numerically unstable and oscillate.

**Why all voltages update simultaneously:** Synaptic currents are computed from the **previous timestep's** voltages, then all voltages advance. This avoids ordering dependencies and matches the parallel nature of biological neural networks.

### Sensory delay buffer

Human sensorimotor delay is approximately 90 ms (afferent + processing + efferent). This is implemented as a circular buffer of length:
```
delay_steps = round(0.090 s / 0.0002 s) = 450 steps
```

At each timestep:
1. Read the oldest value from the buffer (`delay_buffer[-1]`) — this is the body sway angle 90 ms ago
2. Roll the buffer: shift all elements one position forward
3. Write the current body sway angle into `delay_buffer[0]`

This delay applies to the body sway angle entering Stage 2. The surface sway angle is used without delay (the platform motion is a direct mechanical input).

### Inverted pendulum dynamics

The body is modeled as a rigid inverted pendulum pivoting at the ankle:
```
J · θ̈  =  Ta − B · θ̇ − mgh · θ_body
```

| Term | Formula | Value | Meaning |
|------|---------|-------|---------|
| `J · θ̈` | — | — | Angular momentum equation |
| `Ta` | from SNS | — | Active ankle torque (restoring) |
| `B · θ̇` | 351 · θ̇ | — | Passive joint damping (opposing motion) |
| `mgh · θ_body` | 686.7 · θ_body | — | Destabilizing gravity torque (note: positive = destabilizing) |
| `J` | 63 kg·m² | — | Whole-body moment of inertia about ankle |
| `m` | 77.8 kg | — | Body mass |
| `h` | 0.9 m | — | Center of mass height |
| `B` | 351 N·m·s/rad | — | Joint damping coefficient |

The equation is integrated numerically:
```python
acc = (ta_net_nm - B * vel - mgh * angle) / J
vel += acc * dt
angle += vel * dt
```

### What to look for in the simulation output

A well-tuned controller running a TC4 PRTS trial should show:
- `body_sway` tracks `surface_sway` with a lag of approximately 90 ms (the sensory delay)
- Body sway amplitude is slightly larger than surface sway (positive FRF gain)
- Active torque is smooth and proportional to the sway angle and velocity
- Muscle activations `ta_ccw` and `ta_cw` alternate — only one active at a time (bilateral property)
- No unbounded growth (the system is stabilized)

If body sway grows without bound: the gains `bp,bd` are too low (insufficient restoring force) or the delay is causing phase reversal.

---

## 8. File Reference

| File | Purpose | Read first if... |
|------|---------|-----------------|
| `sns_neurons.py` | `Neuron` and `Synapse` class definitions | ...you need to understand the fundamental primitives |
| `sns_subnetworks.py` | Subnetwork factory functions, shared constants | ...you want to add or modify a subnetwork building block |
| `sns_tc4_controller.py` | Full TC4 network construction + simulation step + main loop | ...you want to change the network topology or run the simulation |
| `simulate.py` | Generic open-loop simulator (useful for unit testing single subnetworks) | ...you want to test a subnetwork in isolation |
| `prts_generator.py` | PRTS surface stimulus generator (Peterka 2002, 3^5 sequence) | ...you want to change the stimulus or verify the spectrum |
| `visualize_tc4_network.py` | Network topology diagram (McNeal & Hunt Fig. 3 style) | ...you want to inspect the network wiring visually |
| `draw_reference_figures.py` | All 17 reference and proof figures for the thesis | ...you are working on the figures or the mathematical derivations |

### Constants defined in `sns_subnetworks.py` (imported by the controller)

```python
Er      = -60.0   # mV  resting potential
R       =  20.0   # mV  operating range
Ehi     = -40.0   # mV
Elo     = -60.0   # mV
gs_exc  =  0.115  # µS  unity-gain excitatory synapse
Es_exc  = 134.0   # mV  excitatory reversal potential → ΔEe = 194 mV
gs_inh  =  0.558  # µS  cancellation inhibitory synapse
Es_inh  = -100.0  # mV  inhibitory reversal potential → ΔEinh = -40 mV
GM_DEFAULT = 1.0  # µS  standard membrane conductance
```

---

## 9. draw_reference_figures.py — Helper Function Reference

### 9.1 Palette Constants (module-level)

```python
BLACK = '#111111'   GREY   = '#555555'   RED    = '#C0392B'
BLUE  = '#2471A3'   GREEN  = '#1E8449'   ORANGE = '#E67E22'
BG    = '#FAFAFA'   # figure background
```

### 9.2 Low-Level Circuit Drawing Helpers

These draw individual electrical components in data coordinates. All take an `ax` (matplotlib Axes) as the first argument. Coordinates are in the axes' data space.

| Function | Key inputs | What it draws |
|----------|-----------|---------------|
| `W(ax, xs, ys, lw=1.8, c=BLACK)` | list of x, list of y | Polyline wire |
| `capacitor(ax, x, y0, y1, label, val)` | center-x, bottom y, top y | Capacitor symbol ═══ |
| `resistor(ax, x, y0, y1, label, val, variable=False)` | same | Resistor box □; yellow fill if variable |
| `battery(ax, x, y0, y1, label, color)` | same | Battery with +/− plates |
| `ground(ax, x, y)` | position | 3-line ground symbol |
| `current_arrow(ax, x, y_tip, length, label)` | position | Red upward current arrow |
| `node_dot(ax, x, y)` | position | Black junction dot |
| `rail(ax, x_lo, x_hi, y)` | span | Horizontal wire rail |

### 9.3 SNS Block Diagram Helpers

Used in the node-level figures (`fig_node_sensory`, `fig_node_derivative`, `fig_node_gain`). All coordinates are in data units on panels with `xlim=(0,10)`, `ylim=(0,12)`.

---

#### `_nd(ax, x, y, lbl, fc, NE='#2C3E50', w=1.9, h=0.75, fs=8)`
Draws a rounded-rectangle neuron node centred at (x, y).

| Param | Default | Meaning |
|-------|---------|---------|
| `fc` | required | Fill color hex string |
| `NE` | `'#2C3E50'` | Edge/border color |
| `w, h` | 1.9, 0.75 | Half-dimensions — arrow helper `_syn` must use the same `hw, hh` |
| `fs` | 8 | Font size for label |

**Gotcha:** `w` and `h` here are half-dimensions of the box (used as `FancyBboxPatch` size). When calling `_syn` into this node, pass `hw=w, hh=h` to match. Mismatch → arrows don't stop at the box edge.

---

#### `_syn(ax, x1, y1, x2, y2, clr, lbl='', hw=0.95, hh=0.375)`
Draws an arrow from node centre (x1,y1) to node centre (x2,y2), stopping at the box edges.

Edge clipping: computes the unit vector along the arrow, then finds how far it travels before hitting the source/target box boundary.

| Param | Default | Meaning |
|-------|---------|---------|
| `hw, hh` | 0.95, 0.375 | Match to target node's `w, h` |
| `lbl` | `''` | Label placed at arrow midpoint |

**Gotcha:** `hw` and `hh` should match the **target** node's dimensions (not source). When the same source fans out to multiple differently-sized targets, call `_syn` once per target with the appropriate `hw, hh`.

---

#### `_stp(ax, y, txt, sc, sbg, cx=5.0, fs=8.5)`
Draws a rounded step-box. Text placed with `va='top'` at `(cx, y)`.

| Param | Default | Meaning |
|-------|---------|---------|
| `sc` | required | Border color (same as proof step color) |
| `sbg` | required | Background fill (lighter shade of `sc`) |
| `cx` | 5.0 | Horizontal centre — use `cx=6.0` for proof panels (xlim 0–12) |

**Gotcha:** `cx` defaults to 5.0 for node panels (xlim 0–10). In proof panels (xlim 0–12), use `_pstp` instead, which sets `cx=6.0` automatically.

---

#### `_arrd(ax, y0, y1, cx=5.0)` / `_parr(ax, y0, y1)`
Downward grey arrow. `_parr` is `_arrd` with `cx=6.0` — use only in proof panels.

---

#### `_ptitle(ax, title, color)`
Colored header band at `y ∈ [11.35, 11.95]` on ylim=(0,12) panels. Call once per panel before drawing content.

---

### 9.4 Proof Figure Helpers

Used only in `fig_proof_error/derivative/gain/final`. Proof panels have `xlim=(0,12)`, `ylim=(0,12)`.

---

#### `_proof_fig(title)` → `(fig, ax_net, ax_proof)`
Creates the standard two-panel layout. Left panel (`ax_net`) is for the network fragment; right panel (`ax_proof`) is for the theorem and proof steps.

Returns `(fig, ax_net, ax_proof)`. Both axes already have `axis('off')` and correct xlim/ylim set.

---

#### `_thm_box(ax, y_top, title_line, body_lines)` → `float`
Draws the blue THEOREM header box and returns the y coordinate immediately below it.

| Param | Typical value | Meaning |
|-------|--------------|---------|
| `y_top` | 11.8 | Top of box in data coords — leave 0.2 units for the suptitle |
| `title_line` | `'THEOREM 1'` | Bold blue heading |
| `body_lines` | list of 2–3 strings | Theorem statement at 8.5pt |

**Returns:** `y_top − h − 0.55` where `h = 0.55 + n × 0.48`.  
The 0.55 clearance accounts for `round,pad=0.25` (0.25 units below the declared rectangle) plus 0.30 units breathing room. Without this clearance, the proof label overlaps the box border.

**Typical usage:**
```python
y = _thm_box(ax_p, 11.8, 'THEOREM 1', [
    r'First line of theorem statement',
    r'$U^* = ...$',
])
ax_p.text(0.4, y, 'Proof  (...):',
          fontsize=9, fontweight='bold', color=BLACK, va='top')
y -= 0.25
# then _pstp / _parr chain...
```

---

#### `_qed_box(ax, text, y_bot=0.1)`
Draws the red QED conclusion box. Box height is fixed at 1.05 data units.

**CRITICAL:** Always pass `y_bot` explicitly — do NOT rely on the default 0.1.  
With `ylim=(0,12)`, the default places the box near the bottom of the axes far below the proof steps.

**Pattern to follow after the last `_pstp`:**
```python
_pstp(ax_p, y, r'(4) last step text', ORANGE, '#FEF9E7')
y -= 1.4          # approximate height of a 3-line step box
_qed_box(ax_p, r'$\therefore$ conclusion   □', y_bot=y - 0.45)
```

For 2-line steps use `y -= 1.1` instead of 1.4.

---

#### `_pstp(ax, y, txt, sc, sbg, fs=8)`
Proof-panel step box (`cx=6.0`). Wrapper around `_stp`. Used for every numbered proof step.

**Vertical spacing pattern between steps:**
```python
_pstp(ax_p, y, step_text, color, bgColor)
y -= step_height        # 1.1 for 2-line, 1.5 for 3-line
_parr(ax_p, y, y-0.3)
y -= 0.45
# (next _pstp call...)
```

---

### 9.5 Public Figure Functions

All functions save PNG to `Controller/` via `SCRIPT_DIR = os.path.dirname(__file__)`.

| Function | Output | Content |
|----------|--------|---------|
| `fig_neuron()` | `reference_seg1_neuron.png` | RC circuit of single SNS neuron |
| `fig_activation()` | `reference_seg2_activation.png` | PWL activation function curve |
| `fig_synapse()` | `reference_seg2_synapse.png` | Synapse circuit + conductance curve |
| `fig_substitution()` | `reference_seg3_substitution.png` | Algebraic derivation of Eq. 13 |
| `fig_steady_state()` | `reference_seg4_steady_state.png` | Steady-state U* graphically |
| `fig_fsa_subnetworks()` | `reference_seg5_fsa.png` | FSA design for add/sub/transmission |
| `fig_peterka_eq15()` | `reference_seg5b_peterka.png` | Peterka 2002 Eq. 15 open-loop TF |
| `fig_mcneal_split_deriv()` | `reference_seg6_split_deriv.png` | Split-derivative architecture |
| `fig_seg6b_bias_split()` | `reference_seg6b_bias_split.png` | Bias-gated gain + transistor analogy |
| `fig_tc4_assembly()` | `reference_seg7_assembly.png` | Full TC4 assembly block diagram |
| `fig_node_sensory()` | `reference_seg7a_sensory.png` | Stage 2: FSA derivations (4 panels) |
| `fig_node_derivative()` | `reference_seg7b_derivative.png` | Stage 3: FSA derivations (3 panels) |
| `fig_node_gain()` | `reference_seg7c_gain.png` | Stage 5: gain node derivations (3 panels) |
| `fig_proof_error()` | `reference_proof1_error.png` | Proof 1/4: error signal e(t) |
| `fig_proof_derivative()` | `reference_proof2_derivative.png` | Proof 2/4: derivative d(t) |
| `fig_proof_gain()` | `reference_proof3_gain.png` | Proof 3/4: bias-gated gain |
| `fig_proof_final()` | `reference_proof4_final.png` | Proof 4/4: full control law a(t) |

**Known matplotlib warning (safe to ignore):**
```
UserWarning: This figure includes Axes that are not compatible with tight_layout
```
This fires because axes have `axis('off')`. It does not affect output — `bbox_inches='tight'` in `savefig()` handles the crop correctly.

---

## 10. TODO List

### TODO-1 — Replace proxy muscle model with Hill-type model

**Why it matters:** The three placeholder functions in `sns_tc4_controller.py` compute tension and torque using a crude linear proxy (`net_activation × Fmax`). This ignores the force-length curve (`fl`), force-velocity curve (`fv`), and passive elastic element (`fpe`) that together determine when and how much force a muscle produces. At short lengths or high velocities, the proxy significantly overestimates force. Any FRF comparison against Peterka 2002 data will be quantitatively wrong until this is fixed.

**Dependency:** None — this can be done independently of other TODOs.

**Estimated scope:** 2–4 days (Hill model equations are well-documented in McNeal & Hunt 2026 Appendix; the main work is implementing `fl`, `fv`, `fpe` curves and connecting them to the loop).

**Where to start:** `sns_tc4_controller.py`, lines 109–133:
```python
def compute_tension_proxy_n(...)     # → replace with Hill F(a, l, l_dot)
def tension_to_ib_current_na(...)    # → replace with clip(gIb * T_hill, 0, 10)
def compute_net_torque_nm(...)       # → replace with Lmoment*(F_ccw − F_cw)
```

**Required Hill model constants (McNeal & Hunt Table I):**

| Constant | Value |
|----------|-------|
| `Fmax` | 1500 N |
| Optimal fiber length `L0` | 0.41 m |
| Moment arm `Lmoment` | 0.049 m |
| Length range | 0.394 – 0.423 m |
| `vmax` | 1.5 L0/s |
| `fpmax` | 1.3 |
| `fvmax` | 1.2 |
| fl curve width | 0.75 |

**Equations to implement (McNeal & Hunt 2026, Eqs. 15–18):**
```
Activation dynamics:  τ_act(a, u) · da/dt = u − a                    (Eq. 16)
Force:                F = Fmax · a · fl(l) · fv(l_dot) + fpe(l)      (Eq. 17)
Kinematics:           l(θ_A) = L0 ∓ Lmoment · θ_A                    (Eq. 18)
Net torque:           Ta = Lmoment · (F_ccw − F_cw)                   (Eq. 15)
```

**Acceptance test:** Run the simulation with a step body sway of 5°. The Hill model should produce a roughly sinusoidal torque profile with amplitude bounded by `Fmax × Lmoment ≈ 73.5 N·m`. The proxy model produces a square wave proportional to activation — this is the key difference to verify.

---

### TODO-2 — Run PSO to optimize bias parameters for TC4 FRF match

**Why it matters:** The current bias values (`bp=4.26, bd=5.01, bc=2.48, bt=5.42`) are from McNeal & Hunt's best-fit for their specific plant model, simulator, and sensory conditions (TC4-equivalent but with a different muscle model). Re-optimizing against Peterka 2002 FRF data with the Hill model in place is required for quantitative validation.

**Dependency:** Must be done AFTER TODO-1 (optimizing against the proxy muscle model gives wrong parameters that won't transfer).

**Estimated scope:** 1–3 days for implementation; PSO runtime depends on particle count and FRF computation speed. Recommend 20–50 particles, 50–100 generations.

**Where to start:** Create a new file `pso_optimize.py`. The objective function:
1. Set `BIAS_PROPORTIONAL_NA, BIAS_DIRECTIONAL_DER_NA, BIAS_COACTIVATION_NA, BIAS_TYPE_IB_NA` to trial values
2. Run a full 60.5-second PRTS simulation to steady state (discard first 10s)
3. Compute `H(f) = G_xy / G_xx` between surface sway and body sway
4. Return `Σ [(gain_sim − gain_peterka)² + w·(phase_sim − phase_peterka)²]` across all PRTS frequencies

**Suggested search space:**

| Parameter | Current | Search range |
|-----------|---------|-------------|
| `bp` | 4.26 nA | [0.5, 15.0] nA |
| `bd` | 5.01 nA | [0.5, 15.0] nA |
| `bc` | 2.48 nA | [0.0, 10.0] nA |
| `bt` | 5.42 nA | [0.0, 15.0] nA |

**Acceptance test:** Simulated FRF gain within ±3 dB and phase within ±15° of Peterka 2002 TC4 data across all 15 PRTS test frequencies (0.05–1.75 Hz).

---

### TODO-3 — Wire Type-Ib feedback to real Hill muscle tension

**Why it matters:** The `ib_input` neuron currently receives current from `compute_tension_proxy_n()`. With the Hill model, true muscle tension `T_hill` is available and should be used directly.

**Dependency:** TODO-1 must be complete first.

**Estimated scope:** 1 hour (one-line change once Hill model is in place).

**Where to start:** `sns_tc4_controller.py`, `step_controller()`:
```python
# Current:
tension_proxy_n = compute_tension_proxy_n(ta_ccw_activation_prev, ta_cw_activation_prev)
ib_current_na   = tension_to_ib_current_na(tension_proxy_n)

# Replace with:
T_hill = hill_model.get_tension()   # from Hill implementation
ib_current_na = np.clip(ADAPTER_IB_SCALE_NA_PER_N * T_hill, 0.0, SNS_CLIP_MAGNITUDE_NA)
```

**Acceptance test:** `ib_input.V` should vary smoothly with ankle angle; should saturate at ~−50 mV during high-force phases.

---

### TODO-4 — Replace inverted pendulum with MuJoCo plant

**Why it matters:** The current plant (`J·θ̈ = Ta − B·θ̇ − mgh·θ`) is a single rigid body with no muscle-skeletal geometry, no joint limits, and no realistic inertia distribution. The MuJoCo model includes multi-segment body, muscle insertion points, and proper kinematics. This is required for the thesis's MuJoCo validation.

**Dependency:** TODO-1 and TODO-3 (Hill model must be in place for the MuJoCo muscle actuator to work).

**Estimated scope:** 3–7 days (MuJoCo integration, XML model setup, step function rewrite).

**Where to start:** The `__main__` block in `sns_tc4_controller.py` — the entire simulation loop needs to be re-written around `mujoco.mj_step()` calls with the SNS providing muscle activation at each timestep.

**Acceptance test:** With constant body sway = 0 and no perturbation, the MuJoCo model should hold the upright posture indefinitely (gravity + muscle stiffness balance).

---

### TODO-5 — Validate PRTS generator against Peterka 2002 Fig. 2A

**Why it matters:** The FRF identification relies on the PRTS having the correct power spectrum. If the register initialization or tap positions are wrong, the frequency content will differ from Peterka 2002 and the FRF comparison will be invalid.

**Dependency:** None.

**Estimated scope:** 2–4 hours.

**Where to start:** `prts_generator.py`. Generate the sequence and plot its power spectrum. Compare to Peterka 2002 Fig. 2A — the spectrum should be approximately flat with peaks at intervals of `1/60.5 Hz ≈ 0.0165 Hz`.

**Acceptance test:** The auto-spectrum `G_xx(f)` of the generated PRTS has roughly equal power at all 15 PRTS test frequencies cited in Peterka 2002 Table 2.

---

### TODO-6 — Implement FRF computation module

**Why it matters:** No FRF analysis code exists. Without it, there is no quantitative way to compare the simulation output against Peterka 2002 data. All validation is currently qualitative.

**Dependency:** None (can prototype with proxy muscle model to check the math; useful output pending TODO-1+2 for quantitative validity).

**Estimated scope:** 1–2 days.

**Where to start:** Create `frf_analysis.py`. Core computation:
```python
import numpy as np

def compute_frf(x, y, fs, nperseg=None):
    """Cross-spectral FRF estimate H(f) = Gxy(f) / Gxx(f)."""
    from scipy import signal
    f, Gxx = signal.welch(x, fs, nperseg=nperseg)
    f, Gxy = signal.csd(x, y, fs, nperseg=nperseg)
    H = Gxy / Gxx
    return f, np.abs(H), np.angle(H, deg=True)
```

Then plot gain (dB) and phase (deg) vs. frequency and overlay digitized Peterka 2002 TC4 data points.

**Acceptance test:** Running with a perfect linear system (known H) should recover H exactly. Running with the proxy simulation should produce a smooth FRF curve.

---

### TODO-7 — Check sensory delay on surface sway channel

**Why it matters:** Currently only the body sway angle is delayed by 90 ms. The surface sway proprioceptive channel may also have an afferent delay — check McNeal & Hunt 2026 Section III-A-3 for the specification.

**Dependency:** None.

**Estimated scope:** 30 minutes to 2 hours.

**Where to start:** `sns_tc4_controller.py`, `step_controller()`: check whether `surface_sway_rad` should pass through a separate delay buffer before being fed to `ss_ccw` and `ss_cw`.

---

## 11. Key References

### Peterka 2002 — *Sensorimotor integration in human postural control*

**What it contributes:** The target FRF data (TC4 gain and phase curves), the 6 test conditions definition, the sensory re-weighting model (Wg=0.30, Wp=0.70 for TC4), and the PRTS stimulus specification.

**Read for this project:**
- Section II: Experimental methods — PRTS stimulus, platform setup
- Section III: Results — TC4 FRF data (Fig. 3–5) — these are the target curves
- Section IV: Sensory re-weighting model — defines the TC4 error equation
- Table 2: Sensory weights per condition — where Wg=0.30, Wp=0.70 come from

---

### McNeal & Hunt 2026 — *Split-derivative control and tension feedback in a neuromechanical model of human balance*

**What it contributes:** The full SNS architecture (Stages 1–7 minus TC4's Stage 2), the Hill-type muscle model, the bias parameter optimization via PSO, and the split-derivative design rationale.

**Read for this project:**
- Section III-A: Controller design — SNS architecture, each stage explained
- Section III-A-2a: Bias-gated gain — mod/prod neuron pair
- Section III-A-4: Muscle model — Hill equations, all constants (Table I)
- Table IV: Bias parameters — starting values (`bp=4.26, bd=5.01, bc=2.48, bt=5.42`)
- Eq. 1: Final control law `a(t) = Kp·e + Kd·d + Kc·|d| + Kt·Ib`
- Fig. 3: Network diagram — reference for `visualize_tc4_network.py`

---

### Szczecinski & Quinn 2017 — *A functional subnetwork approach to designing synthetic nervous systems*

**What it contributes:** The FSA design method — the formal procedure for choosing synapse conductances to implement a target computation. Eq. 13 (the steady-state master equation) is central to every conductance value in this network.

**Read for this project:**
- Section II: FSA method — how to design gs from a target function
- Eq. 13: The steady-state membrane voltage equation
- Examples for addition, subtraction, transmission subnetworks

---

### Hilts 2018 — *Biological neural network design for motor control*

**What it contributes:** The transmission subnetwork design formula `gs = k·Gm·R / (ΔEe − k·R/2)` used for the TC4 sensory weights. Also the notation conventions used throughout this codebase.

**Read for this project:**
- Transmission subnetwork section — derivation of the gs formula
- Operating range conventions (R, Elo, Ehi, Er)

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **SNS** | Synthetic Nervous System — network of non-spiking, conductance-based artificial neurons analytically designed to implement target computations |
| **FSA** | Functional Subnetwork Analysis — method (Szczecinski & Quinn 2017) for choosing synapse conductances from target equations using Eq. 13 |
| **FRF** | Frequency Response Function — complex ratio H(f) = output/input in frequency domain; gives gain (dB) and phase (deg) at each frequency |
| **PRTS** | Pseudo-Random Ternary Sequence — deterministic {−A, 0, +A} surface perturbation signal used for broadband FRF identification |
| **TC4** | Test Condition 4 (Peterka 2002) — eyes closed, sway-referenced surface; only vestibular information is reliable |
| **CCW / CW** | Counter-clockwise / clockwise — bilateral encoding convention; CCW = positive sway direction, CW = negative |
| **U\*** | Steady-state neuron activation voltage `V* − Er` in [0, R] mV; the neuron's output signal |
| **gs** | Maximum synaptic conductance (µS); the free parameter FSA solves for |
| **Es** | Synaptic reversal potential (mV); determines whether the synapse is excitatory or inhibitory |
| **Er** | Resting membrane potential = −60 mV in this network |
| **R** | Operating range = Ehi − Elo = 20 mV; the window in which neurons compute |
| **Gm** | Membrane (leak) conductance = 1 µS for all standard neurons |
| **Cm** | Membrane capacitance (nF); sets the neuron's time constant τ = Cm/Gm |
| **τ** | Time constant — how quickly the neuron responds to input changes; τ = Cm/Gm |
| **PSO** | Particle Swarm Optimization — the algorithm used to tune the 4 bias parameters to match Peterka 2002 FRF data |
| **Hill model** | Hill-type muscle model — physically realistic model of muscle force production including force-length (fl), force-velocity (fv), and passive elastic (fpe) components |
| **Ib / Type-Ib** | Type-Ib Golgi tendon organ afferents — sensory neurons that signal muscle tension; provide inhibitory feedback to motor neurons |
| **Wg** | Graviceptive (vestibular) sensory weight = 0.30 for TC4 |
| **Wp** | Proprioceptive (somatosensory/ankle) sensory weight = 0.70 for TC4 |
| **Bilateral** | Architecture where signed signals are split into two non-negative half-waves (CCW and CW), each carried by a separate neuron |
| **Co-activation** | The `|d(t)|` term — simultaneous activation of agonist and antagonist muscles proportional to rate of sway, stiffening the joint |
| **Bias current** | Applied current `I_app = b_α` to a modulator neuron; the only tunable parameters in the network; control effective pathway gains |
| **ΔEe** | Excitatory driving force = `Es_exc − Er` = 134 − (−60) = 194 mV |
| **ΔEinh** | Inhibitory driving force = `Es_inh − Er` = −100 − (−60) = −40 mV |
| **Lmoment** | Muscle moment arm = 0.049 m (distance from ankle joint to muscle insertion) |
| **Fmax** | Maximum isometric muscle force = 1500 N |
