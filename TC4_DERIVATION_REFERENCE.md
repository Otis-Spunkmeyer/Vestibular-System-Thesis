# TC4 SNS — Algebraic Derivation Reference

**Purpose:** Walk from physical sensor angles to `a(t) = Kp·e + Kd·d + Kc·|d| + Kt·Ib`
using one equation (Eq. 13) applied repeatedly at each neuron. If you have not touched
this in a month, read §1–§3 to warm up, then walk §4 stop by stop.

---

## §1 — The One Equation You Need

Every steady-state neuron voltage in the SNS follows **Eq. 13** (McNeal & Hunt 2026):

```
          Σ (gs,i / R) · Upre,i · ΔEs,i  +  Iapp
U* =  ─────────────────────────────────────────────
              Gm  +  Σ (gs,i / R) · Upre,i
```

Symbols:

| Symbol | Meaning |
|--------|---------|
| `U = V − Er` | Membrane voltage relative to resting potential (mV) |
| `R = Ehi − Elo = 20 mV` | Operating range |
| `ΔEs,i = Es,i − Er` | Synapse reversal offset (mV) |
| `gs,i` | Max synapse conductance (µS) |
| `Gm = 1 µS` | Membrane leak conductance |
| `Iapp` | Externally applied current (nA) |

**The "1" in the denominator IS Gm** (= 1 µS, not a dimensionless 1). Units:
numerator in nA, denominator in µS → nA/µS = mV. Do not drop units.

**Design strategy (FSA):** Pick a desired `U*`, set `Upre = R/2` (mid-range input),
solve backward for `gs`. This is how every synapse conductance in the network was derived.

---

## §2 — All Parameters in One Place

### SNS Operating Point
| Symbol | Value | Meaning |
|--------|-------|---------|
| `Er` | −60 mV | Resting potential |
| `Elo` | −60 mV | Lower rail |
| `Ehi` | −40 mV | Upper rail |
| `R` | 20 mV | Operating range |
| `Gm` | 1 µS | Membrane conductance |

### Synapse Constants
Derived from FSA: one exc synapse, `Upre = R/2`, target `U* = R/2` (identity/unity gain).

| Type | `gs` (µS) | `Es` (mV) | `ΔEs = Es − Er` (mV) |
|------|-----------|-----------|----------------------|
| Excitatory | 0.115 | +134 | +194 |
| Inhibitory | 0.558 | −100 | −40 |

`gs_inh` derivation — inhibitory must cancel one excitatory at `Upre = R/2`:
```
gs_inh · |ΔEi| = gs_exc · ΔEe
gs_inh = 0.115 · 194 / 40 = 0.558 µS
```

### TC4 Sensory Weights (Peterka 2002, TC4, 1° PRTS amplitude)
| Symbol | Value | Channel |
|--------|-------|---------|
| `Wg` | 0.30 | Graviceptive (vestibular, eyes closed) |
| `Wp` | 0.70 | Proprioceptive (sway-referenced surface) |

### Transmission Conductances
Derived from FSA: target gain `k = Wk` at `Upre = R/2`:
```
gs = k · Gm · R / (ΔEe − k · R/2)

GS_WG = 0.30 · 1 · 20 / (194 − 0.30 · 10) = 6.0 / 191.0 ≈ 0.0314 µS
GS_WP = 0.70 · 1 · 20 / (194 − 0.70 · 10) = 14.0 / 187.0 ≈ 0.0749 µS
```

### Bias Parameters (McNeal & Hunt 2026 Table IV — starting point for PSO)
| Symbol | Value (nA) | Pathway controlled |
|--------|-----------|-------------------|
| `bp` | 4.26 | Proportional (Kp) |
| `bd` | 5.01 | Directional derivative (Kd) |
| `bc` | 2.48 | Co-activation (Kc) |
| `bt` | 5.42 | Type-Ib tension feedback (Kt) |

### Input Adapter Constant
```
gA = 180/π  nA/rad  ≈  57.3 nA/rad
```
Converts radians to a current such that 1° of sway = 1 nA. This keeps the SNS
in its linear range for typical postural sway amplitudes. Clipped at ±10 nA (≡ ±10°).

---

## §3 — The Bilateral Architecture

Every signed signal is split into two non-negative half-waves before entering the network:

```python
ccw_current = max(0,  I_signed)   # active when signal is positive / CCW
cw_current  = max(0, -I_signed)   # active when signal is negative / CW
```

**Key property:** Both halves are never simultaneously nonzero.

Consequences:
- All SNS voltages are ≥ 0 (the network cannot represent negative voltages)
- Signed signals are encoded as pairs of non-negative channels
- **The absolute value `|x|` is computed for free** by summing CCW + CW (see Stop 7)

**Sign convention in the diagram title** (`e = −Wg·BS − Wp·(BS−SS)`):  
This is the control-theoretic convention: `error = setpoint − output = 0 − sensory_estimate`.
The network itself computes `+e = BS − 0.70·SS` (a positive voltage when body sways CCW).
The bilateral split closes the negative feedback loop *structurally* — CCW sway activates
only the CCW channel, which drives a restoring torque through the muscle model.

---

## §4 — The Derivation: 9 Stops

### Stop 1 — Eq. 13 Warm-Up: Single Excitatory Synapse

Apply Eq. 13 with one excitatory synapse, no `Iapp`, `Upre = R/2 = 10 mV`:

```
U* = [(gs_e / R) · U_pre · ΔEe] / [Gm + (gs_e / R) · U_pre]
   = [(0.115/20) · 10 · 194]    / [1   + (0.115/20) · 10   ]
   = [11.155]                    / [1.0575]
   ≈ 10.55 mV
```

The numerator coefficient `(gs_e/R) · ΔEe = (0.115/20) · 194 = 1.116` appears repeatedly.
The denominator is `1.0575 ≈ 1`. So a single excitatory synapse at `Upre = R/2` gives
`U* ≈ 1.116 · Upre ≈ Upre`. It passes through the input with near-unity gain.

---

### Stop 2 — Stage 1: Input Adapter

Physical angle → current → bilateral split:

```
θ_body (rad)  ──→  I_bs = clip(57.3 · θ_body,  ±10 nA)
                       ↓
              split_to_bilateral:
                bs_ccw = max(0,  I_bs)
                bs_cw  = max(0, -I_bs)
```

Input neurons receive only `Iapp` (no synapses), so Eq. 13 reduces to:
```
U*_bs = Iapp / Gm = Iapp   [mV, since Gm = 1 µS]
```

**Numerical check — 5° CCW:**
```
5° = 0.0873 rad  →  I_bs = 57.3 · 0.0873 = 5.0 nA
bs_ccw = 5.0 nA, bs_cw = 0 nA
U*_bs_ccw = 5.0 mV,  U*_bs_cw = 0 mV
```

One millivolt of SNS voltage corresponds to one degree of sway.

---

### Stop 3 — Stage 2a: sub_diff (Subtraction Subnetwork)

`sub_diff_ccw` receives:
- Excitatory from `bs_ccw` (`gs_exc = 0.115 µS`, `ΔEe = +194 mV`)
- Inhibitory from `ss_ccw` (`gs_inh = 0.558 µS`, `ΔEi = −40 mV`)

**Apply Eq. 13 at the design point `U_bs = U_ss = R/2 = 10 mV`:**

```
Numerator:
  (gs_e/R)·U_bs·ΔEe  +  (gs_i/R)·U_ss·ΔEi
= (0.115/20)·10·(+194)  +  (0.558/20)·10·(−40)
= +11.155               +  (−11.16)
≈ 0
```

When `U_BS = U_SS`, output = 0. The excitatory and inhibitory numerator coefficients
are nearly equal (1.1155 vs 1.116 — the slight asymmetry is the `≈` in the result):

```
U*_sub_diff  ≈  U_bs − U_ss  =  θ_body − θ_surface
```

---

### Stop 4 — Stage 2b: Transmission Synapses (wg and wp)

`wg_ccw` — one excitatory synapse from `bs_ccw` with `gs = GS_WG = 0.0314 µS`:

```
U*_wg = [(0.0314/20) · 10 · 194] / [1 + (0.0314/20) · 10]
       = [3.046] / [1.0157]
       ≈ 3.0 mV  =  0.30 · R/2  =  Wg · U_pre   ✓
```

`wp_ccw` — same structure with `gs = GS_WP = 0.0749 µS`:
```
U*_wp  ≈  0.70 · U_sub_diff  =  Wp · (U_bs − U_ss)   ✓
```

**Why the denominators are ≈ 1:** At these small conductances, the denominator correction
`gs/R · Upre ≈ 0.016–0.037` → denominator ≈ 1.02–1.04 → ~2% error, acceptable.

---

### Stop 5 — Stage 2c: error (Addition Subnetwork)

`error_ccw` receives two excitatory inputs (both `gs_exc = 0.115 µS`):
- `wg_ccw` → `U_wg`
- `wp_ccw` → `U_wp`

Apply Eq. 13:
```
U*_error = [gs_e/R · U_wg · ΔEe  +  gs_e/R · U_wp · ΔEe]
           ───────────────────────────────────────────────────
                   Gm  +  gs_e/R · U_wg  +  gs_e/R · U_wp

         = [(gs_e/R · ΔEe) · (U_wg + U_wp)]
           ────────────────────────────────────────
              [Gm  +  (gs_e/R) · (U_wg + U_wp)]
```

Since `(gs_e/R) · (U_wg + U_wp)` is small relative to `Gm`, the denominator ≈ 1, giving:
```
U*_error  ≈  (gs_e / R) · ΔEe · (U_wg + U_wp)
           =  1.116 · (U_wg + U_wp)
           ≈  U_wg + U_wp                          [since 1.116 ≈ 1]
```

Substituting the upstream results:
```
U*_error  ≈  U_wg + U_wp
           =  Wg·U_bs  +  Wp·(U_bs − U_ss)
           =  0.30·θ_body  +  0.70·(θ_body − θ_surface)
           =  θ_body − 0.70·θ_surface
           =  e(t)   ✓
```

**This is the Peterka 2002 TC4 error signal.** The 0.70 weight was not trained —
it was embedded in `GS_WP` at the FSA design step.

---

### Stop 6 — Stage 3: Derivative Estimation (Laplace Analysis)

Both `deriv_fast_ccw` (τ = 2 ms) and `deriv_slow_ccw` (τ = 10 ms) receive the same
excitatory synapse from `error_ccw`. The neuron ODE in Laplace gives a first-order
low-pass for each:

```
H_fast(s) = 1 / (1 + τf·s)      τf = Cm/Gm = 2 nF / 1 µS = 2 ms
H_slow(s) = 1 / (1 + τs·s)      τs = Cm/Gm = 10 nF / 1 µS = 10 ms
```

`deriv_out_ccw` subtracts them (same subtraction design as sub_diff):

```
D(s)/E(s) = H_fast(s) − H_slow(s)
           = 1/(1+τf·s) − 1/(1+τs·s)
           = [(1+τs·s) − (1+τf·s)] / [(1+τf·s)(1+τs·s)]
           = (τs − τf)·s / [(1+τf·s)(1+τs·s)]
           = 8ms · s / [(1 + 2ms·s)(1 + 10ms·s)]
```

**Low-frequency approximation** (postural sway 0.1–3 Hz = 0.6–19 rad/s ≪ 1/τs = 100 rad/s):

Both denominator terms → 1:
```
D(s) ≈ 8ms · s · E(s)
d(t) ≈ 8ms · ė(t)
```

The network is a **finite-difference differentiator**. At steady state (`s → 0`), `D(s) = 0`
— the derivative pathway contributes nothing at rest, only during motion.

---

### Stop 7 — Stage 4: Co-Activation |d(t)|

`coact_node` receives:
- Excitatory from `deriv_out_ccw` (d+, always ≥ 0)
- Excitatory from `deriv_out_cw` (d−, always ≥ 0)

Both are bilateral half-waves and are **never simultaneously nonzero**:

| Condition | `d+` | `d−` | `d+ + d−` |
|-----------|------|------|-----------|
| `ė > 0` (CCW sway increasing) | `8ms · ė` | `0` | `8ms · ė` |
| `ė < 0` (CW sway increasing) | `0` | `8ms · |ė|` | `8ms · |ė|` |

```
coact_node output  =  d+  +  d−  =  8ms · |ė(t)|  =  |d(t)|   ✓
```

**The absolute value is free** — no rectifier needed. Bilateral architecture guarantees
that only one half-wave is nonzero at any instant, so their sum is always the unsigned
magnitude.

**Critical distinction:**

| Signal | Source | Meaning |
|--------|--------|---------|
| `d+` or `d−` | `deriv_out_ccw/cw` | Signed, bilateral — directional damping (Kd) |
| `\|d\|` | `coact_node` (shared) | Unsigned, one node — symmetric joint stiffening (Kc) |

`coact_node` feeds **both** `kc_prod_ccw` and `kc_prod_cw` — both muscles stiffen
simultaneously regardless of sway direction, which is the definition of co-activation.

---

### Stop 8 — Stage 5: Bias-Gated Gain

Each pathway uses the same two-neuron circuit (Kp shown):

```
bp_ccw (mod) ─────────────────────────────────┐
                                               ▼
error_ccw  ─────────────────────────── kp_prod_ccw
```

**Step A — mod neuron** (`kp_mod_ccw`): no synaptic inputs, only `Iapp = bp`.
```
U*_mod = [0 + Iapp] / [Gm + 0] = bp / Gm = bp   [mV, since Gm = 1 µS]

Example: bp = 4.26 nA  →  U*_mod = 4.26 mV
```

**Step B — prod neuron** (`kp_prod_ccw`): two excitatory inputs (signal + mod),
same addition subnetwork as Stop 5. Denominator ≈ 1:
```
U*_prod  ≈  1.116 · (U_signal + U_mod)  ≈  U_signal + U_mod  =  U_signal + bp
```

**All four pathways:**

| Pathway | Signal source | Signal `U` | Prod output |
|---------|--------------|-----------|-------------|
| Kp | `error_ccw` | `e` | `e + bp` |
| Kd | `deriv_out_ccw` | `d` | `d + bd` |
| Kc | `coact_node` | `\|d\|` | `\|d\| + bc` |
| Kt | `ib_input` | `Ib` | `Ib + bt` |

**Why PSO is required:** The gains `Kp, Kd, Kc, Kt` in the target control law are **not**
simply the bias values in nA. They are the effective linearized gains of the full nonlinear
cascade — SNS saturation + Hill muscle model + closed-loop pendulum dynamics — measured
via frequency response. The mapping `(bp, bd, bc, bt) → (Kp, Kd, Kc, Kt)` has no
closed form; PSO runs the full simulation and minimizes FRF error against Peterka 2002.

---

### Stop 9 — Stage 7: Motor Sum → Final Equation

`ta_ccw` receives four excitatory inputs (all `gs_exc = 0.115 µS`):

```
kp_prod_ccw ──┐
kd_prod_ccw ──┤
kc_prod_ccw ──┼──→  ta_ccw  →  +a_CCW
kt_prod_ccw ──┘
```

Addition subnetwork with four inputs, denominator ≈ 1:
```
U*_ta  ≈  U_kp_prod + U_kd_prod + U_kc_prod + U_kt_prod
        =  (e + bp)  +  (d + bd)  +  (|d| + bc)  +  (Ib + bt)
        =  (e + d + |d| + Ib)  +  (bp + bd + bc + bt)
```

**Two groups:**
- `(e + d + |d| + Ib)` — the **control law**: each signal arrives at the motor neuron
  with approximately unit gain in the linear SNS regime
- `(bp + bd + bc + bt)` — constant **baseline activation**: sets resting muscle stiffness
  and the operating point around which PSO tunes the effective gains

After PSO has found the optimal bias values, the linearized system approximates:
```
a(t)  =  Kp·e  +  Kd·d  +  Kc·|d|  +  Kt·Ib   ✓
```

---

## §5 — Complete Chain (One-Page Summary)

```
Physical inputs
  θ_body, θ_surface  (radians)
         │
         ▼  Stage 1 — Input Adapter
  I = clip(57.3 · θ,  ±10 nA)       [1° sway = 1 nA = 1 mV in SNS]
  bilateral split → CCW / CW half-waves (never both nonzero)
         │
         ▼  Stage 2 — TC4 Sensory Integration (the new piece over McNeal & Hunt)
  sub_diff:  U_bs − U_ss  =  θ_body − θ_surface
  wg node:   GS_WG → 0.30 × U_bs
  wp node:   GS_WP → 0.70 × (U_bs − U_ss)
  error:     wg + wp  ≈  U_bs − 0.70·U_ss  =  e(t)   [Peterka TC4 error]
         │
         ▼  Stage 3 — Derivative Estimation
  fast (τ=2ms) and slow (τ=10ms) both track e(t)
  deriv_out = fast − slow  →  D(s)/E(s) = 8ms·s / [(1+2ms·s)(1+10ms·s)]
  Low-frequency approx:  d(t) ≈ 8ms · ė(t)
         │
         ▼  Stage 4 — Co-Activation
  coact_node = d+ + d−  =  |d(t)|      [bilateral sum = absolute value, free]
         │
         ▼  Stage 5 — Bias-Gated Gain
  mod neurons:    U_mod_α = b_α / Gm = b_α  [mV]
  prod neurons:   U_kα ≈ signal + b_α

    kp_prod ≈ e   + bp   (4.26 mV)
    kd_prod ≈ d   + bd   (5.01 mV)
    kc_prod ≈ |d| + bc   (2.48 mV)
    kt_prod ≈ Ib  + bt   (5.42 mV)
         │
         ▼  Stage 7 — Motor Sum
  U*_ta = (e + d + |d| + Ib)  +  (bp + bd + bc + bt)

            signal group          baseline offset

  After PSO:  a(t) = Kp·e + Kd·d + Kc·|d| + Kt·Ib   ✓
```

---

## §6 — Key Insights (Easy to Forget After a Month)

**1. Eq. 13 is used at every single neuron — there is no other equation.**
The entire network is one equation chained 7 stages deep. Each synapse conductance
was designed by inverting Eq. 13 at the FSA design point.

**2. The "1" in the denominator is Gm = 1 µS, not dimensionless.**
Units: numerator in nA · mV · µS / mV = nA. Denominator in µS.
Output = nA / µS = mV. Correct.

**3. The 57.3 nA/rad conversion is 180/π — intentional design, not coincidence.**
It maps radians to degrees so 1° = 1 nA = 1 mV in the SNS. The ±10 nA clip is not
an approximation artifact — it enforces the SNS operating range and matches ±10°,
beyond which the small-angle linearization breaks down anyway.

**4. The addition subnetwork gain factor is 1.116, not exactly 1.**
It equals `(gs_e/R) · ΔEe = (0.115/20) · 194 = 1.116` when the denominator ≈ 1.
This factor appears in every addition stage. Since 1.116 ≈ 1, the result is "sum of
inputs," but the exact 11.6% excess is real and accumulates through the cascade. PSO
absorbs this into the effective K values.

**5. Absolute value is free in a bilateral network.**
CCW and CW channels are never simultaneously active. Their sum is always the unsigned
magnitude. No dedicated rectifier neuron is needed for `|d|` — just add d+ and d−.

**6. The sign convention in the diagram title is the classical feedback convention.**
`e = −Wg·BS − Wp·(BS−SS)` = `0 − sensory_estimate`. The network computes `+e`.
The bilateral split closes the negative feedback loop structurally — no sign-inversion
neuron required.

**7. The K gains require PSO — there is no closed-form mapping from bias to K.**
In the linear SNS regime, all four pathways have unit gain (≈1) on their signals.
The effective Kp, Kd, Kc, Kt values measured from the FRF emerge from the nonlinear
interaction of the SNS cascade, the Hill muscle model, and the closed-loop dynamics.
PSO is the only way to find bias values that reproduce human FRF data.

**8. Kc is not directional — it feeds both motor channels equally.**
`coact_node` (Stage 4) connects to BOTH `kc_prod_ccw` and `kc_prod_cw`. This produces
symmetric co-activation: both muscles contract more regardless of which direction the
body is moving, stiffening the joint without producing net torque. This is different
from Kd, which is bilateral (each channel has its own signed d signal).

---

## §7 — Common Pitfalls

| Pitfall | Correction |
|---------|-----------|
| Multiplying `gA · θ°` directly | Convert degrees to radians first: `57.3 · (θ° · π/180) = 57.3 · θ_rad` |
| Treating the "1" in the Eq. 13 denominator as dimensionless | It is `Gm = 1 µS`; carry units |
| Assuming `gs_inh` is the same as `gs_exc` | `gs_inh = 0.558 µS` (≈5× larger), derived from `ΔEe/|ΔEi|` ratio |
| Thinking `d+` feeds directly into Kc | `d+` and `d−` feed `coact_node` (Stage 4) first; `coact_node` feeds Kc |
| Thinking `Kp = bp/Gm` | `Kp` is the linearized FRF gain; `bp` is the bias current; mapping requires PSO |
| Reading the diagram title sign as a network error | It is the classical feedback convention: `e_control = 0 − e_sensor` |

---

## §8 — Quick Reference: Where to Find Things in the Code

| What | File | Key lines |
|------|------|-----------|
| Eq. 13 neuron parameters | `sns_neurons.py` | `Neuron`, `Synapse` classes |
| All synapse conductances | `sns_subnetworks.py` | `gs_exc`, `gs_inh`, `GS_WG`, `GS_WP` |
| Full 38-neuron network construction | `sns_tc4_controller.py` | `build_tc4_network()` |
| Bias parameters | `sns_tc4_controller.py` | `BIAS_*_NA` constants (~line 62) |
| Input adapter | `sns_tc4_controller.py` | `angle_to_current_na()`, `split_to_bilateral_na()` |
| Hill model placeholders (TODO) | `sns_tc4_controller.py` | `compute_tension_proxy_n()` etc. (~line 109) |
| Simulation step | `sns_tc4_controller.py` | `step_controller()` |
| Network diagram | `visualize_tc4_network.py` | `draw_network()` |
| Comprehensive design review | `TC4_REVIEW.md` | — |
