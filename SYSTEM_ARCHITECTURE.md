# System Architecture

## Project
Vestibular balance thesis. SNS (synthetic nervous system) controller replicating Peterka 2002 test conditions using McNeal & Hunt 2026 split-derivative architecture.

## Module Routing Table
| Task | Read first |
|------|-----------|
| Neuron/synapse primitives | `Controller/sns_neurons.py` |
| Subnetwork building blocks | `Controller/sns_subnetworks.py` |
| TC4 full controller | `Controller/sns_tc4_controller.py` |
| Simulation runner | `Controller/simulate.py` |
| PRTS stimulus | `Controller/prts_generator.py` |
| Network visualization | `Controller/visualize_tc4_network.py` |
| Transfer function analysis | `TF_analysis.py` |

## File Map
```
/
├── SYSTEM_ARCHITECTURE.md          ← this file
├── TF_analysis.py                  ← open-loop TF analysis, Peterka Eq. 15
├── tf_analysis_gain_only.png
├── tf_analysis_gain_phase.png
└── Controller/
    ├── sns_neurons.py              ← Neuron, Synapse classes (Hilts/Szczecinski)
    ├── sns_subnetworks.py          ← add, sub, mul, div, deriv, integral, transmission subnetworks
    ├── sns_tc4_controller.py       ← TC4 full SNS (McNeal&Hunt base + TC4 sensory upstream)
    ├── peterka_controller.py       ← older TC4 attempt (multiply-based, incomplete)
    ├── simulate.py                 ← generic time-step simulator
    ├── prts_generator.py           ← PRTS sequence from Peterka 2002
    ├── test_addition.py            ← unit test for addition subnetwork
    └── visualize_tc4_network.py    ← network diagram (neurons + synapses)
```

## SNS Primitives (sns_neurons.py)
- `Neuron(Cm_nF, Gm_uS, Er_mV)` — conductance-based, non-spiking
- `Synapse(gs_max_uS, Es_mV, Elo_mV, Ehi_mV)` — piecewise-linear activation
- Operating range R = Ehi - Elo = 20 mV; Er = -60 mV; Ehi = -40 mV

## Subnetwork Constants (sns_subnetworks.py)
| Constant | Value | Meaning |
|----------|-------|---------|
| Er | -60 mV | resting potential |
| R | 20 mV | operating range |
| Ehi | -40 mV | upper voltage bound |
| Elo | -60 mV | lower voltage bound |
| gs_exc | 0.115 µS | default excitatory conductance |
| Es_exc | 134 mV | excitatory reversal potential |
| gs_inh | 0.558 µS | default inhibitory conductance |
| Es_inh | -100 mV | inhibitory reversal potential |

## McNeal & Hunt 2026 Controller (sns_tc4_controller.py)
Architecture: 30-neuron base + 5 TC4 upstream neurons = 35 neurons total.

### Motor drive equation (Eq. 1)
`a(t) = wp·e(t) + wd·d(t) - wc·|d(t)| + wt·yIb(t)`
- a(t) = [a_CCW, a_CW] bilateral motor drive
- e(t) = error signal (TC4: dual-channel, see below)
- d(t) = derivative estimate (vfast - vslow)
- |d(t)| = co-activation (rectified derivative)
- yIb(t) = Type-Ib delayed tension feedback

### TC4 error signal (extends McNeal & Hunt)
`e = -Wg·BS - Wp·(BS - SS)` where Wp=0.70, Wg=0.30
Upstream neurons added over base architecture:
1. `bs_pre` — body sway input
2. `ss_pre` — surface sway input
3. `sub_diff` — BS-SS (subtraction output)
4. `wg_node` — Wg×BS (transmission, gain=0.30)
5. `wp_node` — Wp×(BS-SS) (transmission, gain=0.70)
Output feeds into base error neuron (replaces θSNS-θRef).

### Adapter mappings (physical ↔ neural units)
| Signal | Formula | Constants |
|--------|---------|-----------|
| Ankle angle input | I = clip(gA·θA, ±10 nA) | gA = 180/π nA/rad |
| Type-Ib tension | I = clip(gIb·T, 0, 10 nA) | gIb = -10/Fmax, Fmax=1500 N |
| Motor activation out | A_MJ = clip((A_SNS+60)/20, 0, 1) | — |

### Bias parameters (starting values, Table IV Split+kt)
| Param | Value | Controls |
|-------|-------|---------|
| bp | 4.26 nA | proportional gain |
| bd | 5.01 nA | directional derivative |
| bc | 2.48 nA | co-activation derivative |
| bt | 5.42 nA | Type-Ib tension feedback |

### Simulation parameters
| Param | Value |
|-------|-------|
| τd (sensory delay) | 90 ms |
| Δt | 0.2 ms |
| PRTS amplitude | 2° (McNeal) / 1° (Peterka TC4) |

## Physical Plant Parameters (Table I)
J=63 kg·m², m=77.8 kg, h=0.9 m, g=9.81 m/s², Fmax=1500 N, Lmoment=4.9 cm

## Key Design Decisions
- peterka_controller.py: uses multiplication networks for gains — superseded by sns_tc4_controller.py
- sns_tc4_controller.py: uses transmission synapses for fixed gains (lighter, analytically designed)
- Type-Ib path delayed by τt (same buffer as sensory delay unless specified otherwise)
- Hill-type muscle model lives in plant (MuJoCo in McNeal & Hunt); Python version uses simplified torque proxy
