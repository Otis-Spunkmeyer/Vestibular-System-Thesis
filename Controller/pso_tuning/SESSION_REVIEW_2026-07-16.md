# Session Review — SNS Balance Controller / BPA Rig (2026-07-16)

A comprehensive record of every change made in this working session, why it was made, the
resulting script hierarchy and workflow, and the current state with its known limitations.
Written so the changes can be reviewed after the fact (much of the work was run autonomously).

---

## 1. Executive summary

**Starting point:** the MuJoCo balance model used a Hill `<muscle>` actuator and limit-cycled
at any physiological sensorimotor delay; the fit pipeline against Peterka (2002) Fig 5C could
not be run meaningfully.

**What this session did, in order:**
1. Confirmed the digitized Peterka gain data is a **linear ratio, not dB** (Step 0).
2. Reconciled the pipeline against its four source papers (Peterka 2002; Hilts 2019;
   McNeal & Hunt 2026; Bolen et al. 2026 / BPA), correcting drift and my own earlier mistakes.
3. **Replaced the Hill muscle with a physically-correct BPA actuator model.** This exposed and
   fixed three bugs the Hill muscle had masked:
   - a **fake ~200 N passive force** (a Hill artifact; a deflated BPA is 0 N) that made the
     plant passively stable, voiding Peterka's unstable-pendulum premise;
   - a **backwards control sign** (crossed SNS-output→muscle mapping) that only "worked"
     because the fake passive force stabilized the joint regardless of the active drive;
   - the wrong actuator physics entirely.
4. Result: **the SNS now stably balances a correctly-unstable BPA plant through the full 90 ms
   (and up to ~190 ms) delay** — the session-long limit cycle is gone.
5. Investigated (diagnostic-first) why the controller does not reproduce Peterka's resonant
   overshoot. Localized the limit to **SNS internal saturation** (gain-invariant), not the
   actuator, output stage, co-activation, gains, or delay. Per PI guidance (A. Hunt), an exact
   FRF match is not the goal — the stabilizing system is the deliverable.

**Net outcome:** a working stabilizing controller on a correctly-modeled BPA rig, with the model
reconciled to its source papers and every non-obvious value documented in-code.

---

## 2. Files changed (tracked)

### 2.1 `Controller/balance.py` — the SNS network builder

`generate_sns()` gained several documented, default-preserving parameters and its docstring was
corrected. New signature (defaults reproduce prior behavior unless noted):

```
generate_sns(gains, ctrlr_mode=3, analysis_outputs=False, bs_wt=0.5,
             ib_cm=4000.0, mult_kp_gs=2.2, mult_kd_gs=75.0, mult_kc_gs=75.0,
             t1_cm=0.21, t2_cm=8.0, mult_inter_gs=20.0, mult_inter_erev=-60.1,
             e_lo_mv=-60.0, e_hi_mv=-40.0)
```

| Change | What / why |
|---|---|
| `ib_cm` param (default **4000.0**) | The type-Ib feedback neuron capacitance. Documented that this is **not** a membrane time constant to "biologically correct" — it implements Peterka's low-passed positive torque feedback (Hilts 2019 Eq. 6, `H_T=Kt·ωc/(s+ωc)`, ωc≈0.209 rad/s ⇒ τ≈4 s). Consequence: `kt` is near-invisible in the 0.017–0.744 Hz band **by design** (cf. Hilts Fig 7). |
| `mult_kp_gs`, `mult_kd_gs`, `mult_kc_gs` (2.2 / 75 / 75) | Multiplication-subnetwork pre-amp conductances, exposed as params. Documented as Hilts' deliberate method ("increase synapse conductance to preamplify inputs"), **not** a hack to undo. |
| `t1_cm`, `t2_cm` (0.21 / 8.0) | Differentiator time constants, exposed for diagnostics. |
| `mult_inter_gs`, `mult_inter_erev` (20 / **-60.1**) | Division/multiplication shunting synapse. The −60.1 vs Er=−60 is the FSA design identity (ΔE_s≈0 for division), not a bug. |
| `e_lo_mv`, `e_hi_mv` (−60 / −40) | Synaptic threshold/saturation. Documented that `e_lo==Er` is the FSA identity that produces the pos/neg half-wave split (how the `|d|` co-activation is computed), i.e. the rectification is architecture, not a defect. |
| `analysis_outputs` taps | Added four derivative-circuit taps (`04_t1`, `05_t2_gt_t1`, `06_pos_dErr_dt`, `07_neg_dErr_dt`) so the differentiator is observable. |
| Docstrings | The `bs_wt` "no observable effect" note and the `ib_cm` note were rewritten — the former was stale (bs now has its own input), the latter recorded a mistake I made and reverted (see §6). |

**Not changed:** the network topology, gains semantics, or any default behavior.

### 2.2 `Controller/mujoco_lessons/e6_bilateral.xml` — the plant

| Change | What / why |
|---|---|
| **Removed** `stiffness="69.4"` from `ankle_joint` | No passive stiffness. The rig is intended to be an **unstable** inverted pendulum (mgh=115.6 N·m/rad, J=4.46 kg·m², τ_p=0.196 s). A brief spring experiment was reverted. |
| **Replaced** the two `<muscle>` actuators with `<general>` force actuators on the same tendons | `d.ctrl` is now tendon **tension in Newtons** (gaintype=fixed, gainprm=1), pulling-only (ctrlrange `0 500`). The BPA force-length-pressure law is applied per-step by the control loop (see `bpa.py`). `d.actuator_length` = tendon length; `d.actuator_force` = commanded tension. |
| Removed the `<default class="muscle">` block | No longer needed. |

### 2.3 `Controller/mujoco_lessons/{e6_check,e7_wire_loop,e8_capstone,g2_integrate}.py`

The shared XML actuator changed from 0–1 activation to force-in-Newtons, so all consumers were
updated:
- `g2_integrate.py` (mirrors the pso_tuning loop): imports `bpa`, maps SNS output → pressure →
  BPA force, and applies the **corrected direct** output→muscle sign (see §6).
- `e6_check.py`, `e7_wire_loop.py`, `e8_capstone.py`: teaching demos updated to set representative
  tensions in N (and Hill→BPA label fixes). They run without error; note these are crude scaffolds
  that used `qpos[0]` (platform) loosely and were partly relying on the old passive force.

---

## 3. Files changed / added (pso_tuning, untracked directory)

### 3.1 `pso_tuning/bpa.py` — **NEW** — BPA isometric force model

Implements the Festo φ10 mm BPA force law from **Bolen et al. 2026**:
```
F620(lrest)     = 303.5·arctan(19.03·(lrest−0.0075))                      # Eq.5, N at 620 kPa
Fstar(e*,P*)    = max(0, c0·(exp(−c1·e*)−1) + P*·exp(−c2·e*²))            # Eq.8 (c0,c1,c2=0.5682,4.254,0.5597)
bpa_force(l,lrest,P) = Fstar(e*,P/620)·F620(lrest)                        # Eq.9; e=(lrest−l)/lrest, e*=e/EPS620
```
Constants tied to the **physical build** (documented as such, must match hardware):
- `LREST = 0.267` m — BPA resting/max length (a mounting choice; ~upright tendon length for good
  engagement; the joint-limit value 0.2832 left the actuator too contracted at upright).
- `EPS620 = 0.15` — max contraction (varies per muscle; measure on real actuators).
- `P620 = 620.0` kPa. Static/isometric model (hysteresis omitted, endorsed for force-map control).

Key physics vs a Hill muscle: force **only in tension, only when pressurized** (deflated = 0, no
passive spring); **max at resting length**, monotonically decreasing as it contracts.

### 3.2 `pso_tuning/model_frf.py` — MODIFIED — the closed-loop FRF

Central pipeline file. Session changes (all with defaults reproducing the working baseline):

| Area | Change |
|---|---|
| Actuator | Imports `bpa`; replaced `_v2a` (SNS mV→0–1 activation) with `_v2p` (SNS mV→pressure kPa, ceiling `P_MAX=620`). Loop now computes BPA tension from tendon length + commanded pressure. **Direct** output→muscle mapping (sign fix, §6). |
| PRTS amplitude | `PRTS_PP_DEG = 2.0` — drive is 2° peak-to-peak (matches Peterka's `A_PRTS`); the prior ~4.25° was drift. `_prts_periods` rescales exactly to pp. |
| Sensory reweighting | `bs_wt_for(pp_deg)` — Peterka's graviceptive weight w_g interpolated in log-amplitude between his endpoints (0.5°→0.30, 8°→0.76); ~0.53 at 2°. `model_frf` defaults `bs_wt=None → bs_wt_for(PRTS_PP_DEG)` so it tracks amplitude. |
| Delay | `tau_d_for(pp_deg)` — Peterka's amplitude-dependent effective delay (0.5°→0.191 s, 8°→0.105 s); ~0.148 s at 2°. `SENSORY_DELAY_S` and `IB_DELAY_S` both derive from it. Dual-offset ring-buffer applies the delay to the sensory inputs (long-loop on bf/bs, Ib on tension). |
| Limit-cycle guard | `MIN_DRIVEN_FRAC = 0.5` — rejects runs where <50% of body-sway energy sits at stimulus frequencies (a bounded limit cycle is finite and would otherwise pass the isfinite check and fake a resonant peak). |
| Provenance | `BASELINE_GAINS = (4.26,5.01,2.48,5.42)` documented as McNeal & Hunt Table IV ("Split, k_t on"), not "hand-tuned"; b_d≠b_c is the paper's finding. |

### 3.3 Unchanged pso_tuning files (pre-existing, still current)
`step0_check_units.py` (units check — answered: linear), `target.py` (loads the 3 digitized CSVs,
`interp_target`), `cost.py` (`frf_cost`), `pso.py` (particle-swarm over the 4 gains), `run_fit.py`
(overlay best fit). `README.md` updated earlier in the session for the units result.

---

## 4. Script hierarchy

```
Controller/
├── balance.py .................. SNS network builder (generate_sns) -- the neural controller
├── prts_generator.py ........... PRTS ternary-sequence stimulus (generate_prts)
├── complementary_filter.py ..... graviceptive body-in-space estimate (ComplementaryFilter)
├── mujoco_lessons/
│   └── e6_bilateral.xml ......... the PLANT: inverted pendulum + tendons + BPA force actuators
│   └── g2_integrate.py .......... interactive viewer integration (mirrors model_frf; BPA-updated)
│   └── e6_check / e7 / e8 ....... teaching demos (BPA-updated)
└── pso_tuning/                    ---- the fit pipeline ----
    ├── bpa.py .................. [NEW] BPA force law (Bolen). Imported by model_frf + g2
    ├── target.py ............... digitized Peterka Fig 5C -> interp_target(freqs)
    ├── model_frf.py ............ CORE: runs the closed loop, returns (freqs, gain, phase)
    │                              imports balance, bpa, prts_generator, complementary_filter
    ├── cost.py ................. frf_cost(gains) = coherence-weighted log-gain+phase error vs target
    ├── pso.py .................. particle-swarm over (kp,kd,kc,kt); seeds particle 0 at BASELINE_GAINS
    ├── run_fit.py .............. loads best_gains.npy, overlays model vs target
    ├── step0_check_units.py .... one-off units check (linear vs dB)
    └── README.md / SESSION_REVIEW_2026-07-16.md (this doc)

Data:  Controller/digitized data/{Peterka5c_Gain,Peterka5c_Phase,Coherence}vsFreq.csv
```

**Dependency / data flow:**
```
prts_generator ─┐
complementary_filter ─┤
balance.generate_sns ─┼─> model_frf.model_frf(gains) ──> (freqs, gain, phase)
bpa.bpa_force ────────┘            │
e6_bilateral.xml (plant) ─────────┘
                                   │
target.interp_target ── cost.frf_cost(gains) ── pso.pso() ──> best_gains.npy ── run_fit.py ── overlay PNG
```

---

## 5. Workflow (how to run)

Run from `Controller/pso_tuning/` using the repo venv (`../../venv/Scripts/python.exe`).

1. **Units (already answered):** `python step0_check_units.py` → confirms the gain CSV is linear.
2. **Sanity checks:** `python model_frf.py` (baseline FRF), `python cost.py` (baseline cost).
3. **Optimize:** `python pso.py` → writes `best_gains.npy`. Long (~1 hr); particle 0 seeded at the
   published gains so the result never returns worse than baseline. Guard rejects limit cycles.
4. **Inspect:** `python run_fit.py` → console costs + `frf_fit_result.png`.

Key knobs (all documented in-file): `model_frf.PRTS_PP_DEG` (stimulus amplitude → drives
`bs_wt_for` and `tau_d_for`), `bpa.LREST` (**hardware** resting length), `model_frf.P_MAX`
(SNS→pressure ceiling), `pso.BOUNDS` (search box).

---

## 6. Mistakes made and corrected (full transparency)

These were made and reverted **within** the session; the code reflects the corrected state.

| Mistake | Correction |
|---|---|
| Set `ib_cm` 4000→5.0 arguing it was "800× outside biology" that disabled `kt`. | **Reverted to 4000.** It implements Peterka's low-passed torque feedback (Hilts Eq. 6). `kt` being near-invisible in-band is by design. The "disabled" evidence was an artifact of one operating point. |
| Invented `IB_DELAY_S = 0.035` (Ib "fast spinal reflex"). | **Set to 0.090→ now `tau_d_for`.** Fig 2 shows tau_d on the tension path too; 0.035 was in no source. |
| Called the pos/neg rectifier a "32° phase deficit / bug." | It is the **architecture** — how `|d|` co-activation is computed (McNeal & Hunt Eq. 1). |
| Wanted to "restore" `mult_kd` 75→54. | 75 is Hilts' documented pre-amp method; left as-is. |
| Sized an ankle spring (0.6·mgh) to add passive stiffness. | Removed once the BPA model showed the real rig has **no** passive stiffness. |
| **Crossed** SNS-output→muscle mapping (inherited from g2). | **Direct** mapping. Measured static K_P: −265 (crossed, destabilizing) vs +139 (direct, restoring). The Hill passive force had masked this sign error the whole time. |

---

## 7. Key quantitative findings

- Plant (BPA, no passive): mgh=115.6 N·m/rad, J=4.46, **τ_p=0.196 s** (human ≈0.303 s — a
  different plant, which is why an exact FRF match is unrealistic; confirmed by PI).
- Ideal linear PD on the BPA plant reaches cost **~0.15** vs Peterka; through the BPA (0.16) it is
  the same as pure torque (0.16) → **the actuator is not the limit**.
- The SNS reaches **0.2971**, and that cost is **invariant** to gains (kp 2–12, kd, kc),
  co-activation (kc=kd), output stage (force-realizing vs pressure), delay (90–190 ms), and input
  scaling (DEG_MAX 10–80). Root cause: the SNS's PD-output neuron is **saturated ~97% of the time**
  → it acts as a fixed nonlinear element → rides the platform (gain ≈ 1.0).
- Diagnostic scripts (scratchpad, reproducible, not committed): `ideal_pd.py` (ideal-PD vs target,
  3 configs), `sns_force_realize.py` (output-stage test), `coact_ceiling.py` (co-activation test),
  `satcheck.py` (saturation instrumentation), `unsaturate.py` (DEG_MAX relief). Listed here for
  reproducibility; they can be regenerated from this document's descriptions.

---

## 8. Current state & known limitations

**Working:** the SNS stably balances the unstable BPA plant through the physiological delay; the
graviceptive channel stabilizes the body in space (gain < 1) at short delays; the pipeline runs
end-to-end; all defaults verified intact (do-no-harm check passed).

**Limitations (accepted, per PI guidance — see memory `supervisor-frf-guidance`):**
1. Does **not** reproduce Peterka's resonant overshoot (gain ~3.0 @ 0.7 Hz). Root cause is SNS
   saturation, not a tunable parameter. Another lab member (Stu) also could not reproduce it.
2. The rig inertia/dynamics differ from a human (τ_p 0.196 vs 0.303 s), which alone precludes an
   exact match. It is a simplified model of a complex system.
3. `bpa.LREST` and `EPS620` are nominal — must be set/measured on the real actuators.
4. The `tau_d_for` endpoints (0.191/0.105 s) are approximate — confirm against Peterka's table.

**Recommended next step (PI):** move to hardware, then compare hardware to this simulation — a
now well-characterized comparison, since the sim's behavior and its limits are fully mapped.

**Git note:** the `pso_tuning/` directory is untracked; `balance.py`, `e6_bilateral.xml`, and the
lesson scripts are modified but uncommitted. Nothing has been staged or committed this session.
