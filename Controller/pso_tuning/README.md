# PSO gain tuning — SNS balance controller vs Peterka Fig 5C

Tunes the four SNS gains `(kp, kd, kc, kt)` so the closed-loop **body-sway /
support-surface frequency response** matches digitized human eyes-closed data
(Peterka 2002, Fig 5C). It drives the rescaled MuJoCo pendulum with the PRTS
perturbation, computes the model's frequency response, and uses Particle Swarm
Optimization to minimize the mismatch against the human curves.

## Prerequisites
- Use the repo virtualenv (`../../venv`) — it has `mujoco`, `sns-toolbox`,
  `numpy`, `matplotlib`.
- **Run every script from inside this folder** (`Controller/pso_tuning/`) so the
  sibling imports (`balance`, `prts_generator`, `complementary_filter`) and the
  `../digitized data/` and `../mujoco_lessons/e6_bilateral.xml` paths resolve.

## Files
| File | Role |
|------|------|
| `step0_check_units.py` | One-off, **already answered**: plots the gain CSV as linear vs dB. Result: linear. |
| `target.py` | Loads the 3 digitized CSVs; `interp_target(freqs)` resamples gain/phase/coherence onto any frequencies. |
| `model_frf.py` | Runs the headless mode-3 closed loop and returns the model FRF; `analysis_freqs()` gives the in-band PRTS harmonics. |
| `cost.py` | `frf_cost(gains)` → one coherence-weighted scalar (the PSO objective). |
| `pso.py` | Hand-rolled parallel PSO over the 4 gains; saves `best_gains.npy`. |
| `run_fit.py` | Overlays the best-fit model on the target; writes `frf_fit_result.png`. |

## How to run (in order)
```bash
cd Controller/pso_tuning

# 0. Already done: units confirmed LINEAR, target.py has GAIN_IS_DB = False.
#    Kept for reproducibility - re-run only to re-check the units.
python step0_check_units.py

# 1. (optional sanity checks)
python model_frf.py     # prints the baseline model FRF (expect ~flat gain ≈ 0.9)
python cost.py          # prints the in-band grid and the baseline cost

# 2. Run the optimization (edit swarm size / iters / bounds inside pso.py)
python pso.py           # -> best_gains.npy

# 3. Inspect the fit
python run_fit.py       # -> console costs + frf_fit_result.png
```

## Knobs
- `target.py: GAIN_IS_DB` — confirmed linear in Step 0; leave `False`.
- `pso.py: BOUNDS, n, iters, w, c1, c2, procs` — search box, swarm size, iterations,
  inertia/cognitive/social weights, and worker-process count.
- `model_frf.py: n_periods` — 2 (default, discards a startup transient) or 1 (≈2× faster, slightly noisier).
- `model_frf.py: PRTS_PP_DEG` — support-surface drive amplitude, peak-to-peak.
  **Leave at 2.0**: it must match the stimulus that produced the target data.
  Not a free parameter — the model is nonlinear, so amplitude shifts the model
  gain by ~5% (17% worst case). Changing it invalidates the comparison to Fig 5C.
- `model_frf.model_frf(..., bs_wt=, cf_alpha=)` — fixed sensory split and filter constant (not searched).

## Runtime
Each `frf_cost` call is a full simulation (~1.2M physics steps at 0.1 ms for 2
PRTS periods): **~56 s** standalone. Particles are evaluated in parallel
(`multiprocessing`), and `pool.map` waits on the slowest worker each round, so a
16-particle iteration costs `ceil(16/procs)` rounds.

Measured on a 14-core / 20-thread i5-13600KF (hybrid: 6 P-cores + 8 slower
E-cores — stragglers scheduled onto E-cores set the round time, so scaling is
well short of linear):

| `procs` | full 16×25 run | machine usable meanwhile? |
|---------|----------------|---------------------------|
| 16 (`procs=None` caps at `n`) | ~50 min | sluggish; light editing only |
| 8 | ~68 min | yes, 12 threads left free |

Dropping workers is *not* free — 16 still wins by ~19 min. To get both, pair
`procs=8` with `n_periods=1` (~34 min, noisier FRF) while iterating on the setup.
Note these are single-round extrapolations; real runs vary as divergent particles
bail early via the `fail` path.

## Known limitations
- **Fit band ≈ 0.017–0.74 Hz.** The PRTS only injects power up to ~0.74 Hz, so
  the human roll-off above ~0.85 Hz cannot be fit — those points have no model data.
- **Baseline model FRF is ~flat** (gain 0.79–1.01, phase 0° to −17° at the
  baseline gains `(4.26, 5.01, 2.48, 5.42)`): the controller rides the platform
  rather than stabilizing in space. A good fit must produce the resonant gain
  peak; whether the four gains alone can reach it is exactly what this
  experiment tests. **Baseline cost = 0.2987** (43% gain / 57% phase) — the
  reference the PSO must beat.
- **The target's defining feature sits at the edge of the band.** The Fig 5C
  gain peak is at 0.69 Hz, just inside the ~0.744 Hz limit, and all the baseline
  error concentrates at 0.61–0.74 Hz. There the model is not merely low-gain but
  roughly *antiphase*: at 0.678 Hz it reads gain 0.95 / phase −4° against a
  target of gain 2.99 / phase −152°.
