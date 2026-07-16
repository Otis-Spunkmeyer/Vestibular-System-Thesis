import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../pso_tuning'))

import numpy as np
import mujoco
import mujoco.viewer
import balance
import bpa                        # BPA force model (Bolen et al. 2026)

from prts_generator import generate_prts
from complementary_filter import ComplementaryFilter

# __ MujoCo __________________________________
model = mujoco.MjModel.from_xml_path("e6_bilateral.xml")
data = mujoco.MjData(model)

dt_s = model.opt.timestep       # 0.0001 s
dt_ms = dt_s * 1000             # 0.1 ms (sns_toolbox uses ms)

# __ sns network_______________________________
gains = (4.26, 5.01, 2.48, 5.42)    # (kp, kd, kc, kt) -use PSO to tune to actual
net = balance.generate_sns(gains, ctrlr_mode=3, bs_wt=0.3)
sns = net.compile(backend='numpy', dt=dt_ms, debug=False)

n_inputs = net.get_num_inputs()
n_outputs = net.get_num_outputs()

print(f"Mujoco Timestep:    {dt_s*1000:.2f} ms")
print(f"SNS dt:             {dt_ms:.2f} ms")
print(f"SNS inputs:         {n_inputs}")
print(f"SNS outputs:        {n_outputs}")
print("Skeleton OK")

# ── PRTS surface perturbation ────────────────────────────────────────────────
prts_t, prts_pos = generate_prts(v_amp=np.deg2rad(1.0), dt=dt_s)
prts_pos = prts_pos - prts_pos.mean()   # center around 0 so platform tilts both ways
prts_vel = np.zeros_like(prts_pos)
prts_vel[1:] = np.diff(prts_pos) / dt_s   # numerical velocity for kinematic drive

print(f"PRTS duration:    {prts_t[-1]:.1f} s  ({len(prts_pos)} steps)")

# __Mode selection_________________________________
print("\nSelect simulation mode:")
print("  1 — PRTS test     (bs mirrors bf; proprioceptive only)")
print("  2 — Sim proxy     (bs = MuJoCo world-frame body angle, no filter)")
print("  3 — Sim + filter  (bs = synthetic IMU → complementary filter)")
print("  4 — Real IMU      (bs = live IMU reads → complementary filter)")
mode = int(input("Mode [1/2/3/4]: "))
if mode not in (1, 2, 3, 4):
    raise ValueError(f"Invalid mode: {mode}")

cf = ComplementaryFilter() if mode in (3, 4) else None
print(f"Running in mode {mode}")

#__ Input adapters_________________________________
IB_SCALE = 10/ 1500.0           # nA per Newton (10nA max / 1500N Fmax)
IB_CLIP  = 10.0                 # nA saturation limit
DEG_MAX  = 10.0                 # operating limit in degrees
R        = 20.0                 # SNS operating range in mV (= nA here)

def angle_to_na(angle_rad):
    deg = np.degrees(angle_rad)
    return float(np.clip(deg /DEG_MAX * (R / 2), -R / 2, R / 2))

def tension_to_ib_na(force_n):
    return float(np.clip(abs(force_n) * IB_SCALE, 0.0, IB_CLIP))

# pre-allocate inputs with input vector
inputs = np.zeros(n_inputs)

print("Adaptors OK - Test:")
inputs[0] = angle_to_na(np.radians(5.0))
inputs[1] = angle_to_na(np.radians(5.0))   # bs same as bf for this static test
inputs[2] = tension_to_ib_na(300.0)
inputs[3] = tension_to_ib_na(0.0)

print(f"5 deg → {inputs[0]:.3f} nA (expect  5.000)  [bf ankle]")
print(f"5 deg → {inputs[1]:.3f} nA (expect  5.000)  [bs body]")
print(f"300 N → {inputs[2]:.3f} nA (expect  2.000)  [Flx_Ib]")


#__ Output adapters_________________________________
# The rig uses BPAs, not Hill muscles, so the SNS output commands PRESSURE (kPa),
# and BPA tension (N) is computed from tendon length + pressure via bpa.bpa_force.
# The actuators in e6_bilateral.xml are plain force actuators: data.ctrl is tension in N.
Er    = -60.0                    # Neuron resting potential
P_MAX = 620.0                    # kPa: SNS output [Er, Er+R] -> [0, P_MAX] pressure

def v_to_pressure(v_mv):
    return float(np.clip((v_mv - Er) / R, 0.0, 1.0)) * P_MAX

print(f"Output adapter OK — test:")
print(f"  -60 mV to {v_to_pressure(-60.0):6.1f} kPa (expect   0.0, deflated)")
print(f"  -50 mV to {v_to_pressure(-50.0):6.1f} kPa (expect 310.0, half pressure)")
print(f"  -40 mV to {v_to_pressure(-40.0):6.1f} kPa (expect 620.0, full pressure)")

# ── Logging ───────────────────────────────────────────────────────────────────
n_steps      = len(prts_pos)
time_log     = np.zeros(n_steps)
surface_log  = np.zeros(n_steps)   # surface sway (deg)
ankle_log    = np.zeros(n_steps)   # ankle angle rel. to surface (deg)
body_abs_log = np.zeros(n_steps)   # absolute body angle in world (deg)
ctrl_ccw_log = np.zeros(n_steps)   # Flx_bpa tension (N)  (ctrl[0])
ctrl_cw_log  = np.zeros(n_steps)   # Ext_bpa tension (N)  (ctrl[1])
sns_ccw_log  = np.zeros(n_steps)   # SNS output[0] voltage (mV) -> Flx pressure
sns_cw_log   = np.zeros(n_steps)   # SNS output[1] voltage (mV) -> Ext pressure

#__ Simulation ______________________________________
data.qpos[0] = 0.0   # platform starts flat
data.qpos[8] = 0.0   # ankle starts upright relative to platform

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(len(prts_pos)):
        if not viewer.is_running():
            break

        # Drive surface joint kinematically from PRTS
        data.qpos[0] = prts_pos[step]
        data.qvel[0] = prts_vel[step]

        # 1. Proprioceptive input (bf): ankle angle relative to platform
        inputs[0] = angle_to_na(data.qpos[8])

        # 2. Graviceptive input (bs): mode-dependent body-in-space estimate
        if mode == 1:
            inputs[1] = inputs[0]                                  # mirrors bf; bs_wt has no effect
        elif mode == 2:
            inputs[1] = angle_to_na(data.qpos[0] + data.qpos[8])  # world-frame body angle, no filter
        elif mode == 3:
            body_angle = data.qpos[0] + data.qpos[8]              # synthetic IMU from MuJoCo truth
            body_vel   = data.qvel[0] + data.qvel[7]
            acc_sim    = [np.sin(body_angle) * 9.81, np.cos(body_angle) * 9.81]
            inputs[1]  = angle_to_na(cf.update(body_vel, body_vel, acc_sim, acc_sim, dt_s))
        elif mode == 4:
            gyro1, acc1 = imu1.read()                              # TODO: replace with real IMU calls
            gyro2, acc2 = imu2.read()
            inputs[1]   = angle_to_na(cf.update(gyro1, gyro2, acc1, acc2, dt_s))

        # 3. Ib tension feedback
        inputs[2] = tension_to_ib_na(data.actuator_force[0])  # Flx_Ib  (Flx_muscle = ctrl[0])
        inputs[3] = tension_to_ib_na(data.actuator_force[1])  # Ext_Ib  (Ext_muscle = ctrl[1])


        # 4. Step SNS
        outputs = sns(inputs)

        # 5. Apply BPA tension (N): SNS output -> pressure -> force from tendon length.
        # DIRECT mapping (out[0]->Flx, out[1]->Ext). The crossed mapping this script used
        # to have (out[1]->Flx) is destabilising once the Hill muscle's ~200 N passive
        # force is gone; see pso_tuning/model_frf.py for the measurement.
        data.ctrl[0] = bpa.bpa_force(data.actuator_length[0], bpa.LREST, v_to_pressure(outputs[0]))
        data.ctrl[1] = bpa.bpa_force(data.actuator_length[1], bpa.LREST, v_to_pressure(outputs[1]))

        # 6. Step physics
        mujoco.mj_step(model, data)
        time_log[step]     = data.time
        surface_log[step]  = np.degrees(data.qpos[0])
        ankle_log[step]    = np.degrees(data.qpos[8])
        body_abs_log[step] = surface_log[step] + ankle_log[step]
        ctrl_ccw_log[step] = data.ctrl[0]
        ctrl_cw_log[step]  = data.ctrl[1]
        sns_ccw_log[step]  = outputs[0]
        sns_cw_log[step]   = outputs[1]

        # 7. Update viewer
        viewer.sync()

# ── Plotting ──────────────────────────────────────────────────────────────────
ds = 500   # downsample: every 500 steps = 50 ms resolution (~1210 points total)
t  = time_log[::ds]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

axes[0].scatter(t, surface_log[::ds],  label='Surface sway (PRTS)', color='gray',   s=1.5)
axes[0].scatter(t, body_abs_log[::ds], label='Body angle (world)',  color='blue',   s=1.5)
axes[0].set_ylabel('Angle (deg)')
axes[0].set_title('Surface perturbation vs body response')
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(t, ankle_log[::ds], label='Ankle angle (rel. to surface)', color='green', s=1.5)
axes[1].set_ylabel('Angle (deg)')
axes[1].set_title('Proprioceptive signal fed to SNS')
axes[1].legend()
axes[1].grid(True)

axes[2].scatter(t, ctrl_ccw_log[::ds], label='Flx activation', color='orange', s=1.5)
axes[2].scatter(t, ctrl_cw_log[::ds],  label='Ext activation', color='red',    s=1.5)
axes[2].set_ylabel('Activation [0, 1]')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('Hill muscle activations')
axes[2].legend()
axes[2].grid(True)

plt.show()
