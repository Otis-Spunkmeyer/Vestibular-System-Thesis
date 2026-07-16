import numpy as np
import matplotlib.pyplot as plt
import mujoco

model = mujoco.MjModel.from_xml_path("e6_bilateral.xml")
data = mujoco.MjData(model)
data.qpos[0] = 0.05

n_steps = 4000
time_log = np.zeros(n_steps)
angle_log = np.zeros(n_steps)

for i in range(n_steps):
    body_angle = data.qpos[0]

    # ctrl is BPA tension in N. Flx (ctrl[0]) pulls the ankle negative, so correct a
    # positive lean by pressurising Flx.
    if body_angle > 0:
        data.ctrl[0], data.ctrl[1] = 200.0, 0.0
    else:
        data.ctrl[0], data.ctrl[1] = 0.0, 200.0

    mujoco.mj_step(model, data)

    time_log[i] = data.time
    angle_log[i] = data.qpos[0]

print(f"Final angle after {n_steps} steps: {angle_log[-1]:.6f} rad ({np.degrees(angle_log[-1]):.3f} deg)")

plt.plot(time_log, np.degrees(angle_log))
plt.xlabel("Time (s)")
plt.ylabel("Ankle angle (deg)")
plt.title("Inverted-pendulum ankle stabilized by bilateral BPA actuators")
plt.show()
