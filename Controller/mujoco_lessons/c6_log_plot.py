import numpy as np
import matplotlib.pyplot as plt
import mujoco

model = mujoco.MjModel.from_xml_path('b6_sensor.xml')
data  = mujoco.MjData(model)

data.qpos[0] = 0.5

n_steps = 2000
time_log = np.zeros(n_steps)
qpos_log = np.zeros(n_steps)

for i in range(n_steps):
    mujoco.mj_step(model, data)
    time_log[i] = data.time
    qpos_log[i] = data.qpos[0]

print(f"Logged {n_steps} steps, final qpos: {qpos_log[-1]:.6f}")

plt.plot(time_log, qpos_log)
plt.xlabel("Time (s)")
plt.ylabel("Hinge angle (rad)")
plt.title("Pendulum swing, no control input")
plt.show()