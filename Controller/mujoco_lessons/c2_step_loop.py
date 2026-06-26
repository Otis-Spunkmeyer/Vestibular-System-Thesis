import mujoco

model = mujoco.MjModel.from_xml_path('b6_sensor.xml')
data = mujoco.MjData(model)

# displace the hinge angle (radians) away from equilibrium
data.qpos[0] = 0.5

print("Timestep (seconds):", model.opt.timestep)

for i in range(20):
    mujoco.mj_step(model, data)
    print("step", i, "- time:", data.time, "qpos:", data.qpos, "qvel:", data.qvel)