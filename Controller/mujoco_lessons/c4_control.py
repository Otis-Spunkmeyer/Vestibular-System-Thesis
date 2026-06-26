import mujoco

model = mujoco.MjModel.from_xml_path('b6_sensor.xml')
data = mujoco.MjData(model)

data.ctrl[0] = 0.5

for i in range(10):
    mujoco.mj_step(model, data)
    print(f"Step {i} - qpos {data.qpos[0]:>10.6f} qvel: {data.qvel[0]:>10.6f}")

