import mujoco

model = mujoco.MjModel.from_xml_path('b6_sensor.xml')
data = mujoco.MjData(model)

# initial displacement
data.qpos[0] = 0.5

mujoco.mj_step(model,data)

print("qpos shape:", data.qpos.shape, "value:", data.qpos[0])
print("qvel shape:", data.qvel.shape, "value:", data.qvel[0])
print("sensordata shape:", data.sensordata.shape)
print("jointpos sensor reading:", data.sensordata[0])
print("jointvel sensor reading:", data.sensordata[1])

mujoco.mj_forward(model, data)
print("sensordata after mj_forward:", data.sensordata)
