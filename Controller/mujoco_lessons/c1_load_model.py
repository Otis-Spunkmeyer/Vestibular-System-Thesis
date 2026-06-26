import mujoco

model = mujoco.MjModel.from_xml_path('b6_sensor.xml')
data = mujoco.MjData(model)

print("Number of geoms:", model.ngeom)
print("Number of joints:", model.njnt)
print("Number of degrees of freedom:", model.nv)
print("Initial joint position (qpos):", data.qpos)
print("Initial joint velocity (qvel):", data.qvel)