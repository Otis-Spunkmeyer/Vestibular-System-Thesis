import mujoco

model = mujoco.MjModel.from_xml_path('d1_ankle.xml')
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)
print(f"Effective inertia about hinge: {data.qM[0]:.3f} (project value: 63.0)")

data.qpos[0] = 0.01
for i in range(5):
    mujoco.mj_step(model, data)
    print(f"Step {i} - qpos {data.qpos[0]:>10.6f}")
