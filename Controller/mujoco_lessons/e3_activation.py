import mujoco
model = mujoco.MjModel.from_xml_path('e2_muscle.xml')
data = mujoco.MjData(model)

print(f"Number of activation states (na): {model.na}")

data.ctrl[0] = 1.0

for i in range(300):
    mujoco.mj_step(model, data)
    if i % 50 == 49:
        print(f"step {i:3d} - ctrl: {data.ctrl[0]:.3f} act: {data.act[0]:.6f}")