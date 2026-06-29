import mujoco

model = mujoco.MjModel.from_xml_path('e6_bilateral.xml')
data = mujoco.MjData(model)

# Initialize with small displacement
data.qpos[0] = 0.05

for i in range(2000):
    body_angle = data.qpos[0]

    # Placehodler stand-in for the ral SNS controller output
    # balance.py would compute these from its network's neuron voltages instead.
    if body_angle > 0:
        ccw_activation, cw_activation = 0.0, 0.3
    else:
        ccw_activation, cw_activation = 0.3, 0.0

    data.ctrl[0] = ccw_activation
    data.ctrl[1] = cw_activation

    mujoco.mj_step(model, data)

    if i % 400 == 399:
        print(f"step {i:4d} - angle: {body_angle:>9.5f}  ctrl(ccw,cw): ({data.ctrl[0]:.2f}, {data.ctrl[1]:.2f})")