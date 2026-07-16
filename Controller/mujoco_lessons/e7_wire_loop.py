import mujoco

model = mujoco.MjModel.from_xml_path('e6_bilateral.xml')
data = mujoco.MjData(model)

# Initialize with small displacement
data.qpos[0] = 0.05

for i in range(2000):
    body_angle = data.qpos[0]

    # Placeholder stand-in for the real SNS controller output. balance.py would compute
    # these from its network's neuron voltages instead. ctrl is now BPA TENSION in N
    # (the actuators are force actuators). Flx (ctrl[0]) pulls the ankle negative, so a
    # positive lean is corrected by pressurising Flx.
    if body_angle > 0:
        flx_force, ext_force = 200.0, 0.0
    else:
        flx_force, ext_force = 0.0, 200.0

    data.ctrl[0] = flx_force
    data.ctrl[1] = ext_force

    mujoco.mj_step(model, data)

    if i % 400 == 399:
        print(f"step {i:4d} - angle: {body_angle:>9.5f}  ctrl(ccw,cw): ({data.ctrl[0]:.2f}, {data.ctrl[1]:.2f})")