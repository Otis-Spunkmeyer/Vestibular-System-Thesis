import mujoco

def net_torque(ccw_ctrl, cw_ctrl, angle=0.0):
    model = mujoco.MjModel.from_xml_path("e6_bilateral.xml")
    data = mujoco.MjData(model)
    data.ctrl[0] = ccw_ctrl
    data.ctrl[1] = cw_ctrl
    for i in range(300):
        data.qpos[0] = angle
        data.qvel[0] = 0.0
        mujoco.mj_step(model, data)
    return data.qfrc_actuator[0]

torque_ccw = net_torque(1.0, 0.0)
torque_cw = net_torque(0.0, 1.0)
print(f"CCW muscle alone -> net joint torque: {torque_ccw:>10.3f} N*m")
print(f"CW muscle alone  -> net joint torque: {torque_cw:>10.3f} N*m")


