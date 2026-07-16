import mujoco

# NOTE: the actuators in e6_bilateral.xml are now force actuators (BPAs), so ctrl is
# tendon TENSION in Newtons, not 0-1 activation. This checks that the Flx and Ext
# actuators produce opposite joint torques.
def net_torque(flx_force, ext_force, angle=0.0):
    model = mujoco.MjModel.from_xml_path("e6_bilateral.xml")
    data = mujoco.MjData(model)
    data.ctrl[0] = flx_force
    data.ctrl[1] = ext_force
    for i in range(300):
        data.qpos[0] = angle
        data.qvel[0] = 0.0
        mujoco.mj_step(model, data)
    return data.qfrc_actuator[0]

torque_flx = net_torque(200.0, 0.0)
torque_ext = net_torque(0.0, 200.0)
print(f"Flx BPA alone (200 N) -> net joint torque: {torque_flx:>10.3f} N*m")
print(f"Ext BPA alone (200 N) -> net joint torque: {torque_ext:>10.3f} N*m")


