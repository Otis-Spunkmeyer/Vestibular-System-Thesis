import mujoco

def measure_force_at_angle(angle):
    model = mujoco.MjModel.from_xml_path("e2_muscle.xml")
    data = mujoco.MjData(model)
    data.ctrl[0] = 1.0
    for i in range(300):
        data.qpos[0] = angle
        data.qvel[0] = 0.0
        mujoco.mj_step(model, data)
    return data.actuator_force[0], data.actuator_length[0], model.actuator_lengthrange[0]

for angle in [-0.6, -0.3, 0.0, 0.3, 0.6]:
    force, length, lengthrange = measure_force_at_angle(angle)
    print(f"angle: {angle:+.2f}  length: {length:.4f}  lengthrange: {lengthrange}  force: {force:>10.3f} N")
