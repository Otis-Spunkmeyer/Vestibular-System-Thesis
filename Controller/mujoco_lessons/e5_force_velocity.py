import mujoco

def measure_force_at_velocity(velocity, angle=0.0):
    model = mujoco.MjModel.from_xml_path("e2_muscle.xml")
    data = mujoco.MjData(model)
    data.ctrl[0] = 1.0
    for i in range(300):
        data.qpos[0] = angle
        data.qvel[0] = velocity
        mujoco.mj_step(model, data)
    return data.actuator_force[0]

for velocity in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    force = measure_force_at_velocity(velocity)
    print(f"velocity: {velocity:+.2f} rad/s  ->  actuator_force: {force:>10.3f} N")
