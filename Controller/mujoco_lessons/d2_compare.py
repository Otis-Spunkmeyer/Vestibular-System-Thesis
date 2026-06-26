import mujoco

MOMENT_OF_INERTIA_KGM2 = 63.0
DESTABILIZING_TORQUE_NM_PER_RAD = 77.8 * 9.81 * 0.9
JOINT_DAMPING_NMS_PER_RAD = 351.0
TIMESTEP_S = 0.0002
ta_net = 10.0  # stands in for compute_net_torque_nm()'s output

body_rad = 0.0
body_vel = 0.0

print("Hand-rolled integration (mirrors sns_tc4_controller.py:988-992):")
for i in range(50):
    destab = DESTABILIZING_TORQUE_NM_PER_RAD * body_rad
    damp = JOINT_DAMPING_NMS_PER_RAD * body_vel
    body_vel += (ta_net - damp + destab) / MOMENT_OF_INERTIA_KGM2 * TIMESTEP_S
    body_rad += body_vel * TIMESTEP_S
    if i % 10 == 9:
        print(f"  step {i:2d} - body_rad: {body_rad:>10.6f}  body_vel: {body_vel:>10.6f}")

model = mujoco.MjModel.from_xml_path("d2_ankle_actuated.xml")
data = mujoco.MjData(model)
data.ctrl[0] = ta_net

print("MuJoCo mj_step:")
for i in range(50):
    mujoco.mj_step(model, data)
    if i % 10 == 9:
        print(f"  step {i:2d} - body_rad: {data.qpos[0]:>10.6f}  body_vel: {data.qvel[0]:>10.6f}")

