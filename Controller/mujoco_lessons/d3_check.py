import mujoco

model = mujoco.MjModel.from_xml_path("d3_surface.xml")
data = mujoco.MjData(model)

print(f"Joints: {model.njnt}  DOFs: {model.nv}  qpos slots: {model.nq}")
print(f"qpos (surface, ankle): {data.qpos}")
