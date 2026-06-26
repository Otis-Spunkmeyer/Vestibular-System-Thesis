import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('b6_sensor.xml')
data = mujoco.MjData(model)

data.ctrl[0] = 0.5

mujoco.viewer.launch(model, data)