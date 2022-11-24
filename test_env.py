from mujoco_sim_base import mujoco_sim
import numpy as np

#select robot name, available options can be seen in ./assets/robots/
robot_name='kinova3'

# initialize custom mujoco simulation class
sim = mujoco_sim( 
                  render={'active':True,'on_screen':True},
                  model_path="assets/robots/"+robot_name+"/robot.xml",                  
                  )

# to keep the similation in pause until activated manually, press space to play
if sim.sim_params['render']['active']:
    sim.viewer._paused = True
    sim.viewer._render_every_frame = False
    sim.viewer.cam.distance = 4
    cam_pos = [0.0, 0.0, 0.75]

    for i in range(3):        
        sim.viewer.cam.lookat[i]= cam_pos[i] 
    sim.viewer.cam.elevation = -15
    sim.viewer.cam.azimuth = 180

# reset state / set to intitial configuration
sim.reset()

# acessing genealised pos and vel vaiables fro simulation
print('qpos :',sim.data.qpos,'len:',sim.data.qpos.shape)
print('qvel :',sim.data.qvel,'len:',sim.data.qvel.shape)

ctrl_dt = 0.03 # to get a control frequency of ~33.33 Hz (can go higher though)
nsim_steps_per_ctrl = int(ctrl_dt/sim.dt)

while True:
    # generate new ctrl(randm here) every 0.03s, i.e.at 33 Hz
    torques = np.random.rand(sim.data.ctrl.shape[0])
    for i in range(nsim_steps_per_ctrl):
        sim.set_control(ctrl=torques) # send ctrl to the robot
        sim.simulate_n_steps(n_steps=1) # simulate a step forward in time
        sim.render() # render the simulation