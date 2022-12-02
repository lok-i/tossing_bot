from mujoco_sim_base import mujoco_sim
import numpy as np
from arm1_ctrl import all_yeets
from dm_control import mujoco
# from dm_control.utils import inverse_kinematics
 
from ik_solver import qpos_from_site_pose

class arm_controller:

    def __init__(
                self,
                arm_name,
                jpos_idx_frmto = [0,7],
                jvel_idx_frmto = [0,7],
                ) -> None:
        self.last_jpos = np.zeros(7)
        self.arm_name = arm_name
        self.ee_site_name = self.arm_name + "/force_sensor"
        self.kp = np.zeros((7,7))
        np.fill_diagonal(self.kp , 75)
        self.kd = np.zeros((7,7))
        np.fill_diagonal( self.kd, 7)
        self.jpos_idx_frmto = jpos_idx_frmto
        self.jvel_idx_frmto = jvel_idx_frmto

        self.jpos_nominal = np.array([0.04604284, -0.71705837, -0.0256721,   2.08973486,  0.0056602,   0.0744923, 0])

        self.set_jnames()

    def set_jnames(self):
        
        self.jnames = [
                        self.arm_name + "/Actuator1",
                        self.arm_name + "/Actuator2",
                        self.arm_name + "/Actuator3",
                        self.arm_name + "/Actuator4",
                        self.arm_name + "/Actuator5",
                        self.arm_name + "/Actuator6",
                        self.arm_name + "/Actuator7"
                        ]
    
    def ctrl_for_ee_pos(self,ee_pos,sim):

        iksoln_jpos = qpos_from_site_pose(
                                    physics,
                                    site_names = [self.ee_site_name],
                                    target_pos= [ee_pos],
                                    joint_names=self.jnames,
                                    tol=1e-14,
                                    rot_weight=1.0,
                                    regularization_threshold=0.1,
                                    regularization_strength=3e-2,
                                    max_update_norm=2.0,
                                    progress_thresh=20.0,
                                    max_steps=100,
                                    inplace=False
                                    )
        
        return(self.ctrl_for_jpos2( sim,q_des=iksoln_jpos.qpos[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]]))

        
    def ctrl_for_jpos(self,jpos=None):
        
        self.last_jpos = np.copy(sim.data.qpos[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]])
        
        if jpos is None:
            return(self.kp@( self.jpos_nominal - sim.data.qpos[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]] ) - self.kd@sim.data.qvel[self.jvel_idx_frmto[0]:self.jvel_idx_frmto[-1]])

        else:
            return(self.kp@( jpos - sim.data.qpos[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]] ) - self.kd@sim.data.qvel[self.jvel_idx_frmto[0]:self.jvel_idx_frmto[-1]])


    def ctrl_for_jpos2(self,sim,q_des):

        MassMatrix=np.zeros((sim.model.nv,sim.model.nv))   
        mujoco.mj_fullM(sim.model,MassMatrix,sim.data.qM)

        MassMatrix_arm2=MassMatrix[ self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1], self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]]

        Cq=sim.data.qfrc_bias

        kp=np.eye(7)*4
        kd=np.eye(7)*3

        qd_des = np.zeros(7)
        qdd=kp.dot(q_des-sim.data.qpos[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]])+kd.dot(qd_des-sim.data.qvel[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]])

        tau=Cq[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]]+MassMatrix_arm2.dot(qdd)

        return tau

    def ctrl_to_hold_there(self):
        return(self.kp@( self.last_jpos - sim.data.qpos[self.jpos_idx_frmto[0]:self.jpos_idx_frmto[-1]] ) - self.kd@sim.data.qvel[self.jvel_idx_frmto[0]:self.jvel_idx_frmto[-1]])

physics = mujoco.Physics.from_xml_path("assets/scene.xml")

# initialize custom mujoco simulation class
sim = mujoco_sim( 
                  render={'active':True,'on_screen':True},
                #   model_path="assets/robots/"+robot_name+"/robot.xml",                  
                  model_path="assets/scene.xml",                  

                  )

def move_camera(ct):
    if sim.sim_params['render']['active']:
        cnyr_pos =  sim.data.xpos[ sim.obj_name2id(name='conveyor',type='body') ]
        for i in range(3):        
            sim.viewer.cam.lookat[i]= cnyr_pos[i] 
        # throwing
        if ct > 8 and ct < 15:
            if sim.viewer.cam.distance> 1.5:
                sim.viewer.cam.distance -= 0.0005
            sim.viewer.cam.elevation = -15
            sim.viewer.cam.azimuth = 135
        elif ct> 15 and ct < 24:
            if sim.viewer.cam.azimuth > 90:
                sim.viewer.cam.azimuth -= 0.075
            
        elif ct> 25 and ct < 25.6:
            if sim.viewer.cam.azimuth <= 225:
                sim.viewer.cam.azimuth += 1            
            if sim.viewer.cam.elevation > -25:
                sim.viewer.cam.azimuth -= 0.5            

        elif ct> 25 and ct < 25.6:
            if sim.viewer.cam.azimuth <= 225:
                sim.viewer.cam.azimuth += 1            
            if sim.viewer.cam.elevation > -30:
                sim.viewer.cam.elevation -= 0.75     


        elif ct> 40 and ct < 45:
            if sim.viewer.cam.azimuth >= 135:
                sim.viewer.cam.azimuth -= 0.036            
            if sim.viewer.cam.elevation < -15:
                sim.viewer.cam.elevation += 0.05   
            pass
        
        elif ct> 46 and ct < 49:
            if sim.viewer.cam.azimuth >= 45:
                sim.viewer.cam.azimuth -= 0.06  
            if sim.viewer.cam.distance < 3.4:
                sim.viewer.cam.distance += 0.0011 

        elif ct> 56 and ct < 63:
            if sim.viewer.cam.azimuth <= 180:
                sim.viewer.cam.azimuth += 0.04  

        elif ct> 68 and ct < 76:
            if sim.viewer.cam.azimuth >= 47:
                sim.viewer.cam.azimuth -= 0.033  
            if sim.viewer.cam.elevation >= -23:
                sim.viewer.cam.elevation -= 0.002
        
        elif ct> 85 and ct < 87:
            if sim.viewer.cam.azimuth <= 135:
                sim.viewer.cam.azimuth -= 0.08 

# to keep the similation in pause until activated manually, press space to play
if sim.sim_params['render']['active']:
    sim.viewer._paused = True
    sim.viewer._render_every_frame = False
    sim.viewer.cam.distance = 3.5
    cam_pos = [0.25, 0.0, 0.5]

    for i in range(3):        
        sim.viewer.cam.lookat[i]= cam_pos[i] 
    sim.viewer.cam.elevation = -15
    sim.viewer.cam.azimuth = 135

# reset state / set to intitial configuration
sim.reset()


ctrl = 0*np.random.rand(sim.data.ctrl.shape[0])
# task completion flags
one_cycle_done = False

arm1_threw_screen = False

pick_counter = 0
reach_threshold = 0.1

arm1_threw_rest = False
resume_throwing = False

move_phone2arm2 = False
moved_counter = 0
move_phone2arm1 = False
arm2_unscrew_phone = False
start_unscrewing = False

uncrew_counter = 0

alpha = 0

sim.simulate_n_steps(n_steps=1)

event_verbose = True


arm1_ctrlr = arm_controller(arm_name='arm1',jpos_idx_frmto=[0,7],jvel_idx_frmto=[0,7])
arm2_ctrlr = arm_controller(arm_name='arm2',jpos_idx_frmto=[7,14],jvel_idx_frmto=[7,14])

curr_time=0
t1 = 0
t2 = 1000
t3 = 10000
t4 = 10000


while not one_cycle_done:

    ctrl[0:8]=np.zeros(8)
    ctrl[0:8]+=all_yeets(sim,curr_time,t1,t2,t3,t4)

    if not arm1_threw_screen:
        ctrl[8:15] = arm2_ctrlr.ctrl_for_jpos2(sim,q_des = arm2_ctrlr.jpos_nominal )
        for i in range(len(physics.data.qpos)):
            physics.data.qpos[i] = sim.data.qpos[i]         
        
        if curr_time > 5.0:
            eq_id = sim.obj_name2id(name='sc2con',type='equality')
            sim.model.eq_active[eq_id] = 0            
        if curr_time > 25:
            if event_verbose: print('t:',curr_time,'arm1 threw screen')
            arm1_threw_screen = True

    elif arm1_threw_screen and not move_phone2arm2:

        ctrl[8:15] = arm2_ctrlr.ctrl_for_jpos2(sim,q_des = arm2_ctrlr.jpos_nominal )
        
        if np.abs(sim.data.qpos[-1] - 0.5) <= 0.05:
            if moved_counter >= 10:
                if event_verbose: print('t:',curr_time,'moved converyer to arm 2')
                move_phone2arm2 = True 
                   
            moved_counter+=1
        else:
            ctrl[-1] += 0.01

    elif arm1_threw_screen  and not arm2_unscrew_phone:
        
        screw1_pos =  sim.data.xpos[ sim.obj_name2id(name='screw1',type='body') ]
        screw2_pos =  sim.data.xpos[ sim.obj_name2id(name='screw2',type='body') ]
        
        
        obj_pos = (1-alpha)*screw1_pos + alpha*screw2_pos


        ee_id = sim.obj_name2id(name='arm2/ee',type='body')
        ee_pos = sim.data.xpos[ee_id]

        ee2obj_error = np.linalg.norm(obj_pos - ee_pos)

        ctrl[8:15] = arm2_ctrlr.ctrl_for_ee_pos(sim=sim,ee_pos=obj_pos)

        
        if np.linalg.norm(screw2_pos-ee_pos) <= 0.03:
            if uncrew_counter > 100:
                arm2_unscrew_phone = True
                if event_verbose: print('t:',curr_time,'arm2 removed screws/solder')
            uncrew_counter +=1

        if np.linalg.norm(screw1_pos-ee_pos) <= 0.03 :
            start_unscrewing = True
        if start_unscrewing and alpha<1:    
            alpha += 0.01

    elif arm1_threw_screen and not move_phone2arm1:

        ctrl[8:15] = arm2_ctrlr.ctrl_for_jpos2(sim,q_des = arm2_ctrlr.jpos_nominal )
        
        if np.abs(sim.data.qpos[-1] - 0.) <= 0.1:
            if moved_counter >= 10:
                if event_verbose: print('t:',curr_time,'moved converyer back to arm 1')
                move_phone2arm1 = True 
            moved_counter+=1
        else:
            ctrl[-1] -= 0.01

    elif move_phone2arm1 and not arm1_threw_rest:
        ctrl[8:15] = arm2_ctrlr.ctrl_for_jpos2(sim,q_des = arm2_ctrlr.jpos_nominal )
        if not resume_throwing:


            # <weld name="c2con" active="true" body1="conveyer" body2="cover" relpose="0 0 0 1 0 0 0"/>
            # <!-- <weld name="bt2con" active="true" body1="conveyer" body2="battery" relpose="0 0 0 1 0 0 0"/> -->
            # <weld name="pcb2con" active="true" body1="conveyer" body2="pcb" relpose="0 0 0 1 0 0 0"/>

            eq_id = sim.obj_name2id(name='c2con',type='equality')
            sim.model.eq_active[eq_id] = 0

            eq_id = sim.obj_name2id(name='pcb2con',type='equality')
            sim.model.eq_active[eq_id] = 0

            t2 = curr_time
            t3 = t2+30
            resume_throwing = True
            if event_verbose: print('t:',curr_time,'resumed throwing')
        
    sim.set_control(ctrl) # send ctrl to the robot
    sim.simulate_n_steps(n_steps=1) # simulate a step forward in time
    sim.render() # render the simulation

    #transformations.euler_to_quat(np.radians(base_rpy))
    curr_time += 0.002
    if curr_time > 90:
        one_cycle_done = True
    move_camera(curr_time)
    if not event_verbose: print("curr_time:", curr_time)