from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    from dm_control import mujoco
    import mujoco_viewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install dm_control and mujoco_viewer)".format(e))

equality_constraint_id2type = {
                                'connect':0,
                                'weld':1,
                                'joint':2,
                                'tendon':3,
                                'distance':4
                                }

class mujoco_sim(gym.Env):
    """Superclass for all MuJoCo environments."""
    def __init__(self, **kwargs ):
        
        
        self.sim_params = kwargs
        necessary_env_args = ['model_path']
        
        default_env_args = {
                          'render':{'active':False,'on_screen':False},
                          'set_on_rack': False,
                          'mocap':False,
                          'frame_skip':1
                          }
        
        for key in necessary_env_args:
            if key not in self.sim_params.keys():
                raise Exception('necessary arguments are absent. Check:'+str(necessary_env_args))        
        
        for key in default_env_args.keys():
            if key not in self.sim_params.keys():
                self.sim_params[key] = default_env_args[key]
        
        
        self.model = mujoco.MjModel.from_xml_path(self.sim_params['model_path'])
        self.data = mujoco.MjData(self.model)

        # to deactivate gravity
        # self.model.opt.gravity[2] = 0
        # print(self.model.opt.gravity)

        if self.sim_params['render']['active']:
            if self.sim_params['render']['on_screen']:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            else:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data,'offscreen')

    def reset(self): # -1 is random
        
        # TBD
        # alter qpos0 here to set inital state 
        # eg: model.qpos0[2] = i*0.1 + 0.5
        mujoco.mj_resetData(self.model,self.data)
        
        # TBC
        # self.remove_all_weld_contraints()
        # TBD
        ob = None
        # TBC
        # self.set_required_weld_contraints()

        return ob

    def get_state(self):
        return self.data.qpos, self.data.qvel

    def get_sensor_data(self,sensor_name):
        # TBD
        pass    

    @property
    def dt(self):
        # print(self.model.opt.timestep)
        return self.model.opt.timestep 

    def set_control(self,ctrl):
        self.data.ctrl[:] = ctrl

    def set_qpos(self,qpos):
        # set only for the actual robot, skip the visual clone
        self.data.qpos[0:qpos.shape[0]] = qpos[:]

    def set_qvel(self,qvel):
        # set only for the actual robot, skip the visual clone
        self.data.qvel[0:qvel.shape[0]] = qvel[:]

    def simulate_n_steps(self, n_steps=1):
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    
    def render(self):

        if self.sim_params['render']['active']:
            if self.sim_params['render']['on_screen']:
                self.viewer.render()
            else:
                return self.viewer.read_pixels(camid=0)

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def print_all_contacts(self):
        for coni in range(self.data.ncon):
            print('  Contact %d:' % (coni,))
            con = self.data.contact[coni]
            print('    dist     = %0.3f' % (con.dist,))
            print('    pos      = %s' % (str_mj_arr(con.pos),))
            print('    frame    = %s' % (str_mj_arr(con.frame),))
            print('    friction = %s' % (str_mj_arr(con.friction),))
            print('    dim      = %d' % (con.dim,))
            print('    geom1    = %d' % (con.geom1,))
            print('    g1_name  = %s' % ( self.model.geom_id2name(con.geom1),))
            
            print('    geom2    = %d' % (con.geom2,))
            print('    g2_name  = %s' % ( self.model.geom_id2name(con.geom2),))

    def remove_all_weld_contraints(self):        
        if self.model.eq_type != None:
            for eq_id,eq_type in enumerate(self.model.eq_type):
                if eq_type ==  equality_constraint_id2type['weld']:
                    self.model.eq_active[eq_id] = 0

    def set_required_weld_contraints(self):
        # TBC
        for eq_id,(eq_obj1id,eq_obj2id,eq_type) in enumerate(zip(self.model.eq_obj1id,self.model.eq_obj2id,self.model.eq_type)):
            
            if eq_type ==  equality_constraint_id2type['weld']:
                
                if self.model.body_id2name(eq_obj2id) == 'world':
                    if self.sim_params['set_on_rack']:
                        self.model.eq_active[eq_id] = 1
                
                if 'mocap_' in self.model.body_id2name(eq_obj1id) or 'mocap_' in self.model.body_id2name(eq_obj2id):
                    if self.sim_params['mocap']:
                        self.model.eq_active[eq_id] = 1 
