import mujoco
from mujoco_sim_base import mujoco_sim
from dm_control.utils import transformations
import numpy as np

def all_yeets(sim,t,t1,t2,t3,t4):
    tau=np.zeros(8)

    tau[0:8]+=yeet_sequence(sim,t,t1,t2,0.07,0.528,0,23)
    tau[0:8]+=yeet_sequence(sim,t,t2,t3,0.06,0.528,180,19)
    tau[0:8]+=yeet_sequence(sim,t,t3,t4,0.35,0.524,100,23)
    tau[0:8]+=yeet_sequence(sim,t,t4,10000,0.05,0.528,80,23)

    return tau

def yeet_sequence(sim,t,t_start,t_max,strenght,height,azimuth,omega):
    tau=np.zeros(8)

    dt_yeet1=0.8
    #start
    t0=t_start
    #time to move to approximate location
    t1=t0+4
    #time to activate adhesion
    t2=t0+17
    #time to lift the phone
    t3=t0+19
    #time to move into position
    t3_2=t3+1
    #time to start throw
    t4=t0+23
    #time to let go
    t5=t4+dt_yeet1
    #end
    t6=t_max

    q_d=np.deg2rad([0,10,0,80,0,90,0])
    qd_d=[0,0,0,0,0,0,0]

    #move near-ish
    tau[0:7]=pd_joint_ctrl(sim,q_d,qd_d,t,t0,t1)
    #move up close
    tau[0:7]+=pd_task_ctrl(sim,[0.25,0,height],np.deg2rad([0,180,0]),qd_d,t,t1,t3)
    #grab
    tau[7]+=grab(t,t2,t5,strenght)
    #lift the phone gently
    tau[0:7]+=pd_task_ctrl(sim,[0.25,0,height+0.1],np.deg2rad([0,180,0]),qd_d,t,t3,t3_2)
    #get into throw position
    # tau[7]+=grab(t,t3_2-0.1,t5,strenght*3)
    q_d[5]=np.deg2rad(145)
    q_d[0]=np.deg2rad(azimuth)
    tau[0:7]+=pd_joint_ctrl(sim,q_d,qd_d,t,t3_2,t4)
    #accelerate
    qd_d[5]=-omega
    tau[0:7]+=yeet1(sim,q_d,qd_d,t,t4,t5)
    #reset
    q_d[0]=0
    q_d[5]=np.deg2rad(90)
    qd_d[5]=0
    tau[0:7]+=pd_joint_ctrl(sim,q_d,qd_d,t,t5,t6)

    return tau

def yeet1(sim,q_des,qd_des,t,t_start,t_end):

    MassMatrix=np.zeros((sim.model.nv,sim.model.nv))   
    mujoco.mj_fullM(sim.model,MassMatrix,sim.data.qM)

    MassMatrix_arm1=MassMatrix[0:7,0:7]

    Cq=sim.data.qfrc_bias

    kp=np.eye(7)*4
    kd=np.eye(7)*3

    kp[5,5]=0

    qdd=kp.dot(q_des-sim.data.qpos[0:7])+kd.dot(qd_des-sim.data.qvel[0:7])

    if t<t_end and t>t_start:
        tau=Cq[0:7]+MassMatrix_arm1.dot(qdd)
    else:
        tau=np.zeros(7)

    return tau

def pd_joint_ctrl(sim,q_des,qd_des,t,t_start,t_end):

    MassMatrix=np.zeros((sim.model.nv,sim.model.nv))   
    mujoco.mj_fullM(sim.model,MassMatrix,sim.data.qM)

    MassMatrix_arm1=MassMatrix[0:7,0:7]

    Cq=sim.data.qfrc_bias

    kp=np.eye(7)*4
    kd=np.eye(7)*3

    qdd=kp.dot(q_des-sim.data.qpos[0:7])+kd.dot(qd_des-sim.data.qvel[0:7])

    if t<t_end and t>t_start:
        tau=Cq[0:7]+MassMatrix_arm1.dot(qdd)
    else:
        tau=np.zeros(7)

    return tau

def pd_task_ctrl(sim,p_des,r_des,v_des,t,t_start,t_end):

    id=sim.obj_name2id("arm1/4boxes",type='body')
    pos=sim.data.xpos[id]
    rot=sim.data.xquat[id]

    jv=np.zeros((3,sim.model.nv)) 
    jr=np.zeros((3,sim.model.nv)) 


    jv_arm1=jv[:,0:7]
    jr_arm1=jr[:,0:7]

    J_arm1=np.concatenate((jv_arm1, jr_arm1), axis=0)

    mujoco.mj_jac(sim.model,sim.data, jv, jr, pos, id)

    MassMatrix=np.zeros((sim.model.nv,sim.model.nv))   
    mujoco.mj_fullM(sim.model,MassMatrix,sim.data.qM)

    MassMatrix_arm1=MassMatrix[0:7,0:7]

    Cq=sim.data.qfrc_bias

    pos_d=p_des[0:3]

    pos_err=pos_d-pos

    roll_d=r_des[0]
    pitch_d=r_des[1]
    yaw_d=r_des[2]

    H_d=transformations.euler_to_rmat([yaw_d,pitch_d,roll_d],'ZYX')
    R_d=H_d[0:3,0:3]

    H_s=transformations.quat_to_mat(rot)

    R_eul_temp=transformations.quat_to_euler(rot,'ZYX')
    R_eul=[R_eul_temp[2],R_eul_temp[1],R_eul_temp[0]]


    R_s=H_s[0:3,0:3]
    
    R_err_m=np.dot(R_d,np.transpose(R_s))

    theta=np.arccos((R_err_m[0,0]+R_err_m[1,1]+R_err_m[2,2])/2)
    R_err=[R_err_m[2,1]-R_err_m[1,2],R_err_m[0,2]-R_err_m[2,0],R_err_m[1,0]-R_err_m[0,1]]/(2*np.sin(theta))

    p_err=np.concatenate((pos_err,R_err),axis=0)


    M=MassMatrix_arm1


    kp=np.eye(3)*2
    kd=np.eye(7)*4

    pitch=sim.data.qpos[1]+sim.data.qpos[3]+sim.data.qpos[5]

    # print(f'pitch={np.rad2deg(pitch)}')

    if t<t_end and t>t_start:
        numerator=np.dot(np.linalg.inv(M),np.transpose(J_arm1))
        denumerator=np.dot(J_arm1,np.dot(np.linalg.inv(M),np.transpose(J_arm1)))

        # J_inv=np.dot(numerator,np.linalg.inv(denumerator))

        jv_inv=np.linalg.pinv(jv_arm1)
        jr_inv=np.linalg.pinv(jr_arm1)

        

        xdd=kp.dot(pos_err)
        xdd2=-(r_des-R_eul)

        

        # print(R_eul[1])

        pitch_err=r_des[1]-pitch

        jr_inv=np.zeros(7)
        jr_inv[5]=1



        qdd=np.dot(jv_inv,xdd)+jr_inv*pitch_err*20+kd.dot(-sim.data.qvel[0:7])

        tau=Cq[0:7]+MassMatrix_arm1.dot(qdd)
    else:
        tau=np.zeros(7)

    return tau

def grab(t,t_start,t_end,strenght):
    if t<t_end and t>t_start:
        return strenght
    else:
        return 0

    
