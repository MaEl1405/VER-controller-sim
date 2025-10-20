import numpy as np
import pdb
from src.forward_kinematic import forward_kinematics

def state_derivatives(state, robot, controllers, phase_offsets):
    q1, q2, q1_dot, q2_dot = state
    qdot = np.array([q1_dot, q2_dot])
   
    x,y = forward_kinematics(q1, q2)
    xdot, ydot = robot.get_jacob(q1, q2) @ [q1_dot, q2_dot]


    joint1_state = [x, xdot]
    joint2_state = [y, ydot]
    
    pitch1 = controllers[0].get_pitch(joint1_state)
    pitch2 = controllers[1].get_pitch(joint2_state)

    Fx,_ = controllers[0].get_desired_acceleration(joint1_state, pitch2, phase_offsets[0])
    Fy,_ = controllers[1].get_desired_acceleration(joint2_state, pitch1, phase_offsets[1])
    F_ver = np.array([Fx, Fy])
    print(np.linalg.det(robot.get_jacob(q1,q2)))
    u_joints = np.linalg.inv(robot.get_jacob(q1,q2)) @ F_ver

    M, CG, _ = robot.forward_dynamics(state[:2], state[2:], tau = 0)
    
    
    tau_joints = M@u_joints + CG

    M,CG, q_ddot = robot.forward_dynamics(state[:2], state[2:], tau_joints)
   
    return np.array([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])