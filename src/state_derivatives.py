import numpy as np
import pdb
from scipy.integrate import cumulative_trapezoid
from src.forward_kinematic import forward_kinematics

def state_derivatives(state, robot, controllers, phase_offsets, dt):
    q1, q2, q1_dot, q2_dot = state
    qdot = np.array([q1_dot, q2_dot])
   
    x,y = forward_kinematics(q1, q2)
    xdot, ydot = robot.get_jacob(q1, q2) @ [q1_dot, q2_dot]


    joint1_state = [x, xdot]
    joint2_state = [y, ydot]
    
    pitch1 = controllers[0].get_pitch(joint1_state)
    pitch2 = controllers[1].get_pitch(joint2_state)

    # Ver desired acceleration in x-direction
    xddot_ver,_ = controllers[0].get_desired_acceleration(joint1_state, pitch2, phase_offsets[0])  

    # Ver desied acceleration in y-direction
    yddot_ver,_ = controllers[1].get_desired_acceleration(joint2_state, pitch1, phase_offsets[1]) 
    
    F_ver = np.array([xddot_ver, yddot_ver])

    # xdot_ver_prev, ydot_ver_prev = robot.cartesian_velocity_prev
    # xdot_ver_new = xdot_ver_prev + dt * xddot_ver
    # ydot_ver_new = ydot_ver_prev + dt * yddot_ver
    # robot.cartesian_velocity_prev = np.array([xdot_ver_new, ydot_ver_new])
    
    J = robot.get_jacob(q1, q2)
    # qdot_ver = np.linalg.inv(J) @ [xdot_ver_new, ydot_ver_new]
    Jdot = robot.get_jacob_dot(q1, q2, qdot)


    u_joints = np.linalg.inv(J)@(F_ver-Jdot@qdot)

    temp = u_joints

    robot.temp.append(temp)


    M, CG, _ = robot.forward_dynamics(state[:2], state[2:], tau = 0)
    

    tau_joints = M@u_joints + CG

    M,CG, q_ddot = robot.forward_dynamics(state[:2], state[2:], tau_joints)
   
    return np.array([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])