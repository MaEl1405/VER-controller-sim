import numpy as np
import pdb
alpha_stabilize = 100.0  # virtual spring term 
beta_stabilize  = 20.0   # virtual damper term
 
def state_derivatives(state, robots, controllers, phase_offsets, dt, t_offset):
    # Extract robots & states
    master_arm, slave_arm = robots
    q_master = state[0:2]
    q_slave  = state[2:4]
    q_dot_master = state[4:6]
    q_dot_slave  = state[6:8]

    # Master kinematics
    x_master, y_master= master_arm.forward_kinematics(q_master[0], q_master[1])
    J_master = master_arm.get_jacob(q_master[0], q_master[1])
    xdot_master, ydot_master = J_master @ q_dot_master

    # Joint state vectors (position, velocity) for master
    master_joint1_state = [x_master, xdot_master]
    master_joint2_state = [y_master, ydot_master]
    
    # Compute pitch (phase) for each controller
    x_pitch = controllers[0].get_pitch(master_joint1_state) # X-axis pitch
    y_pitch = controllers[1].get_pitch(master_joint2_state) # Y-axis pitch

    # VER desired accelerations in task space
    xddot_ver = controllers[0].get_desired_acceleration(master_joint1_state, y_pitch, phase_offsets[0])  
    yddot_ver = controllers[1].get_desired_acceleration(master_joint2_state, x_pitch, phase_offsets[1]) 
    
    # Define VER Output
    desired_acc = np.array([xddot_ver, yddot_ver])   # VER generated desired accelerations 

    # Slave dynamics
    M_slave, CG_slave, _ = slave_arm.forward_dynamics(q_slave, q_dot_slave, tau = np.zeros(2))
    J_slave = slave_arm.get_jacob(q_slave[0], q_slave[1])     # Jacobian of slave
    J_dot_slave = slave_arm.get_jacob_dot(q_slave[0], q_slave[1], q_dot_slave) # Jdot of slave
    
    # Master dynamics
    M_master, CG_master, _ = master_arm.forward_dynamics(q_master, q_dot_master, tau=np.zeros(2))
    J_dot_master = master_arm.get_jacob_dot(q_master[0], q_master[1], q_dot_master)
    
    # Desired accelerations 
    q_ddot_master_desired = np.linalg.solve(J_master, desired_acc - J_dot_master @ q_dot_master)
    constraint_term = J_master @ q_ddot_master_desired + J_dot_master @ q_dot_master - J_dot_slave @ q_dot_slave
    q_ddot_slave_required = np.linalg.solve(J_slave, constraint_term)

    # Desired master torque (compensate dynamics + constraint force)
    lambda_force_required = np.linalg.solve(J_slave.T, M_slave @ q_ddot_slave_required + CG_slave)
    tau_joints_master = (M_master @ q_ddot_master_desired) + CG_master + (J_master.T @ lambda_force_required)
    tau_joints_slave = np.zeros(2)

    ## Calculate constraints Error 
    P_master = np.array(master_arm.forward_kinematics(q_master[0], q_master[1])) # master end-effector pos
    P_slave  = np.array(slave_arm.forward_kinematics(q_slave[0], q_slave[1]))    # Slave end-effector pos
    V_master = J_master @ q_dot_master # Master end-effecto velocity
    V_slave  = J_slave @ q_dot_slave   # Slave end-effector velocity

    e_P = P_master - P_slave  # end effector position error
    e_V = V_master - V_slave  # End effector Velocity error

    ## Define K and W matrix (K@X = W) Plant Modeling
    zeros_2x2 = np.zeros((2,2))
    K_row1 = np.hstack([M_master, zeros_2x2, J_master.T])
    K_row2 = np.hstack([zeros_2x2, M_slave, -J_slave.T])
    K_row3 = np.hstack([J_master, -J_slave, zeros_2x2])
    K = np.vstack([K_row1, K_row2, K_row3])


    W_row1 = tau_joints_master - CG_master
    W_row2 = tau_joints_slave -CG_slave
    W_row3 = J_dot_slave@q_dot_slave - J_dot_master@q_dot_master - (alpha_stabilize * e_P) -(beta_stabilize * e_V)

    W = np.vstack([W_row1.reshape(2,1), W_row2.reshape(2,1), W_row3.reshape(2,1)])

    # Solve for accelerations and forces
    X = np.linalg.solve(K, W)  # X = [qddot_mater qddot_slave lambda]

    ## Solve Equations
    qddot_master = X[0:2]
    qddot_slave  = X[2:4]
    lambda_force = X[4:6]

    # Concatenate all flat arrays
    state_derivative = np.concatenate([
        q_dot_master, 
        q_dot_slave, 
        qddot_master.flatten(), 
        qddot_slave.flatten()
    ]) # Shape (8,)

    debug_info = {"tau_joints_master": tau_joints_master, "lambda": lambda_force}
   
    return state_derivative, debug_info
