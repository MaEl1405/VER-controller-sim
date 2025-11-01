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
    zero_matrix_2x2 = np.zeros([2,2])
    identity_matrix_2x2 = np.eye(2)

    # Solve Control Equations
    coef_matrix = np.block([
        [J_master, zero_matrix_2x2, zero_matrix_2x2, zero_matrix_2x2],
        [zero_matrix_2x2, J_slave, zero_matrix_2x2, zero_matrix_2x2],
        [zero_matrix_2x2, M_slave, -J_slave.T, zero_matrix_2x2],
        [M_master, zero_matrix_2x2, J_master.T, -identity_matrix_2x2]
    ])

    b_matrix = np.block([
        [desired_acc.reshape(2,1) - (J_dot_master@q_dot_master).reshape(2,1)],
        [desired_acc.reshape(2,1) - J_dot_slave@q_dot_slave.reshape(2,1)],
        [-CG_slave.reshape(2,1) ],
        [-CG_master.reshape(2,1) ]
    ])

    control_solution = np.linalg.solve(coef_matrix, b_matrix)

    # Desired joint accelerations
    q_ddot_master_desired = control_solution[0:2]
    q_ddot_slave_required = control_solution[2:4]

    # Desired master torque (compensate dynamics + constraint force)
    lambda_force_required = control_solution[4:6]
    tau_joints_master = control_solution[6:8]
    tau_joints_slave = np.zeros([2,1])
    
    ## Calculate constraints Error 
    P_master = np.array(master_arm.forward_kinematics(q_master[0], q_master[1])) # master end-effector pos
    P_slave  = np.array(slave_arm.forward_kinematics(q_slave[0], q_slave[1]))    # Slave end-effector pos
    V_master = J_master @ q_dot_master # Master end-effecto velocity
    V_slave  = J_slave @ q_dot_slave   # Slave end-effector velocity

    e_P = P_master - P_slave  # end effector position error
    e_V = V_master - V_slave  # End effector Velocity error

    ## Solve Plantt Equations
    zeros_2x2 = np.zeros((2,2))
    K_row1 = np.hstack([M_master, zeros_2x2, J_master.T])
    K_row2 = np.hstack([zeros_2x2, M_slave, -J_slave.T])
    K_row3 = np.hstack([J_master, -J_slave, zeros_2x2])
    K = np.vstack([K_row1, K_row2, K_row3])

    # Ordinate variables
    W_row1 = tau_joints_master - CG_master.reshape(2,1)
    W_row2 = tau_joints_slave -CG_slave.reshape(2,1)
    W_row3 = J_dot_slave@q_dot_slave - J_dot_master@q_dot_master - (alpha_stabilize * e_P) -(beta_stabilize * e_V)
    W = np.vstack([W_row1, W_row2, W_row3.reshape(2,1)])

    # Solve for accelerations and forces
    plant_solution = np.linalg.solve(K, W)  # X = [qddot_mater qddot_slave lambda]

    ## Solve Equations
    qddot_master = plant_solution[0:2]
    qddot_slave  = plant_solution[2:4]
    lambda_force = plant_solution[4:6]

    # Concatenate all flat arrays
    state_derivative = np.concatenate([
        q_dot_master, 
        q_dot_slave, 
        qddot_master.flatten(), 
        qddot_slave.flatten()
    ]) # Shape (8,)

    debug_info = {"tau_joints_master": tau_joints_master, "lambda": lambda_force}
   
    return state_derivative, debug_info
