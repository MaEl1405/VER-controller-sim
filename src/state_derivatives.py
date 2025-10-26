import numpy as np

# Compute joint derivatives and dynamics for "task space VER-controlled" 2-DOF robot
def state_derivatives(state, robot, controllers, phase_offsets, dt):
    q1, q2, q1_dot, q2_dot = state              # joint positions and velocities
    qdot = np.array([q1_dot, q2_dot])           # vector form of joint velocities
   
    # Forward kinematics: Cartesian positions
    x, y = robot.forward_kinematics(q1, q2)
    # Jacobian-based Cartesian velocities
    xdot, ydot = robot.get_jacob(q1, q2) @ [q1_dot, q2_dot]

    # Joint state vectors (position, velocity)
    joint1_state = [x, xdot]
    joint2_state = [y, ydot]
    
    # Compute pitch (phase) for each controller
    pitch1 = controllers[0].get_pitch(joint1_state)
    pitch2 = controllers[1].get_pitch(joint2_state)

    # VER desired accelerations in task space
    xddot_ver = controllers[0].get_desired_acceleration(joint1_state, pitch2, phase_offsets[0])  
    yddot_ver = controllers[1].get_desired_acceleration(joint2_state, pitch1, phase_offsets[1]) 
    
    F_ver = np.array([xddot_ver, yddot_ver])   # desired accelerations 

    # Get Jacobian and Jacobian derivative
    J = robot.get_jacob(q1, q2)
    Jdot = robot.get_jacob_dot(q1, q2, qdot)

    # Robot dynamics
    M, CG, _ = robot.forward_dynamics(state[:2], state[2:], tau=0)
    
    # Constraint matrix [ M  -Jᵀ; J  0 ]
    coef_matrix = np.block([
        [M, -np.transpose(J)],
        [J, np.zeros((2, 2))]
    ])
    
    # Construct right-hand side vector [ -CG; -(J̇q̇ - F_ver) ]
    b_matrix = np.block([
        [-CG.reshape([2, 1])],
        [(-Jdot @ qdot + F_ver).reshape([2, 1])]
    ])
    
    # Solve for joint torques and constraint forces
    results = np.linalg.inv(coef_matrix) @ b_matrix

    # Extract joint accelerations and Cartesian forces
    u0, u1 = results[0][0], results[1][0]
    Fx, Fy = results[2][0], results[3][0]
    
    tau_joints = np.transpose(J) @ [Fx, Fy]     # map Cartesian force to joint torque

    # Final forward dynamics with VER torque
    M, CG, q_ddot = robot.forward_dynamics(state[:2], state[2:], tau_joints)
   
    # Return derivative of state vector: [ q̇₁, q̇₂, q̈₁, q̈₂ ]
    return np.array([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])
