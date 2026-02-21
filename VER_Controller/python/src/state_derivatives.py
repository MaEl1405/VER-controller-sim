import numpy as np
def state_derivatives(state, robot, controllers, phase_offsets):
    q1, q2, q1_dot, q2_dot = state
    joint1_state = [q1, q1_dot]
    joint2_state = [q2, q2_dot]
    
    _, pitch1 = controllers[0].get_desired_acceleration(joint1_state, 0, 0)
    _, pitch2 = controllers[1].get_desired_acceleration(joint2_state, 0, 0)
    
    u1, _ = controllers[0].get_desired_acceleration(joint1_state, pitch2, phase_offsets[0])
    u2, _ = controllers[1].get_desired_acceleration(joint2_state, pitch1, phase_offsets[1])
    u_ver = np.array([u1, u2])
    
    M, CG, _ = robot.forward_dynamics(state[:2], state[2:], tau = 0)

    tau = M @ u_ver + CG

    M,CG, q_ddot = robot.forward_dynamics(state[:2], state[2:], tau)
   
    return np.array([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])