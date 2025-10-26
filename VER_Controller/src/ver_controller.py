import numpy as np

class VER_Controller:
    def __init__(self, lookup_tables, P_gain, K_gain):
        self.tables = lookup_tables
        self.P = P_gain
        self.K = K_gain

    def get_desired_acceleration(self, current_state, other_joint_pitch, phase_offset_rad=0.0):

        q, q_dot = current_state
        
        #Calculate current state in polar coordinates
        q_centered = q - self.tables['center']
        current_theta = np.arctan2(q_dot, q_centered)
        current_r_sq = q_centered**2 + q_dot**2

        # Interpolate from lookup table to find desired values for the current phase 
        desired_r_d = np.interp(current_theta, self.tables['theta'], self.tables['r_d'], period=2*np.pi)
        current_pitch = np.interp(current_theta, self.tables['theta'], self.tables['rho'], period=2*np.pi)
        
        # Attractor(alpha): Pull the state towards the desired limit cycle radius.
        alpha = self.P * np.tanh(q_dot) * (desired_r_d**2 - current_r_sq)

        # Synchronizer (gamma): Couples the joint to another joint's phase.
        target_pitch = other_joint_pitch + phase_offset_rad
        gamma = self.K * (np.sin(target_pitch) - np.sin(current_pitch))

        # Accelerator (beta): 
        beta = np.interp(current_theta, self.tables['theta'], self.tables['beta'], period=2*np.pi)  

        # VER control control output without dynamic compensation
        u_ver = alpha + beta + gamma
        
        return u_ver, current_pitch