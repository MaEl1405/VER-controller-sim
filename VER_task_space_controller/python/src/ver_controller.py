import numpy as np
import pdb 

class VER_Controller:
    def __init__(self, lookup_tables, P_gain, K_gain):
        self.tables = lookup_tables  # precomputed lookup table (θ, r_d, β, ρ)
        self.P = P_gain              # radial gain
        self.K = K_gain              # coupling gain
        

    def get_desired_acceleration(self, current_state, other_joint_pitch, phase_offset_rad=0.0):
        x, x_dot = current_state

        # Convert state to polar form
        x_centered = x - self.tables['center']
        current_theta = np.arctan2(x_dot, x_centered)
        current_r_sq = x_centered**2 + x_dot**2

        # Interpolate desired radius and β term from table
        desired_r_d = np.interp(current_theta, self.tables['theta'], self.tables['r_d'], period=2*np.pi)
        beta = np.interp(current_theta, self.tables['theta'], self.tables['beta'], period=2*np.pi)
        current_pitch = self.get_pitch(current_state)

        # Compute radial and coupling terms
        alpha = self.P * np.tanh(x_dot) * (desired_r_d**2 - current_r_sq)
        target_pitch = other_joint_pitch + phase_offset_rad
        gamma = self.K * (np.sin(target_pitch) - np.sin(current_pitch))

        # Total desired acceleration
        u_ver = alpha + beta + gamma
        return u_ver
    
    def get_pitch(self, current_state):    
        x, x_dot = current_state
        x_centered = x - self.tables['center']
        current_theta = np.arctan2(x_dot, x_centered)  # current phase angle
        current_pitch = current_theta                   # direct phase mapping
        # alt: interpolate from self.tables['rho'] if used
        return current_pitch
