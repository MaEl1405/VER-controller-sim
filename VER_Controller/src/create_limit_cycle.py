import numpy as np
from .inverse_kinematics import inverse_kinematics_2link
from scipy.signal import savgol_filter

def prepare_joint_tables(q, q_dot, q_ddot):
    center = np.mean(q)
    q_centered = q - center
    theta = np.unwrap(np.arctan2(q_dot, q_centered))
    r_d = np.sqrt(q_centered**2 + q_dot**2)
    rho = np.linspace(theta[0], theta[-1], len(q))  # calculate pitch for ohase ( first harmonic)
    sort_idx = np.argsort(theta)

    table = {'center': center, 
            'theta': theta[sort_idx], 
            'r_d': r_d[sort_idx], 
            'beta': q_ddot[sort_idx], 
            'rho': rho[sort_idx]}
    
    return table, rho


def create_limit_cycles_from_path(num_points=2000, l1=1.0, l2=1.0, radius=0.5, center=(1.0, -1.0)):
    T = 2 * np.pi
    t = np.linspace(0, T, num_points, endpoint=False)
    dt = T / num_points

    x_path = center[0] + radius * np.cos(t)
    y_path = center[1] + radius * np.sin(t)
    q1, q2 = inverse_kinematics_2link(x_path, y_path, l1, l2)

    window = 51
    q1_dot = savgol_filter(np.unwrap(q1), window_length=window, polyorder=3, deriv=1, delta=dt)
    q2_dot = savgol_filter(np.unwrap(q2), window_length=window, polyorder=3, deriv=1, delta=dt)
    q1_ddot = savgol_filter(np.unwrap(q1), window_length=window, polyorder=3, deriv=2, delta=dt)
    q2_ddot = savgol_filter(np.unwrap(q2), window_length=window, polyorder=3, deriv=2, delta=dt)

    tables1, rho1 = prepare_joint_tables(q1, q1_dot, q1_ddot)
    tables2, rho2 = prepare_joint_tables(q2, q2_dot, q2_ddot)
    
    # Calculate the inherent phase offset between the two joints
    phase_difference_rad = rho1 - rho2
    avg_phase_offset_rad = np.mean(phase_difference_rad)
    
    return tables1, tables2, avg_phase_offset_rad

