import numpy as np
from scipy.signal import savgol_filter

# Compute position, velocity, and acceleration for circular limit cycle motion
def calc_limit_cycle(axis_name, w, t, circle_center, circle_rad):
    if axis_name == "x":  # X-axis: sine-based motion
        x = circle_center[0] + circle_rad * np.sin(w * t)
        xdot = circle_rad * w * np.cos(w * t)
        xddot = -circle_rad * (w ** 2) * np.sin(w * t)
        return x, xdot, xddot
    if axis_name == "y":  # Y-axis: cosine-based motion
        y = circle_center[1] + circle_rad * np.cos(w * t)
        ydot = -circle_rad * w * np.sin(w * t)
        yddot = -circle_rad * (w ** 2) * np.cos(w * t)
        return y, ydot, yddot

# Build Limit Cycle feature tables (phase, amplitude, acceleration)
def prepare_joint_tables(q, q_dot, q_ddot, axis_name):
    center = -1 if axis_name == "y" else np.mean(q)    # define center point
    q_centered = q - center                            # shift to zero-centered
    theta = np.unwrap(np.arctan2(q_dot, q_centered))   # compute continuous phase
    r_d = np.sqrt(q_centered**2 + q_dot**2)            # radial amplitude 
    
    rho = np.linspace(theta[0], theta[-1], len(q))     # evenly spaced phase vector
    sort_idx = np.argsort(theta)                       # sort by phase order
    
    table = {
        'center': center,
        'theta': theta[sort_idx],
        'r_d': r_d[sort_idx],
        'beta': q_ddot[sort_idx],
        'rho': rho[sort_idx]
    }
    return table, rho

# Wrapper: compute phase table for limit-cycle trajectory
def table_rho(axis_name, w, t, circle_center, circle_rad):
    pos, vel, acc = calc_limit_cycle(axis_name, w, t, circle_center, circle_rad)
    table, rho = prepare_joint_tables(pos, vel, acc, axis_name)
    return table, rho
