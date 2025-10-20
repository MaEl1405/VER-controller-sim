import numpy as np
from scipy.signal import savgol_filter

def calc_limit_cycle(axis_name, w, t, circle_center, circle_rad):
    """
    """

    if axis_name == "x":
        x = circle_center[0] +  circle_rad*np.sin(w*t)
        xdot =  circle_rad*w*np.cos(w*t)
        xddot = -circle_rad*(w**2)*np.sin(w*t) 
        return x, xdot, xddot
    if axis_name == "y":
        y = circle_center[1] + circle_rad*np.cos(w*t)
        ydot = -circle_rad*w*np.sin(w*t)
        yddot = -circle_rad*(w**2)*np.cos(w*t) 
        return y, ydot, yddot


def prepare_joint_tables(q, q_dot, q_ddot, axis_name):
    if axis_name == "y":
        center = -1
    else:
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

def table_rho(axis_name, w, t, circle_center, circle_rad):
    pos , vel, acc = calc_limit_cycle(axis_name=axis_name, w=w, t=t, circle_center=circle_center , circle_rad=circle_rad)
    table, rho  = prepare_joint_tables(pos, vel, acc, axis_name=axis_name)

    return table, rho

    
   

