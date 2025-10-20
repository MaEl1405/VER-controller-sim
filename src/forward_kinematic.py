import numpy as np

def forward_kinematics(q1, q2, L1=1.0, L2=1.0):
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)

    return x,y
