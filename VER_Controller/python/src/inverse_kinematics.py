import numpy as np
def inverse_kinematics_2link(x, y, L1, L2):
    r_sq = x**2 + y**2
    cos_th2 = np.clip((r_sq - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
    th2 = np.arccos(cos_th2)
    k1 = L1 + L2 * np.cos(th2)
    k2 = L2 * np.sin(th2)
    th1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return th1, th2