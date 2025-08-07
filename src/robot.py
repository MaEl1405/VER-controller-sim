import numpy as np



class TwoLinkArm:
    """
    Represent the physical properties and dynamics of a 2-link planar manipulator
    """

    def __init__(self, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81, link_width=0.1, method = "point"):
        self.m1 ,self.m2 = m1, m2 
        self.L1, self.L2 = L1, L2  
        self.g = g
        self.link_width = link_width
        self.method = method.lower()  # get method for calculating forward dynamics (point or distributed)

        if self.method not in ["point", "distributed"]:
            raise ValueError("Selected method must be 'point' or 'distributed'.")
    

    def _compute_point_mass_dynamics(self, q, q_dot):
        """ """
        q1, q2 = q
        q1_dot, q2_dot = q_dot

        c2 = np.cos(q2)
        c12 = np.cos(q1 + q2)

        # Mass matrix
        M11  = (self.m1 + self.m2) * self.L1**2 + self.m2 * self.L2**2 + 2 * self.m2 * self.L1 * self.L2 * c2
        M12 = self.m2 * self.L2**2 + self.m2 * self.L1 * self.L2 * c2
        M22 = self.m2 * self.L2**2

        M = np.array([[M11, M12], [M12, M22]])

        # Coriolis vector
        s2 = np.sin(q2)
        c1 = -self.m2 * self.L1 * self.L2 * s2 * (2 * q1_dot * q2_dot + q2_dot**2)
        c2 = self.m2 * self.L1 * self.L2 * s2 * q1_dot**2
        
        C = np.array([c1,c2])

        # Gravity vector
        G1 = self.g * ((self.m1 + self.m2) * self.L1 * c1 + self.m2 * self.L2 * c12)
        G2 = self.g * self.m2 * self.L2 * c12

        G = np.array([G1, G2])

        return M, C, G
    
    def _compute_distributed_mass_dynamics(self, q, q_dot):
        # Parameters
        l1 = self.L1
        l2 = self.L2
        m1 = self.m1
        m2 = self.m2
        g = self.g
        d = self.link_width

        # Inertia
        I1 = m1 * (l1**2 + d**2) / 12
        I2 = m2 * (l2**2 + d**2) / 12

        # State Variables 
        q1 = q[0]
        q2 = q[1]
        dq1 = q_dot[0]
        dq2 = q_dot[1]
        dx = 0
        dy = 0

        d11 = I1 + I2 + (m2*((l2*np.cos(q1 + q2) + 2*l1*np.cos(q1))*((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1)) + \
            (l2*np.sin(q1 + q2) + 2*l1*np.sin(q1))*((l2*np.sin(q1 + q2))/2 + l1*np.sin(q1))))/2 + \
            (m1*((l1**2*np.cos(q1)**2)/2 + (l1**2*np.sin(q1)**2)/2))/2

        d12 = I2 + (m2*(l2*np.sin(q1 + q2)*((l2*np.sin(q1 + q2))/2 + l1*np.sin(q1)) + \
            l2*np.cos(q1 + q2)*((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1))))/2

        d22 = I2 + (m2*((l2**2*np.cos(q1 + q2)**2)/2 + (l2**2*np.sin(q1 + q2)**2)/2))/2

        # Assemble the 2x2 M matrix
        M = np.array([
            [d11, d12],
            [d12, d22]  
        ])

        cg1 =(m2*((dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2)*(2*dq1*l1*np.sin(q1) - 2*dx + l2*np.sin(q1 + q2)*(dq1 + dq2)) - \
            (dq1*l1*np.sin(q1) + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)*(2*dy + 2*dq1*l1*np.cos(q1) + l2*np.cos(q1 + q2)*(dq1 + dq2)) - \
            ((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1))*(l2*np.sin(q1 + q2)*(dq1 + dq2)**2 + 2*dq1**2*l1*np.sin(q1)) + \
            ((l2*np.sin(q1 + q2))/2 + l1*np.sin(q1))*(2*dq1**2*l1*np.cos(q1) + l2*np.cos(q1 + q2)*(dq1 + dq2)**2)))/2 - \
            (m2*((dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2)*(2*dq1*l1*np.sin(q1) - 2*dx + l2*np.sin(q1 + q2)*(dq1 + dq2)) - \
            (dq1*l1*np.sin(q1) + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)*(2*dy + 2*dq1*l1*np.cos(q1) + l2*np.cos(q1 + q2)*(dq1 + dq2))))/2 + \
            g*m2*((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1)) + (g*l1*m1*np.cos(q1))/2

        cg2 = (m2*(l2*np.sin(q1 + q2)*(dq1**2*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2)**2)/2) - \
            l2*np.cos(q1 + q2)*((l2*np.sin(q1 + q2)*(dq1 + dq2)**2)/2 + dq1**2*l1*np.sin(q1)) - \
            l2*np.sin(q1 + q2)*(dq1 + dq2)*(dy + dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2) + \
            l2*np.cos(q1 + q2)*(dq1 + dq2)*(dq1*l1*np.sin(q1) - dx + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)))/2 + \
            (m2*(l2*np.sin(q1 + q2)*(dq1 + dq2)*(dy + dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2) - \
            l2*np.cos(q1 + q2)*(dq1 + dq2)*(dq1*l1*np.sin(q1) - dx + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)))/2 + \
            (g*l2*m2*np.cos(q1 + q2))/2

        # Assemble the 2-element CG vector
        CG = np.array([cg1, cg2])

        return M, CG

    def forward_dynamics(self, q, q_dot, tau):
        if self.method == "point":
            M,C,G =  self._compute_point_mass_dynamics(q, q_dot)
            q_ddot = np.linalg.solve(M, tau - C - G)
            return M, C, G, q_ddot
        
        elif self.method == "distributed":
            M,CG = self._compute_distributed_mass_dynamics(q, q_dot)
            q_ddot = np.linalg.solve(M, tau - CG)
            return M,CG, q_ddot
        



    

