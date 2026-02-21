import numpy as np

class TwoLinkArm:
    """
    Represents the physical properties and dynamics of a 2-link planar manipulator.
    
    This class can calculate dynamics using two models:
    1. 'point': Assumes mass is concentrated at the end of the links.
    2. 'distributed': Assumes mass is distributed along the links (as rectangles).
    """

    def __init__(self, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81, link_width=0.1,base_point = None,
                  method="point"):
        # --- Physical Parameters ---
        self.m1 = m1  # Mass of link 1
        self.m2 = m2  # Mass of link 2
        self.L1 = L1  # Length of link 1
        self.L2 = L2  # Length of link 2
        self.g = g    # Gravitational acceleration
        self.base_point = base_point # base point of instantiated model
        
        # Parameter for distributed mass model
        self.link_width = link_width 

        # --- Dynamics Method ---
        self.method = method.lower()  # 'point' or 'distributed'
        if self.method not in ["point", "distributed"]:
            raise ValueError("Selected method must be 'point' or 'distributed'.")


    def get_mass_matrix(self, q):
        """
        Computes the Mass Matrix M(q) for the point-mass model.
        """
        q1, q2 = q
        c2 = np.cos(q2)
        
        # Calculate terms of M(q)
        M11 = (self.m1 + self.m2) * self.L1**2 + self.m2 * self.L2**2 + 2 * self.m2 * self.L1 * self.L2 * c2
        M12 = self.m2 * self.L2**2 + self.m2 * self.L1 * self.L2 * c2
        M22 = self.m2 * self.L2**2
        
        return np.array([[M11, M12], 
                         [M12, M22]])

    def get_coriolis_vector(self, q, q_dot):
        """
        Computes the Coriolis/Centrifugal Vector C(q, q_dot) for the point-mass model.
        """
        q1, q2 = q
        q1_dot, q2_dot = q_dot
        s2 = np.sin(q2)
        
        # Calculate terms of C
        C1 = -self.m2 * self.L1 * self.L2 * s2 * (2 * q1_dot * q2_dot + q2_dot**2)
        C2 = self.m2 * self.L1 * self.L2 * s2 * q1_dot**2
        
        return np.array([C1, C2])

    def get_gravity_vector(self, q):
        """
        Computes the Gravity Vector G(q) for the point-mass model.
        """
        q1, q2 = q
        c1 = np.cos(q1)
        c12 = np.cos(q1 + q2)
        
        # Calculate terms of G
        G1 = self.g * ((self.m1 + self.m2) * self.L1 * c1 + self.m2 * self.L2 * c12)
        G2 = self.g * self.m2 * self.L2 * c12
        
        return np.array([G1, G2])

    def _compute_point_mass_dynamics(self, q, q_dot):
        """Internal helper to compute M and CG for the point-mass model."""
        # Use helper methods for clean, reusable code
        M = self.get_mass_matrix(q)
        C = self.get_coriolis_vector(q, q_dot)
        G = self.get_gravity_vector(q)
        
        CG = C + G  # Combine Coriolis and Gravity
        
        return M, CG
    
    def _compute_distributed_mass_dynamics(self, q, q_dot):
        """Internal helper to compute M and CG for the distributed-mass model."""
        # Parameters
        l1, l2 = self.L1, self.L2
        m1, m2 = self.m1, self.m2
        g = self.g
        d = self.link_width  # Link width

        # Inertia 
        I1 = m1 * (l1**2 + d**2) / 12
        I2 = m2 * (l2**2 + d**2) / 12

        # State Variables
        q1, q2 = q
        dq1, dq2 = q_dot
        dx, dy = 0, 0  # Assuming fixed base (0 velocity)

        # Mass Matrix M(q)
        d11 = (I1 + I2 + (m2*((l2*np.cos(q1 + q2) + 2*l1*np.cos(q1))*((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1)) +
               (l2*np.sin(q1 + q2) + 2*l1*np.sin(q1))*((l2*np.sin(q1 + q2))/2 + l1*np.sin(q1))))/2 +
               (m1*((l1**2*np.cos(q1)**2)/2 + (l1**2*np.sin(q1)**2)/2))/2)

        d12 = (I2 + (m2*(l2*np.sin(q1 + q2)*((l2*np.sin(q1 + q2))/2 + l1*np.sin(q1)) +
               l2*np.cos(q1 + q2)*((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1))))/2)

        d22 = I2 + (m2*((l2**2*np.cos(q1 + q2)**2)/2 + (l2**2*np.sin(q1 + q2)**2)/2))/2

        M = np.array([
            [d11, d12],
            [d12, d22]  
        ])

        # Coriolis + Gravity Vector C(q, q_dot) + G(q)
        cg1 = ((m2*((dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2)*(2*dq1*l1*np.sin(q1) - 2*dx + l2*np.sin(q1 + q2)*(dq1 + dq2)) -
               (dq1*l1*np.sin(q1) + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)*(2*dy + 2*dq1*l1*np.cos(q1) + l2*np.cos(q1 + q2)*(dq1 + dq2)) -
               ((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1))*(l2*np.sin(q1 + q2)*(dq1 + dq2)**2 + 2*dq1**2*l1*np.sin(q1)) +
               ((l2*np.sin(q1 + q2))/2 + l1*np.sin(q1))*(2*dq1**2*l1*np.cos(q1) + l2*np.cos(q1 + q2)*(dq1 + dq2)**2)))/2 -
               (m2*((dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2)*(2*dq1*l1*np.sin(q1) - 2*dx + l2*np.sin(q1 + q2)*(dq1 + dq2)) -
               (dq1*l1*np.sin(q1) + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)*(2*dy + 2*dq1*l1*np.cos(q1) + l2*np.cos(q1 + q2)*(dq1 + dq2))))/2 +
               g*m2*((l2*np.cos(q1 + q2))/2 + l1*np.cos(q1)) + (g*l1*m1*np.cos(q1))/2)

        cg2 = ((m2*(l2*np.sin(q1 + q2)*(dq1**2*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2)**2)/2) -
               l2*np.cos(q1 + q2)*((l2*np.sin(q1 + q2)*(dq1 + dq2)**2)/2 + dq1**2*l1*np.sin(q1)) -
               l2*np.sin(q1 + q2)*(dq1 + dq2)*(dy + dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2) +
               l2*np.cos(q1 + q2)*(dq1 + dq2)*(dq1*l1*np.sin(q1) - dx + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)))/2 +
               (m2*(l2*np.sin(q1 + q2)*(dq1 + dq2)*(dy + dq1*l1*np.cos(q1) + (l2*np.cos(q1 + q2)*(dq1 + dq2))/2) -
               l2*np.cos(q1 + q2)*(dq1 + dq2)*(dq1*l1*np.sin(q1) - dx + (l2*np.sin(q1 + q2)*(dq1 + dq2))/2)))/2 +
               (g*l2*m2*np.cos(q1 + q2))/2)

        CG = np.array([cg1, cg2])

        return M, CG

    def forward_dynamics(self, q, q_dot, tau):
        """
        Computes the forward dynamics of the arm.
        Given q, q_dot, and tau, calculates q_ddot.
        """
        # Select the correct dynamics calculation based on the method
        if self.method == "point":
            M, CG = self._compute_point_mass_dynamics(q, q_dot)
        elif self.method == "distributed":
            M, CG = self._compute_distributed_mass_dynamics(q, q_dot)
        
        # M * q_ddot = tau - CG
        q_ddot = np.linalg.solve(M, tau - CG)
        
        return M, CG, q_ddot

    def forward_kinematics(self, q1, q2):
        """
        Computes the end-effector position (x, y) from joint positions.
        """
        x = self.L1 * np.cos(q1) + self.L2 * np.cos(q1 + q2)
        y = self.L1 * np.sin(q1) + self.L2 * np.sin(q1 + q2)
        
        return x + self.base_point[0], y + self.base_point[1]

    def get_jacob(self, q1, q2):
        """
        Computes the end-effector Jacobian matrix J(q).
        """
        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s12 = np.sin(q1 + q2)
        c12 = np.cos(q1 + q2)
        
        J = np.array([
            [-self.L1*s1 - self.L2*s12, -self.L2*s12],
            [ self.L1*c1 + self.L2*c12,  self.L2*c12]
        ])
        return J
    
    def get_jacob_dot(self, q1, q2, q_dot):
        """
        Computes the derivative of the Jacobian matrix, J_dot(q, q_dot).
        """
        q1_dot, q2_dot = q_dot
        
        l1, l2 = self.L1, self.L2
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        q12 = q1 + q2
        c12 = np.cos(q12)
        s12 = np.sin(q12)
        dq_sum = q1_dot + q2_dot  # (q1_dot + q2_dot)
        
        # Calculate each term of the J_dot matrix
        Jdot_11 = -l1*c1*q1_dot - l2*c12*dq_sum
        Jdot_12 = -l2*c12*dq_sum
        Jdot_21 = -l1*s1*q1_dot - l2*s12*dq_sum
        Jdot_22 = -l2*s12*dq_sum
        
        Jdot = np.array([
            [Jdot_11, Jdot_12],
            [Jdot_21, Jdot_22]
        ])
        return Jdot