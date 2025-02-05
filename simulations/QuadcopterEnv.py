import numpy as np
class QuadcopterEnv:
    def __init__(self):
        # Physical parameters
        self.m = 0.5  # mass in kg
        self.g = 9.81  # gravity
        self.l = 0.25  # arm length in meters
        self.Ixx = 5e-3  # moment of inertia
        self.Iyy = 5e-3
        self.Izz = 1e-2
        self.k = 3e-6   # thrust coefficient
        self.b = 1e-7   # drag coefficient
        self.kd = 0.25  # drag coefficient
        
        # Inertia matrix
        self.I = np.array([[self.Ixx, 0, 0],
                          [0, self.Iyy, 0],
                          [0, 0, self.Izz]])
        self.I_inv = np.linalg.inv(self.I)
        
    def rotation_matrix(self, angles):
        """Compute rotation matrix from euler angles (phi, theta, psi)"""
        phi, theta, psi = angles
        
        # Individual rotation matrices
        R_x = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])
        
        return R_z @ R_y @ R_x
    
    def compute_forces_moments(self, state, inputs):
        """Compute forces and moments from motor inputs"""
        # Motor angular velocities (squared)
        w1, w2, w3, w4 = inputs
        
        # Total thrust
        thrust = self.k * sum(inputs)
        
        # Moments
        tau_phi = self.l * self.k * (w1 - w3)
        tau_theta = self.l * self.k * (w2 - w4)
        tau_psi = self.b * (w1 - w2 + w3 - w4)
        
        return thrust, np.array([tau_phi, tau_theta, tau_psi])
    
    def dynamics(self, state, inputs):
        """Compute state derivatives"""
        # Unpack state
        # [x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot]
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        omega = state[9:12]
        
        # Compute rotation matrix
        R = self.rotation_matrix(angles)
        
        # Compute forces and moments
        thrust, moments = self.compute_forces_moments(state, inputs)
        
        # Linear accelerations
        gravity = np.array([0, 0, -self.g])
        thrust_body = np.array([0, 0, thrust/self.m])
        acc = gravity + R @ thrust_body - self.kd/self.m * vel
        
        # Angular accelerations
        omega_skew = np.array([[0, -omega[2], omega[1]],
                              [omega[2], 0, -omega[0]],
                              [-omega[1], omega[0], 0]])
        angular_acc = self.I_inv @ (moments - omega_skew @ self.I @ omega)
        
        return np.concatenate([vel, acc, omega, angular_acc])
    def wrap_angle(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def rk4_step(self, state, inputs, dt):
        k1 = self.dynamics(state, inputs)
        k2 = self.dynamics(state + dt/2 * k1, inputs)
        k3 = self.dynamics(state + dt/2 * k2, inputs)
        k4 = self.dynamics(state + dt * k3, inputs)
        new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        new_state[6] = self.wrap_angle(new_state[6])
        new_state[7] = self.wrap_angle(new_state[7])
        new_state[8] = self.wrap_angle(new_state[8])
        return new_state
    
    def simulate(self, initial_state, inputs_sequence, dt=0.01, steps=1000):
        """Simulate the quadcopter for multiple steps"""
        state = initial_state
        trajectory = [state]
        
        for t in range(steps):
            inputs = inputs_sequence[t] if t < len(inputs_sequence) else inputs_sequence[-1]
            state = self.rk4_step(state, inputs, dt)
            trajectory.append(state)
            
        return np.array(trajectory)