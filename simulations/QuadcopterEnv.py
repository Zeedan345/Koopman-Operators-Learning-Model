import numpy as np

class QuadcopterEnv:
    def __init__(self):
        # Physical parameters
        self.m = 0.5  # mass in kg
        self.g = 9.81  # gravity in m/s^2
        self.l = 0.25  # arm length in meters
        self.Ixx = 5e-3  # moment of inertia about x-axis
        self.Iyy = 5e-3  # moment of inertia about y-axis
        self.Izz = 1e-2  # moment of inertia about z-axis
        self.k = 0.00122625  # thrust coefficient (force per squared radian speed)
        self.b = 1e-5  # drag coefficient (for yaw moment)
        self.kd = 0.25  # linear drag coefficient
        
        # Inertia matrix and its inverse
        self.I = np.array([[self.Ixx, 0, 0],
                           [0, self.Iyy, 0],
                           [0, 0, self.Izz]])
        self.I_inv = np.linalg.inv(self.I)
        
    def rotation_matrix(self, angles):
        phi, theta, psi = angles
        
        # Rotation about x-axis (roll)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        
        # Rotation about y-axis (pitch)
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        
        # Rotation about z-axis (yaw)
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x
    
    def compute_forces_moments(self, state, inputs):
        w1, w2, w3, w4 = inputs
        
        # Total thrust (force magnitude)
        thrust = self.k * (w1 + w2 + w3 + w4)
        
        # Moments computed as in the derivation:
        tau_phi = self.l * self.k * (w1 - w3)
        tau_theta = self.l * self.k * (w2 - w4)
        tau_psi = self.b * (w1 - w2 + w3 - w4)
        
        return thrust, np.array([tau_phi, tau_theta, tau_psi])
    
    def euler_angle_rates(self, angles, omega):

        phi, theta, _ = angles
        cos_theta = np.cos(theta) if np.abs(np.cos(theta)) > 1e-6 else 1e-6
        
        phi_dot = omega[0] + np.sin(phi) * np.tan(theta) * omega[1] + np.cos(phi) * np.tan(theta) * omega[2]
        theta_dot = np.cos(phi) * omega[1] - np.sin(phi) * omega[2]
        psi_dot = (np.sin(phi) / cos_theta) * omega[1] + (np.cos(phi) / cos_theta) * omega[2]
        
        return np.array([phi_dot, theta_dot, psi_dot])
    
    def dynamics(self, state, inputs):
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        omega = state[9:12]
        
        # Compute rotation matrix from body to inertial frame
        R = self.rotation_matrix(angles)
        
        # Compute thrust and moments from motor inputs
        thrust, moments = self.compute_forces_moments(state, inputs)
        
        gravity = np.array([0, 0, -self.g])
        thrust_body = np.array([0, 0, thrust])
        acc = gravity + (R @ thrust_body) / self.m - (self.kd / self.m) * vel
        

        omega_skew = np.array([[0, -omega[2], omega[1]],
                               [omega[2], 0, -omega[0]],
                               [-omega[1], omega[0], 0]])
        angular_acc = self.I_inv @ (moments - omega_skew @ self.I @ omega)
        

        euler_rates = self.euler_angle_rates(angles, omega)
        

        return np.concatenate([vel, acc, euler_rates, angular_acc])
    
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
        state = initial_state
        trajectory = [state]
        
        for t in range(steps):
            # If inputs_sequence is shorter than steps, repeat the last input.
            inputs = inputs_sequence[t] if t < len(inputs_sequence) else inputs_sequence[-1]
            state = self.rk4_step(state, inputs, dt)
            trajectory.append(state)
            
        return np.array(trajectory)
