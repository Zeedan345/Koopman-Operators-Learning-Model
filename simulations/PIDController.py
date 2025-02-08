import numpy as np
import matplotlib.pyplot as plt
from QuadcopterEnv import QuadcopterEnv

class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=100, output_limit = 10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.integral_limit = integral_limit
        self.output_limit = output_limit

    def compute(self, error, dt):
        p = self.kp*error #proportional: k_prop * pitch_err
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = p + i + d
        return np.clip(output, -self.output_limit, self.output_limit)

# env = QuadcopterEnv()

# pid_x = PIDController(kp=0.3, ki=0.04, kd=0.3)
# pid_y = PIDController(kp=0.3, ki=0.04, kd=0.3)
# pid_z = PIDController(kp=0.4, ki=0.05, kd=1.2) 

# pid_phi = PIDController(kp=10.0, ki=0.05, kd=0.5)
# pid_theta = PIDController(kp=10.0, ki=0.05, kd=0.5)
# pid_psi = PIDController(kp=10.0, ki=0.05, kd=0.5)

# desired_pos = np.array([3, 9, 4])
# desired_yaw = 0.0
# dt = 0.01
# steps = 4000
# current_state = np.zeros(12)
# trajectory = [current_state]
# 0
# for _ in range(steps):
#     x, y, z = current_state[0:3]
#     vx, vy, vz = current_state[3:6]
#     phi, theta, psi = current_state[6:9]

#     # Attitude control Z
#     error_z = desired_pos[2] - z
#     acc_z_desired = pid_z.compute(error_z, dt)
#     thrust = env.m * (acc_z_desired + env.g + (env.kd / env.m) * vz) ##

#     # Horizontal Pos Contorl x,y
#     error_x = desired_pos[0] - x
#     acc_x_desired = pid_x.compute(error_x, dt)

#     error_y = desired_pos[1] - y
#     acc_y_desired = pid_y.compute(error_y, dt)

#     #convert acc to angles
#     thrust = max(thrust, 1e-6)  # Prevent division by zero
#     desired_theta = (env.m / thrust) * acc_x_desired
#     desired_phi = - (env.m / thrust) * acc_y_desired

#     #Attitude Control
#     error_phi = desired_phi - phi
#     tau_phi = pid_phi.compute(error_phi, dt)
    
#     error_theta = desired_theta - theta
#     tau_theta = pid_theta.compute(error_theta, dt)
    
#     error_psi = desired_yaw - psi
#     tau_psi = pid_psi.compute(error_psi, dt)

#     #Motor Mixing
#     T_total = thrust / env.k  # Thrust scaling
#     a = tau_phi / (env.l * env.k)
#     b_tau_theta = tau_theta / (env.l * env.k)
#     c_tau_psi = tau_psi / env.b

#     w1 = (T_total + 2*a + c_tau_psi) / 4
#     w2 = (T_total + 2*b_tau_theta - c_tau_psi) / 4
#     w3 = (T_total - 2*a + c_tau_psi) / 4
#     w4 = (T_total - 2*b_tau_theta - c_tau_psi) / 4

#     w1 = max(w1, 0)
#     w2 = max(w2, 0)
#     w3 = max(w3, 0)
#     w4 = max(w4, 0)
    
#     inputs = [w1, w2, w3, w4]
#     current_state = env.rk4_step(current_state, inputs, dt)
#     trajectory.append(current_state)

# trajectory = np.array(trajectory)

# # 2. Visualization Code
# plt.figure(figsize=(12, 8))

# # Position plot
# plt.subplot(2, 2, 1)
# plt.plot(trajectory[:, 0], label='X')
# plt.plot(trajectory[:, 1], label='Y')
# plt.plot(trajectory[:, 2], label='Z')
# plt.plot([0, steps], [desired_pos[0], desired_pos[0]], 'k--', label='Desired X')
# plt.title('Position vs Time')
# plt.xlabel('Time steps')
# plt.ylabel('Position (m)')
# plt.legend()

# # Velocity plot
# plt.subplot(2, 2, 2)
# plt.plot(trajectory[:, 3], label='VX')
# plt.plot(trajectory[:, 4], label='VY')
# plt.plot(trajectory[:, 5], label='VZ')
# plt.title('Velocity vs Time')
# plt.xlabel('Time steps')
# plt.ylabel('Velocity (m/s)')
# plt.legend()

# # Angles plot
# plt.subplot(2, 2, 3)
# plt.plot(np.degrees(trajectory[:, 6]), label='Roll (φ)')
# plt.plot(np.degrees(trajectory[:, 7]), label='Pitch (θ)')
# plt.plot(np.degrees(trajectory[:, 8]), label='Yaw (ψ)')
# plt.title('Euler Angles vs Time')
# plt.xlabel('Time steps')
# plt.ylabel('Degrees')
# plt.legend()

# # Altitude zoom
# plt.subplot(2, 2, 4)
# plt.plot(trajectory[:, 2], label='Actual')
# plt.plot([0, steps], [1, 1], 'r--', label='Desired')
# plt.title('Altitude Control')
# plt.xlabel('Time steps')
# plt.ylabel('Z Position (m)')
# plt.legend()

# plt.tight_layout()
# plt.show()