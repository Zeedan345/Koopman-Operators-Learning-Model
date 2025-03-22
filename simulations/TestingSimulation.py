import numpy as np
import matplotlib.pyplot as plt
from QuadcopterEnv import QuadcopterEnv
from PIDController import PIDController



env = QuadcopterEnv()

pid_rate_roll = PIDController(kp=0.005, ki=0.0006, kd=0.002)
pid_rate_pitch = PIDController(kp=0.005, ki=0.0006, kd=0.002)
pid_rate_yaw = PIDController(kp=0.005, ki=0.0006, kd=0.002)

pid_angle_phi   = PIDController(kp=0.2, ki=0.02, kd=0.005)
pid_angle_theta = PIDController(kp=0.2, ki=0.02, kd=0.005)
pid_angle_psi   = PIDController(kp=0.1, ki=0.01, kd=0.002)

pid_pos_x = PIDController(kp=0.1, ki=0.0005, kd=0.7)
pid_pos_y = PIDController(kp=0.1, ki=0.0005, kd=0.7)
pid_pos_z = PIDController(kp=0.1, ki=0.0, kd=0.0)


# desired_pos = np.array([6, 7, 4])

# Making Array for the qaudcopter to follow
x = np.linspace(0, 20, 50)
y=x**0.5
desired_pos_array = np.column_stack((x, y))
z_const = 4
z_column = np.full((desired_pos_array.shape[0], 1), z_const)
desired_pos_array_3d = np.hstack((desired_pos_array, z_column))

waypoint_idx = 0
waypoint_threshold = 0.5
update_inteval = 5 #100/5 = 20hz

desired_rate = np.array([0.1, 0.1, 0.1])
desired_angle = np.array([0.2, 0.3, 0.4])
psi_desired = 0.0
dt = 0.01
steps = 6000
current_state = np.zeros(12)
trajectory = [current_state]

max_motor_speed = 2000 

for step in range(steps):
    x, y, z = current_state[0:3]
    vx, vy, vz = current_state[3:6]
    phi, theta, psi = current_state[6:9]
    rate_roll, rate_pitch, rate_yaw = current_state[9:12]

    if step % update_inteval == 0:
        current_waypoint = desired_pos_array_3d[waypoint_idx]
        distance_to_waypoint = np.linalg.norm(current_waypoint - np.array([x, y, z]))
        if distance_to_waypoint < waypoint_threshold and waypoint_idx < len(desired_pos_array_3d) - 1:
            waypoint_idx += 1
    desired_pos = desired_pos_array_3d[waypoint_idx]

    error_pos_x = desired_pos[0] - x
    error_pos_y = desired_pos[1] - y
    error_pos_z = desired_pos[2] - z

    desired_acc_x = pid_pos_x.compute(error_pos_x, dt)
    desired_acc_y = pid_pos_y.compute(error_pos_y, dt)
    desired_acc_z = pid_pos_z.compute(error_pos_z, dt)

    thrust_nominal = env.m * env.g
    desired_phi = (env.m/thrust_nominal) * (np.sin(psi) * desired_acc_x - np.cos(psi) * desired_acc_y)
    desired_theta = (env.m/thrust_nominal) * (np.cos(psi) * desired_acc_x + np.sin(psi) * desired_acc_y)
    desired_psi = psi_desired - psi

    error_phi = desired_phi - phi
    error_theta = desired_theta - theta
    error_psi = desired_psi - psi

    desired_roll_rate = pid_angle_phi.compute(error_phi, dt)
    desired_pitch_rate = pid_angle_theta.compute(error_theta, dt)
    desired_psi_rate = pid_angle_psi.compute(error_psi, dt)

    error_roll  = desired_roll_rate - rate_roll
    error_pitch = desired_pitch_rate - rate_pitch
    error_yaw   = desired_psi_rate - rate_yaw

    tau_phi = pid_rate_roll.compute(error_roll, dt)
    tau_theta = pid_rate_pitch.compute(error_pitch, dt)
    tau_psi = pid_rate_yaw.compute(error_yaw, dt)

    thrust = env.m * (env.g + desired_acc_z)

    T_total = thrust / env.k
    a = tau_phi / (env.l * env.k)
    b = tau_theta / (env.l * env.k)
    c = tau_psi / env.b

    raw_speeds = [
        (T_total + 2*a + c) / 4,
        (T_total + 2*b - c) / 4,
        (T_total - 2*a + c) / 4,
        (T_total - 2*b - c) / 4 
    ]


    current_time = step * dt
    for motor_idx, speed in enumerate(raw_speeds):
        if speed > max_motor_speed:
            print(f"[WARNING] Motor {motor_idx+1} exceeded MAX speed at {current_time:.2f}s: {speed:.2f} RPM")
        elif speed < 0:
            print(f"[WARNING] Motor {motor_idx+1} reversed at {current_time:.2f}s: {speed:.2f} RPM")


    w1 = np.clip(raw_speeds[0], 0, max_motor_speed)
    w2 = np.clip(raw_speeds[1], 0, max_motor_speed)
    w3 = np.clip(raw_speeds[2], 0, max_motor_speed)
    w4 = np.clip(raw_speeds[3], 0, max_motor_speed)

    inputs = [w1, w2, w3, w4]
    current_state = env.rk4_step(current_state, inputs, dt)

    trajectory.append(current_state.copy())


trajectory = np.array(trajectory)

time = np.arange(trajectory.shape[0]) * dt


state_labels = [
    "X Position (m)", "Y Position (m)", "Z Position (m)",
    "X Velocity (m/s)", "Y Velocity (m/s)", "Z Velocity (m/s)",
    "Roll (rad)", "Pitch (rad)", "Yaw (rad)",
    "Roll Rate (rad/s)", "Pitch Rate (rad/s)", "Yaw Rate (rad/s)"
]


fig, axs = plt.subplots(4, 3, figsize=(15, 12))
axs = axs.ravel()

for i in range(12):
    axs[i].plot(time, trajectory[:, i])
    axs[i].set_title(state_labels[i])
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel(state_labels[i])
    axs[i].grid(True)

plt.tight_layout()
plt.show()
