import numpy as np
import torch
import os
from torch_geometric.data import Data
from QuadcopterEnv import QuadcopterEnv
from PIDController import PIDController

def generate_pid_trajectory(env, steps=1000, dt=0.01):
    pid_rate_roll = PIDController(kp=0.005, ki=0.0006, kd=0.002)
    pid_rate_pitch = PIDController(kp=0.005, ki=0.0006, kd=0.002)
    pid_rate_yaw = PIDController(kp=0.005, ki=0.0006, kd=0.002)

    pid_angle_phi   = PIDController(kp=0.2, ki=0.02, kd=0.005)
    pid_angle_theta = PIDController(kp=0.2, ki=0.02, kd=0.005)
    pid_angle_psi   = PIDController(kp=0.1, ki=0.01, kd=0.002)

    pid_pos_x = PIDController(kp=0.1, ki=0.0005, kd=0.7)
    pid_pos_y = PIDController(kp=0.1, ki=0.0005, kd=0.7)
    pid_pos_z = PIDController(kp=0.1, ki=0.0, kd=0.0)

    

    desired_pos = np.array([
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0.1, 0.5)  # 1 because above ground
    ])
    psi_desired = 0.0

    current_state = np.zeros(12)
    current_state[2] = np.random.uniform(0.1, 0.5)

    trajectory = [current_state]
    inputs_sequence = []

    thrust_history = []
    max_motor_speed = 2000 

    for step in range(steps):
        x, y, z = current_state[0:3]
        vx, vy, vz = current_state[3:6]
        phi, theta, psi = current_state[6:9]
        rate_roll, rate_pitch, rate_yaw = current_state[9:12]

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
        inputs_sequence.append(inputs)
        
        current_state = env.rk4_step(current_state, inputs, dt)
        trajectory.append(current_state)
    
    return trajectory, inputs_sequence, desired_pos

def create_graph_quad(trajectory, inputs_sequence):
    edge_index = []
    edge_attr = []

    for i in range(len(trajectory) - 1):
        edge_index.append([i, i + 1])
        edge_attr.append(inputs_sequence[i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    x = torch.tensor(np.array(trajectory), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def generate_training_dataset(num_trajectories=100, steps=1000, dt=0.01):
    env = QuadcopterEnv()
    dataset = []

    for _ in range(num_trajectories):
        trajectory, inputs_sequence, desired_pos = generate_pid_trajectory(env, steps, dt)
        data = create_graph_quad(trajectory, inputs_sequence)
        #data.desired_pos = torch.tensor(desired_pos, dtype=torch.float)
        dataset.append(data)
    return dataset

data_save_folder = "data"
os.makedirs(data_save_folder, exist_ok=True)
data_save_path = os.path.join(data_save_folder, "pid_dataset_2_medium_tiny.pth")

dataset = generate_training_dataset(num_trajectories=100, steps=2000, dt=0.01)
torch.save(dataset, data_save_path)
