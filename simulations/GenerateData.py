import numpy as np
import torch
import os
from torch_geometric.data import Data
from QuadcopterEnv import QuadcopterEnv
from PIDController import PIDController

def generate_pid_trajectory(env, steps=1000, dt=0.01):
    pid_x = PIDController(kp=0.3, ki=0.04, kd=0.3)
    pid_y = PIDController(kp=0.3, ki=0.04, kd=0.3)
    pid_z = PIDController(kp=0.4, ki=0.05, kd=1.2) 

    pid_phi = PIDController(kp=10.0, ki=0.05, kd=0.5)
    pid_theta = PIDController(kp=10.0, ki=0.05, kd=0.5)
    pid_psi = PIDController(kp=10.0, ki=0.05, kd=0.5)

    desired_pos = np.array([
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0.1, 1)  # 1 because above ground
    ])
    desired_yaw = 0.0  # Make it easier
    current_state = np.zeros(12)
    current_state[2] = np.random.uniform(0.1, 0.5)

    trajectory = [current_state]
    inputs_sequence = []

    for _ in range(steps):
        x, y, z = current_state[0:3]
        vx, vy, vz = current_state[3:6]
        phi, theta, psi = current_state[6:9]

        # Z Error and Control
        error_z = desired_pos[2] - z
        acc_z_desired = pid_z.compute(error_z, dt)
        thrust = env.m * (acc_z_desired + env.g + (env.kd / env.m) * vz)

        # X, Y Error and Control
        error_x = desired_pos[0] - x
        acc_x_desired = pid_x.compute(error_x, dt)

        error_y = desired_pos[1] - y
        acc_y_desired = pid_y.compute(error_y, dt)

        thrust = max(thrust, 1e-6)  # Prevent division by 0
        desired_theta = (env.m / thrust) * acc_x_desired
        desired_phi = - (env.m / thrust) * acc_y_desired

        # Attitude Control
        error_phi = desired_phi - phi
        tau_phi = pid_phi.compute(error_phi, dt)
        
        error_theta = desired_theta - theta
        tau_theta = pid_theta.compute(error_theta, dt)
        
        error_psi = desired_yaw - psi
        tau_psi = pid_psi.compute(error_psi, dt)

        # Motor Mixing
        T_total = thrust / env.k  # Thrust scaling
        a = tau_phi / (env.l * env.k)
        b_tau_theta = tau_theta / (env.l * env.k)
        c_tau_psi = tau_psi / env.b

        w1 = (T_total + 2 * a + c_tau_psi) / 4
        w2 = (T_total + 2 * b_tau_theta - c_tau_psi) / 4
        w3 = (T_total - 2 * a + c_tau_psi) / 4
        w4 = (T_total - 2 * b_tau_theta - c_tau_psi) / 4

        w1 = max(w1, 0)
        w2 = max(w2, 0)
        w3 = max(w3, 0)
        w4 = max(w4, 0)
        
        inputs = [w1, w2, w3, w4]
        inputs_sequence.append(inputs)  # Append the control inputs
        
        # Advance the simulation
        current_state = env.rk4_step(current_state, inputs, dt)
        trajectory.append(current_state)
    
    # Return after completing all steps
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
        # Optional: attach the desired position to the data
        #data.desired_pos = torch.tensor(desired_pos, dtype=torch.float)
        dataset.append(data)
    return dataset

data_save_folder = "data"
os.makedirs(data_save_folder, exist_ok=True)
data_save_path = os.path.join(data_save_folder, "pid_dataset_1_small.pth")

dataset = generate_training_dataset(num_trajectories=100, steps=1000, dt=0.01)
torch.save(dataset, data_save_path)
