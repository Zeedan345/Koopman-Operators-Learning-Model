import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from model.KoopmanModel import AdvancedKoopmanModel
from mpl_toolkits.mplot3d import Axes3D 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
dataset = torch.load("./data/pid_dataset_2_medium.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")
# Set the model to evaluation mode
model = AdvancedKoopmanModel(input_dim=12, koopman_dim=64, num_objects=4, h=4).to(device)
model.load_state_dict(torch.load("./quadcopter-koopman-models/quadcopter-koopman-model-02-v1.2.pth", weights_only=True))
model.eval()

# Choose a sample from the dataset; here we use the first sample
sample_index = 0
sample_data = dataset[sample_index].to(device)

with torch.no_grad():
    # Forward pass: get the autoencoder reconstruction and rollout prediction
    decoded_ae, decoded_rollout, koopman_states = model(sample_data)

# Move tensors to CPU and convert to numpy arrays
true_states = sample_data.x.cpu().numpy()    
recon_states = decoded_ae.cpu().numpy()       
rollout_states = decoded_rollout.cpu().numpy()


T = true_states.shape[0]
dt = 0.01 
time = np.arange(T) * dt

# Create plots for the first 3 state dimensions (x, y, z positions)
plt.figure(figsize=(12, 8))
state_labels = ['x (position)', 'y (position)', 'z (position)']

for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(time, true_states[:, i], label='Ground Truth', linewidth=2)
    plt.plot(time, recon_states[:, i], label='Autoencoder Recon.', linestyle='dashed')
    plt.plot(time, rollout_states[:, i], label='Koopman Rollout', linestyle='dotted')
    plt.title(f"State {i+1}: {state_labels[i]}")
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the ground truth trajectory (first 3 state dimensions: x, y, z)
ax.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2],
        label='Ground Truth', color='blue', linewidth=2)

# Plot the Koopman rollout predicted trajectory
ax.plot(rollout_states[:, 0], rollout_states[:, 1], rollout_states[:, 2],
        label='Koopman Rollout', color='red', linestyle='dotted', linewidth=2)

# Optionally, highlight the starting and ending points of the rollout
ax.scatter(rollout_states[0, 0], rollout_states[0, 1], rollout_states[0, 2],
           color='green', s=100, label='Start')
ax.scatter(rollout_states[-1, 0], rollout_states[-1, 1], rollout_states[-1, 2],
           color='magenta', s=100, label='End')

# Label the axes and add a title
ax.set_xlabel("X (position)")
ax.set_ylabel("Y (position)")
ax.set_zlabel("Z (position)")
ax.set_title("3D Trajectory: Ground Truth vs. Koopman Rollout")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


angles_labels = ['Roll', 'Pitch', 'Yaw']
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 1, i+1)
    # Angles are assumed to be at indices 6, 7, and 8
    plt.plot(time, true_states[:, 6+i], label='Ground Truth', linewidth=2)
    plt.plot(time, recon_states[:, 6+i], label='AE Reconstruction', linestyle='dashed')
    plt.plot(time, rollout_states[:, 6+i], label='Koopman Rollout', linestyle='dotted')
    plt.title(f"{angles_labels[i]} Angle")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# Plot linear velocities (v_x, v_y, v_z)
velocity_labels = ['v_x', 'v_y', 'v_z']
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 1, i+1)
    # Velocities are assumed to be at indices 3, 4, and 5
    plt.plot(time, true_states[:, 3+i], label='Ground Truth', linewidth=2)
    plt.plot(time, recon_states[:, 3+i], label='AE Reconstruction', linestyle='dashed')
    plt.plot(time, rollout_states[:, 3+i], label='Koopman Rollout', linestyle='dotted')
    plt.title(f"Velocity: {velocity_labels[i]}")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [units/s]")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
