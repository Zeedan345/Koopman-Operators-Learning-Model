import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from model.KoopmanModel import AdvancedKoopmanModel
from simulations.QuadcopterEnv import QuadcopterEnv

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

def generate_random_trajectory(env, steps=500, dt=0.002):
    # Reduce initial state perturbations
    initial_state = np.zeros(12)
    initial_state[2] = np.random.uniform(5, 15)
    initial_state[3:6] = np.random.uniform(-0.1, 0.1, size=3) 
    initial_state += np.random.normal(0, 0.1, size=12)
    
    # Generate smoother control inputs
    hover_thrust = env.m * env.g / (4 * env.k)
    inputs = np.ones(4) * hover_thrust
    inputs_sequence = []
    current_inputs = inputs.copy()
    
    for _ in range(steps):
        # Add small, smooth changes to inputs
        current_inputs += np.random.normal(0, hover_thrust*0.01, size=4)
        current_inputs = np.clip(current_inputs, hover_thrust*0.5, hover_thrust*1.5)
        inputs_sequence.append(current_inputs.copy())
    trajectory = env.simulate(initial_state, inputs_sequence, dt, steps)
    return trajectory, inputs_sequence

def create_graph_quadcopter(trajectory, inputs_sequence):
    edge_index = []
    edge_attr = []
    
    for i in range(len(trajectory)-1):
        edge_index.append([i, i+1])
        edge_attr.append(inputs_sequence[i])  
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    
    x = torch.tensor(trajectory, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr) 

def generate_training_dataset(num_trajectories=100, steps=1000):
    """Generate multiple trajectories for training"""
    env = QuadcopterEnv()
    dataset = []
    
    for _ in range(num_trajectories):
        trajectory, inputs_sequence = generate_random_trajectory(env, steps)
        data = create_graph_quadcopter(trajectory, inputs_sequence)
        dataset.append(data)
    return dataset
def advanced_loss(model, decoded_ae, decoded_rollout, koopman_states, data, lambda1=0.1, lambda2=0.1):
    Lae = F.mse_loss(decoded_ae, data.x)
    Lpred = F.mse_loss(decoded_rollout,data.x[1:])
    Lmetric = model.metric_loss(koopman_states, data.x)
    total_loss = Lae + lambda1 * Lpred + lambda2 * Lmetric
    return total_loss, Lae, Lpred, Lmetric

def train_advanced_model(model, dataset, epochs=10, lr=0.001, lambda1=0.1, lambda2=0.1):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_ae = 0
        epoch_pred = 0
        epoch_metric = 0
        
        for data in dataset:
            optimizer.zero_grad()
            decoded_ae, decoded_rollout, koopman_states = model(data)
            total_loss, Lae, Lpred, Lmetric = advanced_loss(
                model, decoded_ae, decoded_rollout, koopman_states, data, lambda1, lambda2
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ae += Lae.item()
            epoch_pred += Lpred.item()
            epoch_metric += Lmetric.item()
        
        avg_loss = epoch_loss / len(dataset)
        scheduler.step(avg_loss)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f} (AE: {epoch_ae/len(dataset):.4f}, Pred: {epoch_pred/len(dataset):.4f}, Metric: {epoch_metric/len(dataset):.4f})")
    
    return train_losses

dataset = generate_training_dataset(num_trajectories=100)
model = AdvancedKoopmanModel(input_dim=12, koopman_dim=32)
losses = train_advanced_model(model, dataset, lambda1=1.0, lambda2=0.1)
save_folder = "quadcopter-koopman-models"
os.makedirs(save_folder, exist_ok = True)
save_path = os.path.join(save_folder, "quadcopter-koopman-model-5.0.pth")
torch.save(model.state_dict(), save_path)

print(f"Model saved to {save_path}")
