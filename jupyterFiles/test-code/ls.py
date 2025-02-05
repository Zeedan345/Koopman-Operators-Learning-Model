#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")


# In[4]:


def simulate_lorenz(initial_state, steps = 500, dt= 0.01, sigma = 10.0, rho = 28.0, beta = 8.0/3.0):
    def lorenz_system(state):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return np.array([dxdt, dydt, dzdt])
    state = np.array(initial_state)
    trajectory = []
    for _ in range(steps):
        trajectory.append(state.copy())
        k1 = lorenz_system(state)
        k2 = lorenz_system(state + 0.5 * dt * k1)
        k3 = lorenz_system(state + 0.5 * dt * k2)
        k4 = lorenz_system(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return np.array(trajectory)
        


# In[5]:


def create_graph_lorenz(trajectory):
    edge_index = []
    for i in range(len(trajectory)-1):
        edge_index.append([i, i+1])
    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()
    x = torch.tensor(trajectory, dtype = torch.float)

    data = Data(x=x, edge_index = edge_index)
    return data


# In[6]:


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def create_full_graph(self, x):
        n = x.size(0)
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n), n),
            torch.tile(torch.arange(n), (n,))
        ]).to(x.device)
        return edge_index
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        edge_index = self.create_full_graph(x)
        
        # GNN path
        gnn_x = self.conv1(x, edge_index).relu()
        gnn_x = self.norm1(gnn_x)
        gnn_x = self.conv2(gnn_x, edge_index)
        gnn_x = self.norm2(gnn_x)

        # FC path - fixed the layer usage
        fc_x = self.fc1(x).relu()
        fc_x = self.fc2(fc_x).relu() 
        fc_x = self.fc3(fc_x)         

        return (gnn_x + fc_x)/2


# In[7]:


class KoopmanModel(torch.nn.Module):
    def __init__(self, input_dim, koopman_dim):
        super(KoopmanModel, self).__init__()
        self.encoder = GNN(input_dim, koopman_dim, koopman_dim)
        self.koopman_matrix = torch.nn.Parameter(torch.eye(koopman_dim))
        self.decoder = GNN(koopman_dim, koopman_dim, input_dim)

    def forward(self, data):
        koopman_space = self.encoder(data)
        next_koopman_space = koopman_space @ self.koopman_matrix
        new_data = Data(x=next_koopman_space, edge_index = data.edge_index)
        new_state = self.decoder(new_data)
        return new_state


# In[8]:


def auto_encoding_loss(decoded, original_states):
    #return F.l1_loss(decoded, original_states)
    return F.mse_loss(decoded, original_states, reduction='mean')

def prediction_loss(model, koopman_space, data):
    T = data.x.size(0)
    all_predictions = []
    current_state = koopman_space[0].unsqueeze(0)

    for t in range(T):
        decoded_state = model.decoder(Data(x=current_state, edge_index=data.edge_index))
        all_predictions.append(decoded_state)
        if t < T - 1:
            current_state = (current_state @ model.koopman_matrix).detach().clone()
    all_predictions = torch.cat(all_predictions, dim=0)
    #return F.l1_loss(all_predictions, data.x)
    return F.mse_loss(all_predictions, data.x, reduction='mean')



def metric_loss(koopman_space, original_space):
    distances_koopman = torch.cdist(koopman_space, koopman_space, p=2)
    distances_original = torch.cdist(original_space, original_space, p=2)
    #return F.l1_loss(distances_koopman, distances_original)
    return F.mse_loss(distances_koopman, distances_original, reduction='mean')


def total_loss(model, data, lambda1=1.0, lambda2=1.0):
    koopman_space = model.encoder(data)
    decoded = model.decoder(Data(x=koopman_space, edge_index=data.edge_index))
    
    ae_loss = auto_encoding_loss(decoded, data.x)
    pred_loss = prediction_loss(model, koopman_space, data)
    m_loss = metric_loss(koopman_space, data.x)
    #print(f"AE Loss: {ae_loss}, Predicted Loss: Total Loss {m_loss}")

    return ae_loss  + lambda1 * pred_loss + lambda2 * m_loss 


# In[9]:


def train_model(model, dataset, epochs = (10),lambda1=1.0, lambda2=0.3, initial_lr = 0.007):
    optimizer = Adam(model.parameters(), lr = initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose = True
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=5,
    #     T_mult=2,
    #     eta_min=1e-5
    # )
    model = model.to(device)
    train_losses = []
    #learning_rates = []
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for data in dataset:
            data = data.to(device)
            optimizer.zero_grad()
            #prediction = model(data)
            loss = total_loss(model, data,  lambda1=lambda1, lambda2=lambda2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataset)
        scheduler.step(avg_loss)
        train_losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        #learning_rates.append(current_lr)
        #scheduler.step()
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        print(f"Epoch {epoch + 1}, Loss {avg_loss:.4f}, LR: {current_lr:.6f}")

        #For Reduce LR
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10: 
                print("Early stopping triggered")
                break
    plt.plot(range(epochs), train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.show()


# In[10]:


def normalize_lorenz_trajectory(trajectory):
    scale_factors = np.array([20.0, 30.0, 50.0])
    normalized_trajectory = trajectory / scale_factors
    return normalized_trajectory


initial_state = [1.0, 0.0, 0.0]
lorenz_trajectory = simulate_lorenz(initial_state)
normalized_trajectory = normalize_lorenz_trajectory(lorenz_trajectory)
dataset = [
    create_graph_lorenz(
        normalize_lorenz_trajectory(
            simulate_lorenz([
                np.random.normal(1.0, 0.1), 
                np.random.normal(0.0, 0.1), 
                np.random.normal(0.0, 0.1)
            ])
        )
    ) 
    for _ in range(100)
]


# In[11]:


model = KoopmanModel(input_dim = 3, koopman_dim = 16).to(device)


# In[147]:


# train_model(model, dataset, epochs = 10)


# In[12]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# In[13]:


save_folder = "lorenz-koopman-models"
os.makedirs(save_folder, exist_ok = True)


# In[14]:


save_path = os.path.join(save_folder, "lorenz-koopman-model-1.0.pth")


# In[150]:


# torch.save(model.state_dict(), save_path)

print(f"Model saved to {save_path}")


# In[27]:


model = KoopmanModel(input_dim=3, koopman_dim=16).to(device)
model.load_state_dict(torch.load(save_path, weights_only=True))
model.eval()


# In[ ]:


def visualize_comparison(model, initial_state=[1.0, 0.0, 0.0], steps=200):  # Reduced steps
    # Generate actual trajectory
    actual_trajectory = normalize_lorenz_trajectory(simulate_lorenz(initial_state, steps=steps))
    
    # Convert to PyTorch and create graph
    data = create_graph_lorenz(actual_trajectory)
    data = data.to(device)
    
    with torch.no_grad():  # Ensure no gradients are computed
        koopman_space = model.encoder(data)
        predicted_states = []
        current_state = koopman_space[0].unsqueeze(0)
        
        # Only predict for specified number of steps
        for _ in range(steps):
            decoded_state = model.decoder(Data(x=current_state, edge_index=data.edge_index))
            predicted_states.append(decoded_state.cpu().numpy())
            current_state = (current_state @ model.koopman_matrix)
            
    # Clear some memory
    torch.cuda.empty_cache()
            
    predicted_trajectory = np.array(predicted_states).squeeze()
    
    # Denormalize
    scale_factors = np.array([20.0, 30.0, 50.0])
    actual_trajectory = actual_trajectory * scale_factors
    predicted_trajectory = predicted_trajectory * scale_factors
    
    # Plot
    fig = plt.figure(figsize=(10, 5))
    
    # Actual
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(actual_trajectory[:, 0], 
             actual_trajectory[:, 1], 
             actual_trajectory[:, 2], 
             'b-', linewidth=0.5)
    ax1.set_title('Actual')
    
    # Predicted
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(predicted_trajectory[:, 0], 
             predicted_trajectory[:, 1], 
             predicted_trajectory[:, 2], 
             'r-', linewidth=0.5)
    ax2.set_title('Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Clear the plot to free memory
    plt.close()

# Call with fewer steps
visualize_comparison(model, steps=200)


# In[ ]:




