#!/usr/bin/env python
# coding: utf-8

# Preparing Data For Training (Rope Data)

# In[1]:


import numpy as np


# In[3]:


#Parmeters
num_masses = 4
dt = 0.01 
k = 1.0
mass = 1.0

num_features = 2  # Position and velocity
koopman_dim = 3  # Size of the Koopman embedding


# In[4]:


positions = np.random.rand(num_masses, 1)
velocities = np.zeros((num_masses, 1))


# In[7]:


#Simulate System And Store Data

def simulate(positions, velocities, steps=100):
    trajectory = []
    for _ in range(steps):
        forces = np.zeros_like(positions)
        for i in range(1, num_masses -1 ):
            #Hooke's Law
            left_force = -k*(positions[i] - positions[i - 1])
            right_force = -k * (positions[i] - positions[i+1])
            forces[i] = (left_force + right_force)
        accelerations = forces / mass
        velocities += accelerations * dt
        positions += velocities * dt
        trajectory.append((positions.copy(), velocities.copy()))
    return trajectory

data = simulate(positions, velocities)


# Building A Graph Neural Network(GNN)

# In[10]:


import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[11]:


#Create a Graph Representationz
def create_graph(pos, val, num_masses):   
    pos = torch.tensor(pos, dtype = torch.float).squeeze()
    val = torch.tensor(val, dtype = torch.float).squeeze()
    # if(pos.size(0) != val.size(0)):
    #     raise ValueError(f"Incompatible size pos={pos.size(0)}, val= {val.size(0)}")
    nodes = torch.stack([pos, val], dim=1)
    edges = torch.tensor([[i, i+1] for i in range(num_masses - 1)], dtype = torch.long).T
    return Data(x=nodes, edge_index =edges)


# In[12]:


#GNN Model
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    def forward(self, data):
        #print("Type of data in SimpleGNN:", type(data))
        x, edge_index = data.x, data.edge_index
        #print("x:", x.shape, "edge_index:", edge_index.shape)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# In[13]:


class KoopmanModel(torch.nn.Module):
    def __init__(self, input_dim, koopman_dim):
        super(KoopmanModel, self).__init__()
        self.encoder = SimpleGNN(input_dim, koopman_dim)
        self.koopman_matrix = torch.nn.Parameter(torch.eye(koopman_dim).to(device))
        self.decoder = SimpleGNN(koopman_dim, input_dim)
    def forward(self, data):
        #print("Type of data in KoopmanModel:", type(data))
        koopman_space = self.encoder(data)
        #print("koopman_space shape:", koopman_space.shape)
        next_koopman_space = koopman_space @ self.koopman_matrix
        new_data = Data(x=next_koopman_space, edge_index=data.edge_index)
        next_state = self.decoder(new_data)
        #print("next_state shape:", next_state.shape)
        return next_state


# In[79]:


from torch.optim import Adam

def train_model(model, dataset, epochs=(10)):
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    model = model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0 
        for data in dataset:
            data = data.to(device)
            #print(type(data))

            optimizer.zero_grad()

            prediction = model(data)

            loss = loss_fn(prediction, data.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            threshold = 0.15
            correct += (torch.abs(prediction - data.x) < threshold).sum().item()
            total_samples += data.x.numel()  
        accuracy = (correct / total_samples) * 100 
        print(f"Epoch {epoch + 1}, Loss {total_loss/len(dataset)}, Accuracy{accuracy:.2f}")


# In[81]:


dataset = [create_graph(pos, val, num_masses) for pos, val in data]
print(type(dataset[0]))
print(dataset[0].x)


# In[83]:


model = KoopmanModel(input_dim=2, koopman_dim=3).to(device)


# In[85]:


train_model(model, dataset, epochs=10)


# In[ ]:


import os


# In[28]:


save_folder = "spring-koopman-models"
os.makedirs(save_folder, exist_ok = True)


# In[30]:


save_path = os.path.join(save_folder, "spring-koopman-model-2.0.pth")
torch.save(model.state_dict(), save_path)

print(f"Model saved to {save_path}")


# In[34]:


model = KoopmanModel(input_dim=2, koopman_dim=3).to(device)
model.load_state_dict(torch.load(save_path, weights_only=True))
model.eval()


# In[ ]:





# In[ ]:




