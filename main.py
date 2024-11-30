#!/usr/bin/env python
# coding: utf-8

# Preparing Data For Training (Rope Data)

# In[1]:


import numpy as np


# In[2]:


#Parmeters
num_masses = 4
dt = 0.01 
k = 1.0
mass = 1.0

num_features = 2  # Position and velocity
koopman_dim = 3  # Size of the Koopman embedding


# In[3]:


positions = np.random.rand(num_masses, 1)
velocities = np.zeros((num_masses, 1))


# In[6]:


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

# In[20]:


import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[136]:


#Create a Graph Representationz
def create_graph(pos, val, num_masses):
    nodes = torch.tensor([pos, val], dtype = torch.float).T
    edges = torch.tensor([[i, i+1] for i in range(num_masses - 1)], dtype = torch.long).T
    return Data(x=nodes, edge_index =edges)


# In[138]:


dataset = [create_graph(pos, val, num_masses) for pos, val in data]


# In[140]:


#GNN Model
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# In[142]:


class KoopmanModel(torch.nn.Module):
    def __init__(self, input_dim, koopman_dim):
        super(KoopmanModel, self).__init__()
        self.encoder = SimpleGNN(input_dim, koopman_dim)
        self.koopman_matrix = torch.nn.Parameter(torch.eye(koopman_dim))
        self.decoder = SimpleGNN(koopman_dim, input_dim)
    def forward(self, data):
        koopman_space = self.encoder(data)
        next_koopman_space = koopman_space @ self.koopman_matrix
        next_state = self.decoder(next_koopman_space)
        return next_state


# In[144]:


from torch.optim import Adam

def train_model(model, dataset, epochs=(100)):
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0;
        for pos, val in dataset:
            data = create_graph(pos, val, num_masses)
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)

            optimizer.zero_grad()

            prediction = model(data)

            loss = loss_fn(prediction, data.x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss {total_loss/len(dataset)}")


# In[148]:


dataset = [(pos, val) for pos, val in data]
model = KoopmanModel(input_dim=1,koopman_dim = 3).to(device)
train_model(model, dataset, epochs=100)


# In[ ]:




