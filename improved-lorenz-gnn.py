import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam

def simulate_lorenz(initial_state, steps=500, dt=0.01, 
                    sigma=10.0, rho=28.0, beta=8.0/3.0):
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
        # Runge-Kutta 4th order integration for better accuracy
        k1 = lorenz_system(state)
        k2 = lorenz_system(state + 0.5 * dt * k1)
        k3 = lorenz_system(state + 0.5 * dt * k2)
        k4 = lorenz_system(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return np.array(trajectory)

def create_sequence_data(trajectory):
    """Create sequences for training with next-step prediction"""
    X, y = [], []
    for i in range(len(trajectory) - 1):
        X.append(trajectory[i])
        y.append(trajectory[i+1])
    return np.array(X), np.array(y)

class ImprovedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedGNN, self).__init__()
        # Add fully connected layers for comparison
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # GNN layers (with fully connected graph)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def create_full_graph(self, x):
        """Create a fully connected graph"""
        n = x.size(0)
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n), n),
            torch.tile(torch.arange(n), (n,))
        ]).to(x.device)
        return edge_index

    def forward(self, x):
        # Ensure input is a tensor
        x = x.float()
        
        # Create fully connected graph
        edge_index = self.create_full_graph(x)
        
        # GNN path
        gnn_x = self.conv1(x, edge_index).relu()
        gnn_x = self.norm1(gnn_x)
        gnn_x = self.conv2(gnn_x, edge_index)
        gnn_x = self.norm2(gnn_x)
        
        # Fully connected path
        fc_x = self.fc1(x).relu()
        fc_x = self.fc2(fc_x).relu()
        fc_x = self.fc3(fc_x)
        
        # Combine GNN and FC outputs
        return (gnn_x + fc_x) / 2

def train_improved_model(X, y, epochs=100, learning_rate=0.001):
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Initialize model
    model = ImprovedGNN(input_dim=3, hidden_dim=64, output_dim=3)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_tensor)
        
        # Compute loss
        loss = criterion(predictions, y_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy (within a threshold)
        threshold = 0.1  # 10% tolerance
        accuracy = torch.mean((torch.abs(predictions - y_tensor) < threshold).float()) * 100
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}%")
    
    return model

# Example usage
initial_state = [1.0, 1.0, 1.0]
trajectory = simulate_lorenz(initial_state)
X, y = create_sequence_data(trajectory)

# Train the model
trained_model = train_improved_model(X, y)
