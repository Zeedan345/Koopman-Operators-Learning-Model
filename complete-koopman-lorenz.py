import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

class KoopmanOperatorNetwork(nn.Module):
    def __init__(self, state_dim, koopman_dim, observable_dim):
        super(KoopmanOperatorNetwork, self).__init__()
        
        # Encoder to lift the state to Koopman space
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, koopman_dim)
        )
        
        # Koopman operator (linear dynamics in lifted space)
        self.koopman_matrix = nn.Parameter(torch.randn(koopman_dim, koopman_dim))
        
        # Decoder to reconstruct original state from Koopman space
        self.decoder = nn.Sequential(
            nn.Linear(koopman_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        
        # Additional observable network (optional)
        self.observable_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, observable_dim)
        )

    def forward(self, x):
        # Encode to Koopman space
        koopman_state = self.encoder(x)
        
        # Evolve in Koopman space using linear operator
        next_koopman_state = F.linear(koopman_state, self.koopman_matrix)
        
        # Decode back to original state space
        predicted_next_state = self.decoder(next_koopman_state)
        
        return predicted_next_state

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
        # Runge-Kutta 4th order integration
        k1 = lorenz_system(state)
        k2 = lorenz_system(state + 0.5 * dt * k1)
        k3 = lorenz_system(state + 0.5 * dt * k2)
        k4 = lorenz_system(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return np.array(trajectory)

def prepare_sequence_data(trajectory, lookback=1):
    X, y = [], []
    for i in range(len(trajectory) - lookback):
        # Create input sequence
        input_seq = trajectory[i:i+lookback]
        # Target is the next state
        target = trajectory[i+lookback]
        X.append(input_seq.flatten())
        y.append(target)
    
    return np.array(X), np.array(y)

def train_koopman_model(X, y, epochs=200, learning_rate=0.001, 
                        koopman_dim=10, observable_dim=5):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Initialize model 
    model = KoopmanOperatorNetwork(
        state_dim=X.shape[1], 
        koopman_dim=koopman_dim, 
        observable_dim=observable_dim
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=20, 
        verbose=True
    )
    
    # Training history
    training_loss_history = []
    training_accuracy_history = []
    
    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        
        # Forward pass
        predictions = model(X_tensor)
        
        # Compute loss
        loss = criterion(predictions, y_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Scheduler step
        scheduler.step(loss)
        
        # Compute accuracy (within a threshold)
        threshold = 0.1  # 10% tolerance
        with torch.no_grad():
            accuracy = torch.mean((torch.abs(predictions - y_tensor) < threshold).float()) * 100
        
        # Store history
        training_loss_history.append(loss.item())
        training_accuracy_history.append(accuracy.item())
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy.item():.2f}%")
    
    # Plot training progress
    plt.figure(figsize=(12,5))
    
    # Loss subplot
    plt.subplot(1,2,1)
    plt.plot(training_loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Accuracy subplot
    plt.subplot(1,2,2)
    plt.plot(training_accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()
    
    return model

def predict_lorenz_trajectory(model, initial_state, steps=100):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()  # Set to evaluation mode
    trajectory = [initial_state]
    
    current_state = torch.tensor(initial_state, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for _ in range(steps):
            # Predict next state
            next_state = model(current_state)
            trajectory.append(next_state.cpu().numpy())
            current_state = next_state
    
    return np.array(trajectory)

def visualize_trajectories(true_trajectory, predicted_trajectory):
    # 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # True trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], 
             label='True Trajectory', color='blue')
    ax1.set_title('True Lorenz Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Predicted trajectory
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], 
             label='Predicted Trajectory', color='red')
    ax2.set_title('Predicted Koopman Trajectory')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Difference trajectory
    ax3 = fig.add_subplot(133, projection='3d')
    diff_traj = true_trajectory[:len(predicted_trajectory)] - predicted_trajectory
    ax3.plot(diff_traj[:, 0], diff_traj[:, 1], diff_traj[:, 2], 
             label='Trajectory Difference', color='green')
    ax3.set_title('Trajectory Difference')
    ax3.set_xlabel('X Difference')
    ax3.set_ylabel('Y Difference')
    ax3.set_zlabel('Z Difference')
    
    plt.tight_layout()
    plt.show()
    
    # Compute and print error metrics
    mse = np.mean((true_trajectory[:len(predicted_trajectory)] - predicted_trajectory)**2)
    mae = np.mean(np.abs(true_trajectory[:len(predicted_trajectory)] - predicted_trajectory))
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Simulation parameters
    initial_state = [1.0, 1.0, 1.0]
    
    # Generate true Lorenz trajectory
    true_trajectory = simulate_lorenz(initial_state, steps=5000)
    
    # Prepare training data
    X, y = prepare_sequence_data(true_trajectory, lookback=1)
    
    # Train Koopman model
    koopman_model = train_koopman_model(X, y, 
                                        epochs=700, 
                                        learning_rate=0.002,
                                        koopman_dim=10, 
                                        observable_dim=5)
    
    
    # Predict trajectory
    predicted_trajectory = predict_lorenz_trajectory(
        koopman_model, 
        initial_state, 
        steps=len(true_trajectory) - 1
    )
    
    # Visualize results
    visualize_trajectories(true_trajectory, predicted_trajectory)

if __name__ == "__main__":
    main()
