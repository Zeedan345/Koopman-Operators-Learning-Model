import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphKoopmanEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim=64):
        super(GraphKoopmanEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        # Global pooling
        h = global_mean_pool(h, batch)
        # Project to embedding space
        return self.linear(h)

class GraphKoopmanDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim=64):
        super(GraphKoopmanDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, g):
        return self.net(g)

class KoopmanGraphLoss(nn.Module):
    def __init__(self, lambda_pred=1.0, lambda_metric=1.0):
        super(KoopmanGraphLoss, self).__init__()
        self.lambda_pred = lambda_pred
        self.lambda_metric = lambda_metric
        
    def forward(self, model, x_seq, u_seq, edge_index, batch):
        """
        Calculate all three losses for the Koopman GNN model
        
        Parameters:
        - x_seq: Sequence of states [batch_size, seq_len, state_dim]
        - u_seq: Sequence of controls [batch_size, seq_len-1, control_dim]
        - edge_index: Graph connectivity
        - batch: Batch indices for graphs
        
        Returns:
        - total_loss: Combined loss value
        - loss_dict: Dictionary containing individual loss components
        """
        T = x_seq.size(1)
        batch_size = x_seq.size(0)
        
        # 1. Auto-encoding Loss (Lae)
        # Encode and decode each state
        embeddings = []
        reconstructions = []
        for t in range(T):
            g_t = model.encoder(x_seq[:, t], edge_index, batch)
            x_recon_t = model.decoder(g_t)
            embeddings.append(g_t)
            reconstructions.append(x_recon_t)
            
        embeddings = torch.stack(embeddings, dim=1)  # [batch_size, T, embedding_dim]
        reconstructions = torch.stack(reconstructions, dim=1)  # [batch_size, T, state_dim]
        
        Lae = torch.mean(torch.norm(reconstructions - x_seq, dim=-1))
        
        # 2. Prediction Loss (Lpred)
        # Rollout in Koopman space
        g_hat = embeddings[:, 0]  # Initial embedding
        predicted_states = [model.decoder(g_hat)]
        
        for t in range(T-1):
            # Apply Koopman operator: g_hat_next = K*g_hat + L*u
            g_hat = model.koopman(g_hat) + torch.mm(u_seq[:, t], model.control_matrix)
            predicted_states.append(model.decoder(g_hat))
            
        predicted_states = torch.stack(predicted_states, dim=1)
        Lpred = torch.mean(torch.norm(predicted_states - x_seq, dim=-1))
        
        # 3. Metric Loss (Lmetric)
        # Compute pairwise distances in both spaces
        def compute_pairwise_distances(x):
            n = x.size(0)
            square = torch.sum(x**2, dim=-1, keepdim=True)
            distances = square - 2 * torch.matmul(x, x.transpose(-2, -1)) + square.transpose(-2, -1)
            return torch.sqrt(torch.clamp(distances, min=1e-12))
        
        Lmetric = 0
        for t in range(T):
            # Distances in state space
            state_distances = compute_pairwise_distances(x_seq[:, t])
            # Distances in Koopman space
            koopman_distances = compute_pairwise_distances(embeddings[:, t])
            # Compute metric loss
            Lmetric += torch.mean(torch.abs(koopman_distances - state_distances))
        
        Lmetric = Lmetric / T
        
        # Combine losses
        total_loss = Lae + self.lambda_pred * Lpred + self.lambda_metric * Lmetric
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'autoencoding_loss': Lae.item(),
            'prediction_loss': Lpred.item(),
            'metric_loss': Lmetric.item()
        }
        
        return total_loss, loss_dict

def train_graph_koopman_model(model, train_loader, epochs=100, device='cpu'):
    """Train the Graph Koopman model with all three losses"""
    criterion = KoopmanGraphLoss(lambda_pred=1.0, lambda_metric=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (x_seq, u_seq, edge_index, batch) in enumerate(train_loader):
            x_seq = x_seq.to(device)
            u_seq = u_seq.to(device)
            edge_index = edge_index.to(device)
            batch = batch.to(device)
            
            # Calculate all losses
            total_loss, loss_components = criterion(model, x_seq, u_seq, edge_index, batch)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss_components)
        
        # Print epoch statistics
        if (epoch + 1) % 10 == 0:
            avg_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                         for k in epoch_losses[0].keys()}
            print(f'Epoch [{epoch+1}/{epochs}]')
            for loss_name, loss_value in avg_losses.items():
                print(f'{loss_name}: {loss_value:.4f}')
            print('-' * 50)
    
    return model
