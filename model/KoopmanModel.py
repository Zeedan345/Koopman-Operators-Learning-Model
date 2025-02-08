import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, output_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.norm3 = nn.LayerNorm(output_dim)

        #self.projection = nn.Linear(hidden_dim * 2, ouput_dim)

    def create_full_graph(self, x):
        n = x.size(0)
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n), n),
            torch.tile(torch.arange(n), (n,))
        ]).to(x.device)
        return edge_index
        
    def forward(self, data):
        x = data.x.float()
        # Use the edge_index provided in data instead of a full graph.
        # edge_index = self.create_full_graph(x)
        edge_index = data.edge_index  # Use the chain graph structure
        
        # GNN path
        gnn_x = self.conv1(x, edge_index).relu()
        gnn_x = self.norm1(gnn_x)
        
        gnn_x = self.conv2(gnn_x, edge_index).relu()
        gnn_x = self.norm2(gnn_x)

        gnn_x = self.conv3(gnn_x, edge_index).relu()
        gnn_x = self.norm3(gnn_x)

        # FC path 
        fc_x = self.fc1(x).relu()
        fc_x = self.fc2(fc_x).relu() 
        fc_x = self.fc3(fc_x).relu()
        fc_x = self.fc4(fc_x)

        return (gnn_x + fc_x) / 2

    
class AdvancedKoopmanModel(torch.nn.Module):
    def __init__(self, input_dim, koopman_dim, hidden_dim=128, u_dim =4):
        super(AdvancedKoopmanModel, self).__init__()
        self.encoder = GNN(input_dim, hidden_dim, koopman_dim)
        self.decoder = GNN(koopman_dim, hidden_dim, input_dim)
        init_matrix = torch.zeros(koopman_dim, koopman_dim)
        for i in range(0, koopman_dim-1, 2):
            init_matrix[i:i+2, i:i+2] = torch.tensor([[0., -1.], [1., 0.]])
        self.koopman_matrix = torch.nn.Parameter(init_matrix)
        self.L = nn.Linear(u_dim, koopman_dim, bias=False)
        nn.init.normal_(self.L.weight, mean=0, std=0.1)
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_std', torch.ones(input_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def update_statistics(self, x):
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_std = x.std(dim=0)
                
                if self.num_batches_tracked == 0:
                    self.running_mean = batch_mean
                    self.running_std = batch_std
                else:
                    momentum = 0.1
                    self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                    self.running_std = (1 - momentum) * self.running_std + momentum * batch_std
                
                self.num_batches_tracked += 1

    def metric_loss(self, g, states):
        dist_g = torch.cdist(g, g, p=2)
        dist_x = torch.cdist(states, states, p=2)
        return torch.mean(torch.abs(dist_g - dist_x))

    def system_identify(self, G, H, regularization=0.1):
        batch_size = G.size(0)
        I = torch.eye(self.koopman_matrix.size(0)).to(G.device)
        A = torch.matmul(H.transpose(1, 2), G) @ torch.inverse(
            torch.matmul(G.transpose(1, 2), G) + regularization * I
        )
        return A

    def forward(self, data):
        self.update_statistics(data.x)
        

        koopman_states = self.encoder(data)
        

        decoded_ae = self.decoder(Data(x=koopman_states, edge_index=data.edge_index))
        T = data.x.shape[0]
        g_hat = [koopman_states[0]]  # ĝ₁ = g₁
        
        for t in range(1, T):
            u_t = data.edge_attr[t - 1]
            next_g = g_hat[-1] @ self.koopman_matrix + self.L(u_t)
            g_hat.append(next_g)
        
        g_hat = torch.stack(g_hat, dim=0)
        
        decoded_rollout = self.decoder(Data(x=g_hat, edge_index=data.edge_index))
        
        return decoded_ae, decoded_rollout, koopman_states

    # def forward(self, data):
    #     # Update running statistics
    #     self.update_statistics(data.x)
    
    #     # Normalize input
    #     x_normalized = (data.x - self.running_mean) / (self.running_std + 1e-5)
    #     data_normalized = Data(x=x_normalized, edge_index=data.edge_index)
        
    #     # Rest of the forward pass
    #     koopman_space = self.encoder(data_normalized)
    #     next_koopman_space = koopman_space @ self.koopman_matrix
    #     new_data = Data(x=next_koopman_space, edge_index=data.edge_index)
    #     decoded_state = self.decoder(new_data)
        
    #     return decoded_state