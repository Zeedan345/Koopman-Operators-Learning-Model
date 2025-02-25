import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing

class CustomMessagePassing(MessagePassing):
    def __init__(self, in_channel, edge_dim,out_channel):
        super(CustomMessagePassing, self).__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channel + edge_dim, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel)
        )
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_attr):
        m = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(m)
    def update(self, aggr_out):
        return aggr_out




class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim):
        super(GNN, self).__init__()

        self.conv1 = CustomMessagePassing(in_channel=input_dim, edge_dim=edge_dim, out_channel=hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.conv2 = CustomMessagePassing(in_channel=hidden_dim, edge_dim=edge_dim, out_channel=hidden_dim//2)
        self.norm2 = nn.LayerNorm(hidden_dim//2)
        
        self.conv3 = CustomMessagePassing(in_channel=hidden_dim//2, edge_dim=edge_dim, out_channel=output_dim)
        self.norm3 = nn.LayerNorm(output_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)

    def create_full_graph(self, x):
        n = x.size(0)
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n), n),
            torch.tile(torch.arange(n), (n,))
        ]).to(x.device)
        return edge_index
        
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # GNN path
        gnn_x = self.conv1(x, edge_index, edge_attr)
        #assert not torch.isnan(gnn_x).any(), "NaNs after conv1"
        gnn_x = torch.relu(gnn_x)
        gnn_x = self.norm1(gnn_x)
        
        gnn_x = self.conv2(gnn_x, edge_index, edge_attr)
        gnn_x = torch.relu(gnn_x)
        gnn_x = self.norm2(gnn_x)

        gnn_x = self.conv3(gnn_x, edge_index, edge_attr)
        gnn_x = torch.relu(gnn_x)
        gnn_x = self.norm3(gnn_x)

        # FC path 
        fc_x = self.fc1(x).relu()
        fc_x = self.fc2(fc_x).relu() 
        fc_x = self.fc3(fc_x).relu()
        fc_x = self.fc4(fc_x)

        return (gnn_x + fc_x) / 2


    
class AdvancedKoopmanModel(torch.nn.Module):
    def __init__(self, input_dim, koopman_dim, num_objects, h, hidden_dim=128, u_dim =4):
        super(AdvancedKoopmanModel, self).__init__()

        self.num_objects = num_objects
        self.m = koopman_dim //num_objects #embeddings per object
        self.h = h #number of block types

        self.encoder = GNN(input_dim, hidden_dim, koopman_dim, edge_dim=4)
        self.decoder = GNN(koopman_dim, hidden_dim, input_dim, edge_dim=4)

        self.koopman_blocks = nn.Parameter(torch.zeros(h, self.m, self.m))
        
        for i in range(0, h, 2):
            if self.m >= 2:
                rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
                self.koopman_blocks.data[i, :2, :2] = rotation

        self.register_buffer('sigma', self.create_sigma(num_objects, h))

        self.L = nn.Linear(u_dim, koopman_dim, bias=False)
        nn.init.normal_(self.L.weight, mean=0, std=0.1)

        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_std', torch.ones(input_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def create_sigma(self, num_objects, h):
        sigma = torch.zeros(num_objects, num_objects, h)
        for i in range(num_objects):
            for j in range(num_objects):
                if i==j:
                    sigma[i, j, 0] = 1.0
                else:
                    sigma[i, j, 1] = 1.0
        return sigma
    def compute_koopman_matrix(self):
        sigma_expanded = self.sigma.unsqueeze(-1).unsqueeze(-1)
        K_blocks = (sigma_expanded * self.koopman_blocks).sum(dim=2)
        K = K_blocks.permute(0, 2, 1, 3).contiguous().view(self.num_objects * self.m, self.num_objects * self.m)
        # norm_k = K.norm()
        # max_norm = 0.99
        # if norm_k > max_norm:
        #     K = K*(max_norm/norm_k)
        # print("K norm:", K.norm().item())
        return K
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
        #x_normed = (data.x - self.running_mean) / (self.running_std + 1e-6)
        
        koopman_states = self.encoder(Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr))
        
        decoded_ae = self.decoder(Data(x=koopman_states, edge_index=data.edge_index, edge_attr=data.edge_attr))
        K = self.compute_koopman_matrix()

        T = data.x.shape[0]  # Total timesteps
        g_hat = [koopman_states[0]]  # Initialize with g₁
        
        for t in range(1, T):
            u_t = data.edge_attr[t - 1] 
            next_g = g_hat[-1] @ K + self.L(u_t)
            g_hat.append(next_g)
        
        g_hat = torch.stack(g_hat, dim=0)
        #print("g_hat min, max, mean:", g_hat.min(), g_hat.max(), g_hat.mean())
        decoded_rollout = self.decoder(Data(x=g_hat, edge_index=data.edge_index, edge_attr=data.edge_attr))
        
        return decoded_ae, decoded_rollout, koopman_states
    
    def compute_losses(self, data, decoded_ae, decoded_rollout, koopman_states, lambda1=1.0, lambda2=1.0):
        loss_ae = torch.mean(torch.norm(decoded_ae - data.x, dim=1))
        loss_pred = torch.mean(torch.norm(decoded_rollout[1:] - data.x[1:], dim=1)) 
        dist_g = torch.cdist(koopman_states, koopman_states, p=2)
        dist_x = torch.cdist(data.x, data.x, p=2)
        loss_metric = torch.mean(torch.abs(dist_g - dist_x))
        #print(f"Loss AE: {loss_ae.item()} Loss Pred: {loss_pred.item()} Metric Loss: {loss_metric.item()}")
        
        # Total loss
        total_loss = loss_ae + lambda1 * loss_pred + lambda2 * loss_metric
        return total_loss, loss_ae, loss_pred, loss_metric
