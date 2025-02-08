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


def advanced_loss(model, decoded_ae, decoded_rollout, koopman_states, data, lambda1=1.0, lambda2=0.1):
    Lae = F.mse_loss(decoded_ae, data.x)
    Lpred = F.mse_loss(decoded_rollout, data.x)
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

dataset = torch.load("./data/pid_dataset_1.pth")
model = AdvancedKoopmanModel(input_dim=12, koopman_dim=32)
losses = train_advanced_model(model, dataset, lambda1=1.0, lambda2=0.2)
save_folder = "quadcopter-koopman-models"
os.makedirs(save_folder, exist_ok = True)
save_path = os.path.join(save_folder, "quadcopter-koopman-model-5.1.pth")
torch.save(model.state_dict(), save_path)

print(f"Model saved to {save_path}")
