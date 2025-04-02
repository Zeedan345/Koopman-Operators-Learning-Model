import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from model.KoopmanModel import AdvancedKoopmanModel
from simulations.QuadcopterEnv import QuadcopterEnv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

def train_advanced_model(model, dataset, epochs=50, lr=0.0001, lambda1=0.5, lambda2=0.2, lambda3 = 1e-3):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5, verbose=True)
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_ae = 0.0
        epoch_pred = 0.0
        epoch_metric = 0.0
        epoch_bl_loss = 0.0

        for i, data in enumerate(dataset):
            data = data.to(device)
            optimizer.zero_grad()

            decoded_ae, decoded_rollout, koopman_states = model(data)
            
            total_loss, Lae, Lpred, Lmetric, loss_bilinear = model.compute_losses(
                data, decoded_ae, decoded_rollout, koopman_states, 
                lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
            )
            # if epoch < 3:
            #     total_loss = Lae + lambda2 * Lmetric
            # else:
            #     total_loss = Lae + lambda1 * Lpred + lambda2 * Lmetric

            # Check loss before backward
            # try:
            #     loss_val = total_loss.item()
            #     print(f"Epoch {epoch}, Sample {i}: total_loss = {loss_val:.4f}")
            # except Exception as e:
            #     print("Error converting total_loss to float:", e)

            total_loss.backward()
            # for name, param in model.named_parameters():
            #     if 'koopman_blocks' in name and param.grad is not None:
            #         print("koopman_blocks.grad norm:", param.grad.norm().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            

            # # Check model parameters after update
            # for name, param in model.named_parameters():
            #     if torch.isnan(param).any():
            #         print(f"NaNs detected in parameter {name} after optimizer step.")

            epoch_loss += total_loss.item()
            epoch_ae += Lae.item()
            epoch_pred += Lpred.item()
            epoch_metric += Lmetric.item()
            epoch_bl_loss += loss_bilinear.item()
        
        avg_loss = epoch_loss / len(dataset)
        scheduler.step(avg_loss)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f} "
              f"(AE: {epoch_ae/len(dataset):.4f}, Pred: {epoch_pred/len(dataset):.4f}, "
              f"Metric: {epoch_metric/len(dataset):.4f},"
              f"Bilinear Loss: {epoch_bl_loss/len(dataset):.4f})")
    
    return train_losses


dataset = torch.load("./data/pid_dataset_2_big.pth")
model = AdvancedKoopmanModel(input_dim=12, koopman_dim=64, num_objects=4, h=4).to(device)


losses = train_advanced_model(model, dataset, lambda1=1.0, lambda2=0.3, lambda3=0.0)

save_folder = "quadcopter-koopman-models"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "quadcopter-koopman-model-02-v2.1.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
