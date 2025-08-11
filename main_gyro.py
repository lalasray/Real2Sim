import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import Model
from loader.utd_mhad import UTDMHAD_IMUDataset, collate_fn_no_pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_sim = '/home/lala/Documents/Data/VQIMU/UTD_MHAD'
root_real = '/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/Inertial'

window_size = 24
stride = 1
batch_size = 512

train_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[1, 2, 3, 4, 5], window_size=window_size, stride=stride)
val_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[6], window_size=window_size, stride=stride)
test_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[7, 8], window_size=window_size, stride=stride)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_no_pad, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_no_pad, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_no_pad, num_workers=16)

# Compute mean and variance for normalization
real_gyro_sum = real_gyro_sum_sq = real_gyro_count = 0
synthetic_gyro_sum = synthetic_gyro_sum_sq = synthetic_gyro_count = 0

for loader in [train_loader, val_loader, test_loader]:
    for batch in loader:
        for sample in batch:
            gyro_sim = sample['gyro_sim']  # synthetic gyro
            synthetic_gyro_sum += gyro_sim.sum().item()
            synthetic_gyro_sum_sq += (gyro_sim ** 2).sum().item()
            synthetic_gyro_count += gyro_sim.numel()

            if 'gyro_real' in sample:
                gyro_real = sample['gyro_real']
                real_gyro_sum += gyro_real.sum().item()
                real_gyro_sum_sq += (gyro_real ** 2).sum().item()
                real_gyro_count += gyro_real.numel()

epsilon = 1e-8
real_gyro_mean = real_gyro_sum / real_gyro_count
real_gyro_var = (real_gyro_sum_sq / real_gyro_count) - (real_gyro_mean ** 2) + epsilon

synthetic_gyro_mean = synthetic_gyro_sum / synthetic_gyro_count
synthetic_gyro_var = (synthetic_gyro_sum_sq / synthetic_gyro_count) - (synthetic_gyro_mean ** 2) + epsilon

print("Real Gyro Mean/Var:", real_gyro_mean, real_gyro_var)
print("Synthetic Gyro Mean/Var:", synthetic_gyro_mean, synthetic_gyro_var)

# Model parameters
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 4

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3

model_real_gyro = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                        num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model_synth_gyro = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                         num_embeddings, embedding_dim, commitment_cost, decay).to(device)

opt_real_gyro = torch.optim.Adam(model_real_gyro.parameters(), lr=learning_rate)
opt_synth_gyro = torch.optim.Adam(model_synth_gyro.parameters(), lr=learning_rate)

lambda_cross = 1
num_epochs = 500
patience = 50
epochs_no_improve = 0
best_val_loss = float('inf')

best_model_dir = './best_models'
os.makedirs(best_model_dir, exist_ok=True)

for epoch in range(num_epochs):
    model_real_gyro.train()
    model_synth_gyro.train()
    print(f"Epoch {epoch+1}/{num_epochs}")

    train_metrics = {
        'total_loss_real': 0,
        'recon_error_real': 0,
        'vq_loss_real': 0,
        'perplexity_real': 0,
        'total_loss_synth': 0,
        'recon_error_synth': 0,
        'vq_loss_synth': 0,
        'perplexity_synth': 0,
        'cross_loss': 0
    }

    loop = tqdm(train_loader, desc='Training', leave=False)
    for batch in loop:
        real_gyro_list = [sample['gyro_real'] for sample in batch]
        synth_gyro_list = [sample['gyro_sim'] for sample in batch]

        real_gyro = torch.stack(real_gyro_list).to(device)
        synth_gyro = torch.stack(synth_gyro_list).to(device)

        real_gyro = real_gyro.transpose(1, 2)   # (B, 3, W)
        synth_gyro = synth_gyro.transpose(1, 2) # (B, 3, W)

        vq_loss_real_gyro, recon_real_gyro, perplexity_real_gyro, _ = model_real_gyro(real_gyro)
        vq_loss_synth_gyro, recon_synth_gyro, perplexity_synth_gyro, _ = model_synth_gyro(synth_gyro)

        # Removed matching loss entirely

        z_real_gyro = model_real_gyro._encoder(real_gyro)
        z_real_gyro = model_real_gyro._pre_vq_conv(z_real_gyro).unsqueeze(-1)
        _, quantized_real_gyro, _, _ = model_real_gyro._vq_vae(z_real_gyro)
        quantized_real_gyro = quantized_real_gyro.squeeze(-1)
        cross_recon_real_to_synth_gyro = model_synth_gyro._decoder(quantized_real_gyro)

        cross_loss_gyro = F.mse_loss(cross_recon_real_to_synth_gyro, synth_gyro) / synthetic_gyro_var
        recon_error_real_gyro = F.mse_loss(recon_real_gyro, real_gyro) / (real_gyro_var + epsilon)
        recon_error_synth_gyro = F.mse_loss(recon_synth_gyro, synth_gyro) / (synthetic_gyro_var + epsilon)

        total_loss_real_gyro = recon_error_real_gyro + vq_loss_real_gyro + lambda_cross * cross_loss_gyro
        total_loss_synth_gyro = recon_error_synth_gyro + vq_loss_synth_gyro

        opt_real_gyro.zero_grad()
        total_loss_real_gyro.backward(retain_graph=True)
        opt_real_gyro.step()

        opt_synth_gyro.zero_grad()
        total_loss_synth_gyro.backward()
        opt_synth_gyro.step()

        train_metrics['total_loss_real'] += total_loss_real_gyro.item()
        train_metrics['recon_error_real'] += recon_error_real_gyro.item()
        train_metrics['vq_loss_real'] += vq_loss_real_gyro.item()
        train_metrics['perplexity_real'] += perplexity_real_gyro.item()

        train_metrics['total_loss_synth'] += total_loss_synth_gyro.item()
        train_metrics['recon_error_synth'] += recon_error_synth_gyro.item()
        train_metrics['vq_loss_synth'] += vq_loss_synth_gyro.item()
        train_metrics['perplexity_synth'] += perplexity_synth_gyro.item()

        train_metrics['cross_loss'] += cross_loss_gyro.item()

        loop.set_postfix({
            'RealLoss': total_loss_real_gyro.item(),
            'SynthLoss': total_loss_synth_gyro.item(),
            'PerpReal': perplexity_real_gyro.item(),
            'PerpSynth': perplexity_synth_gyro.item()
        })

    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    print(f"Train Real Gyro Loss: {train_metrics['total_loss_real']:.6f}, Recon: {train_metrics['recon_error_real']:.6f}, VQ: {train_metrics['vq_loss_real']:.6f}, Perplexity: {train_metrics['perplexity_real']:.6f}")
    print(f"Train Synth Gyro Loss: {train_metrics['total_loss_synth']:.6f}, Recon: {train_metrics['recon_error_synth']:.6f}, VQ: {train_metrics['vq_loss_synth']:.6f}, Perplexity: {train_metrics['perplexity_synth']:.6f}")
    print(f"Cross Loss: {train_metrics['cross_loss']:.6f}")

    # Validation
    model_real_gyro.eval()
    model_synth_gyro.eval()

    val_metrics = {
        'total_loss_real': 0,
        'recon_error_real': 0,
        'vq_loss_real': 0,
        'perplexity_real': 0,
        'total_loss_synth': 0,
        'recon_error_synth': 0,
        'vq_loss_synth': 0,
        'perplexity_synth': 0,
        'cross_loss': 0
    }

    with torch.no_grad():
        for batch in val_loader:
            real_gyro_list = [sample['gyro_real'] for sample in batch]
            synth_gyro_list = [sample['gyro_sim'] for sample in batch]

            real_gyro = torch.stack(real_gyro_list).to(device)
            synth_gyro = torch.stack(synth_gyro_list).to(device)

            real_gyro = real_gyro.transpose(1, 2)
            synth_gyro = synth_gyro.transpose(1, 2)

            vq_loss_real_gyro, recon_real_gyro, perplexity_real_gyro, _ = model_real_gyro(real_gyro)
            vq_loss_synth_gyro, recon_synth_gyro, perplexity_synth_gyro, _ = model_synth_gyro(synth_gyro)

            z_real_gyro = model_real_gyro._encoder(real_gyro)
            z_real_gyro = model_real_gyro._pre_vq_conv(z_real_gyro).unsqueeze(-1)
            _, quantized_real_gyro, _, _ = model_real_gyro._vq_vae(z_real_gyro)
            quantized_real_gyro = quantized_real_gyro.squeeze(-1)
            cross_recon_real_to_synth_gyro = model_synth_gyro._decoder(quantized_real_gyro)

            cross_loss_gyro = F.mse_loss(cross_recon_real_to_synth_gyro, synth_gyro) / synthetic_gyro_var
            recon_error_real_gyro = F.mse_loss(recon_real_gyro, real_gyro) / (real_gyro_var + epsilon)
            recon_error_synth_gyro = F.mse_loss(recon_synth_gyro, synth_gyro) / (synthetic_gyro_var + epsilon)

            total_loss_real_gyro = recon_error_real_gyro + vq_loss_real_gyro + lambda_cross * cross_loss_gyro
            total_loss_synth_gyro = recon_error_synth_gyro + vq_loss_synth_gyro

            val_metrics['total_loss_real'] += total_loss_real_gyro.item()
            val_metrics['recon_error_real'] += recon_error_real_gyro.item()
            val_metrics['vq_loss_real'] += vq_loss_real_gyro.item()
            val_metrics['perplexity_real'] += perplexity_real_gyro.item()

            val_metrics['total_loss_synth'] += total_loss_synth_gyro.item()
            val_metrics['recon_error_synth'] += recon_error_synth_gyro.item()
            val_metrics['vq_loss_synth'] += vq_loss_synth_gyro.item()
            val_metrics['perplexity_synth'] += perplexity_synth_gyro.item()

            val_metrics['cross_loss'] += cross_loss_gyro.item()

    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    print(f"Val Real Gyro Loss: {val_metrics['total_loss_real']:.6f}, Recon: {val_metrics['recon_error_real']:.6f}, VQ: {val_metrics['vq_loss_real']:.6f}, Perplexity: {val_metrics['perplexity_real']:.6f}")
    print(f"Val Synth Gyro Loss: {val_metrics['total_loss_synth']:.6f}, Recon: {val_metrics['recon_error_synth']:.6f}, VQ: {val_metrics['vq_loss_synth']:.6f}, Perplexity: {val_metrics['perplexity_synth']:.6f}")
    print(f"Cross Loss: {val_metrics['cross_loss']:.6f}")

    # Early stopping & best model saving
    if val_metrics['total_loss_real'] < best_val_loss:
        best_val_loss = val_metrics['total_loss_real']
        epochs_no_improve = 0
        best_model_real = copy.deepcopy(model_real_gyro.state_dict())
        best_model_synth = copy.deepcopy(model_synth_gyro.state_dict())
        torch.save(best_model_real, os.path.join(best_model_dir, 'best_model_real_gyro.pth'))
        torch.save(best_model_synth, os.path.join(best_model_dir, 'best_model_synth_gyro.pth'))
        print("Saved Best Models")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
