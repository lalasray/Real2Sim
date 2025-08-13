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
real_acc_sum = real_acc_sum_sq = real_acc_count = 0
synthetic_acc_sum = synthetic_acc_sum_sq = synthetic_acc_count = 0

for loader in [train_loader, val_loader, test_loader]:
    for batch in loader:
        for sample in batch:
            acc_sim = sample['acc_sim']  # synthetic
            synthetic_acc_sum += acc_sim.sum().item()
            synthetic_acc_sum_sq += (acc_sim ** 2).sum().item()
            synthetic_acc_count += acc_sim.numel()
            if 'acc_real' in sample:
                acc_real = sample['acc_real']
                real_acc_sum += acc_real.sum().item()
                real_acc_sum_sq += (acc_real ** 2).sum().item()
                real_acc_count += acc_real.numel()

epsilon = 1e-8
real_acc_mean = real_acc_sum / real_acc_count
real_acc_var = (real_acc_sum_sq / real_acc_count) - (real_acc_mean ** 2) + epsilon

synthetic_acc_mean = synthetic_acc_sum / synthetic_acc_count
synthetic_acc_var = (synthetic_acc_sum_sq / synthetic_acc_count) - (synthetic_acc_mean ** 2) + epsilon

print("Real Acc Mean/Var:", real_acc_mean, real_acc_var)
print("Synthetic Acc Mean/Var:", synthetic_acc_mean, synthetic_acc_var)

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 4

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3

model_real_acc = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model_synth_acc = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                        num_embeddings, embedding_dim, commitment_cost, decay).to(device)

opt_real_acc = torch.optim.Adam(model_real_acc.parameters(), lr=learning_rate)
opt_synth_acc = torch.optim.Adam(model_synth_acc.parameters(), lr=learning_rate)



def build_models():
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 4
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99

    model_real_acc = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                           num_embeddings, embedding_dim, commitment_cost, decay)
    model_synth_acc = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                            num_embeddings, embedding_dim, commitment_cost, decay)
    return model_real_acc, model_synth_acc

lambda_cross = 1

num_epochs = 500
patience = 50 
epochs_no_improve = 0
best_val_loss = float('inf')

best_model_dir = './best_models'
os.makedirs(best_model_dir, exist_ok=True)

for epoch in range(num_epochs):
    model_real_acc.train()
    model_synth_acc.train()
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
        real_acc_list = [sample['acc_real'] for sample in batch]
        synth_acc_list = [sample['acc_sim'] for sample in batch]

        real_acc = torch.stack(real_acc_list).to(device)    # (B, W, 3)
        synth_acc = torch.stack(synth_acc_list).to(device)  # (B, W, 3)

        real_acc = real_acc.transpose(1, 2)   # (B, 3, W)
        synth_acc = synth_acc.transpose(1, 2) # (B, 3, W)

        vq_loss_real_acc, recon_real_acc, perplexity_real_acc, _ = model_real_acc(real_acc)
        vq_loss_synth_acc, recon_synth_acc, perplexity_synth_acc, _ = model_synth_acc(synth_acc)

        # Removed matching loss entirely

        z_real_acc = model_real_acc._encoder(real_acc)
        z_real_acc = model_real_acc._pre_vq_conv(z_real_acc).unsqueeze(-1)
        _, quantized_real_acc, _, _ = model_real_acc._vq_vae(z_real_acc)
        quantized_real_acc = quantized_real_acc.squeeze(-1)
        cross_recon_real_to_synth_acc = model_synth_acc._decoder(quantized_real_acc)

        cross_loss_acc = F.mse_loss(cross_recon_real_to_synth_acc, synth_acc) / synthetic_acc_var
        recon_error_real_acc = F.mse_loss(recon_real_acc, real_acc) / (real_acc_var + epsilon)
        recon_error_synth_acc = F.mse_loss(recon_synth_acc, synth_acc) / (synthetic_acc_var + epsilon)

        total_loss_real_acc = recon_error_real_acc + vq_loss_real_acc + lambda_cross * cross_loss_acc
        total_loss_synth_acc = recon_error_synth_acc + vq_loss_synth_acc

        opt_real_acc.zero_grad()
        total_loss_real_acc.backward(retain_graph=True)
        opt_real_acc.step()

        opt_synth_acc.zero_grad()
        total_loss_synth_acc.backward(retain_graph=True)
        opt_synth_acc.step()

        train_metrics['total_loss_real'] += total_loss_real_acc.item()
        train_metrics['recon_error_real'] += recon_error_real_acc.item()
        train_metrics['vq_loss_real'] += vq_loss_real_acc.item()
        train_metrics['perplexity_real'] += perplexity_real_acc.item()

        train_metrics['total_loss_synth'] += total_loss_synth_acc.item()
        train_metrics['recon_error_synth'] += recon_error_synth_acc.item()
        train_metrics['vq_loss_synth'] += vq_loss_synth_acc.item()
        train_metrics['perplexity_synth'] += perplexity_synth_acc.item()

        train_metrics['cross_loss'] += cross_loss_acc.item()

        loop.set_postfix({
            'RealLoss': total_loss_real_acc.item(),
            'SynthLoss': total_loss_synth_acc.item(),
            'PerpReal': perplexity_real_acc.item(),
            'PerpSynth': perplexity_synth_acc.item()
        })

    # Average metrics
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    print(f"Train Real Acc Loss: {train_metrics['total_loss_real']:.6f}, Recon: {train_metrics['recon_error_real']:.6f}, VQ: {train_metrics['vq_loss_real']:.6f}, Perplexity: {train_metrics['perplexity_real']:.6f}")
    print(f"Train Synth Acc Loss: {train_metrics['total_loss_synth']:.6f}, Recon: {train_metrics['recon_error_synth']:.6f}, VQ: {train_metrics['vq_loss_synth']:.6f}, Perplexity: {train_metrics['perplexity_synth']:.6f}")
    print(f"Cross Loss: {train_metrics['cross_loss']:.6f}")

    # Validation
    model_real_acc.eval()
    model_synth_acc.eval()

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
            real_acc_list = [sample['acc_real'] for sample in batch]
            synth_acc_list = [sample['acc_sim'] for sample in batch]

            real_acc = torch.stack(real_acc_list).to(device)
            synth_acc = torch.stack(synth_acc_list).to(device)

            real_acc = real_acc.transpose(1, 2)
            synth_acc = synth_acc.transpose(1, 2)

            vq_loss_real_acc, recon_real_acc, perplexity_real_acc, _ = model_real_acc(real_acc)
            vq_loss_synth_acc, recon_synth_acc, perplexity_synth_acc, _ = model_synth_acc(synth_acc)

            # Removed matching loss entirely

            z_real_acc = model_real_acc._encoder(real_acc)
            z_real_acc = model_real_acc._pre_vq_conv(z_real_acc).unsqueeze(-1)
            _, quantized_real_acc, _, _ = model_real_acc._vq_vae(z_real_acc)
            quantized_real_acc = quantized_real_acc.squeeze(-1)
            cross_recon_real_to_synth_acc = model_synth_acc._decoder(quantized_real_acc)

            cross_loss_acc = F.mse_loss(cross_recon_real_to_synth_acc, synth_acc) / synthetic_acc_var
            recon_error_real_acc = F.mse_loss(recon_real_acc, real_acc) / (real_acc_var + epsilon)
            recon_error_synth_acc = F.mse_loss(recon_synth_acc, synth_acc) / (synthetic_acc_var + epsilon)

            total_loss_real_acc = recon_error_real_acc + vq_loss_real_acc + lambda_cross * cross_loss_acc
            total_loss_synth_acc = recon_error_synth_acc + vq_loss_synth_acc

            val_metrics['total_loss_real'] += total_loss_real_acc.item()
            val_metrics['recon_error_real'] += recon_error_real_acc.item()
            val_metrics['vq_loss_real'] += vq_loss_real_acc.item()
            val_metrics['perplexity_real'] += perplexity_real_acc.item()

            val_metrics['total_loss_synth'] += total_loss_synth_acc.item()
            val_metrics['recon_error_synth'] += recon_error_synth_acc.item()
            val_metrics['vq_loss_synth'] += vq_loss_synth_acc.item()
            val_metrics['perplexity_synth'] += perplexity_synth_acc.item()

            val_metrics['cross_loss'] += cross_loss_acc.item()

    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    print(f"Val Real Acc Loss: {val_metrics['total_loss_real']:.6f}, Recon: {val_metrics['recon_error_real']:.6f}, VQ: {val_metrics['vq_loss_real']:.6f}, Perplexity: {val_metrics['perplexity_real']:.6f}")
    print(f"Val Synth Acc Loss: {val_metrics['total_loss_synth']:.6f}, Recon: {val_metrics['recon_error_synth']:.6f}, VQ: {val_metrics['vq_loss_synth']:.6f}, Perplexity: {val_metrics['perplexity_synth']:.6f}")
    print(f"Cross Loss: {val_metrics['cross_loss']:.6f}")

    # Early stopping & saving best model
    if val_metrics['total_loss_real'] < best_val_loss:
        best_val_loss = val_metrics['total_loss_real']
        epochs_no_improve = 0
        best_model_real = copy.deepcopy(model_real_acc.state_dict())
        best_model_synth = copy.deepcopy(model_synth_acc.state_dict())
        torch.save(best_model_real, os.path.join(best_model_dir, 'best_model_real_acc.pth'))
        torch.save(best_model_synth, os.path.join(best_model_dir, 'best_model_synth_acc.pth'))
        print("Saved Best Models")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
