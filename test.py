import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.model import Model
from loader.utd_mhad import UTDMHAD_IMUDataset, collate_fn_no_pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Same params as in train.py ===
root_sim = '/home/lala/Documents/Data/VQIMU/UTD_MHAD'
root_real = '/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/Inertial'
window_size = 24
stride = 1
batch_size = 512
epsilon = 1e-8
lambda_cross = 1

# === Test dataset & loader ===
test_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[7, 8],
                                  window_size=window_size, stride=stride)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_fn_no_pad, num_workers=16)

# === Compute normalization ===
real_acc_sum = real_acc_sum_sq = real_acc_count = 0
synthetic_acc_sum = synthetic_acc_sum_sq = synthetic_acc_count = 0
real_gyro_sum = real_gyro_sum_sq = real_gyro_count = 0
synthetic_gyro_sum = synthetic_gyro_sum_sq = synthetic_gyro_count = 0

for batch in test_loader:
    for sample in batch:
        acc_sim = sample['acc_sim']
        synthetic_acc_sum += acc_sim.sum().item()
        synthetic_acc_sum_sq += (acc_sim ** 2).sum().item()
        synthetic_acc_count += acc_sim.numel()
        acc_real = sample['acc_real']
        real_acc_sum += acc_real.sum().item()
        real_acc_sum_sq += (acc_real ** 2).sum().item()
        real_acc_count += acc_real.numel()

        gyro_sim = sample['gyro_sim']
        synthetic_gyro_sum += gyro_sim.sum().item()
        synthetic_gyro_sum_sq += (gyro_sim ** 2).sum().item()
        synthetic_gyro_count += gyro_sim.numel()
        gyro_real = sample['gyro_real']
        real_gyro_sum += gyro_real.sum().item()
        real_gyro_sum_sq += (gyro_real ** 2).sum().item()
        real_gyro_count += gyro_real.numel()

# === Stats ===
real_acc_mean = real_acc_sum / real_acc_count
real_acc_var = (real_acc_sum_sq / real_acc_count) - (real_acc_mean ** 2) + epsilon
synthetic_acc_mean = synthetic_acc_sum / synthetic_acc_count
synthetic_acc_var = (synthetic_acc_sum_sq / synthetic_acc_count) - (synthetic_acc_mean ** 2) + epsilon

real_gyro_mean = real_gyro_sum / real_gyro_count
real_gyro_var = (real_gyro_sum_sq / real_gyro_count) - (real_gyro_mean ** 2) + epsilon
synthetic_gyro_mean = synthetic_gyro_sum / synthetic_gyro_count
synthetic_gyro_var = (synthetic_gyro_sum_sq / synthetic_gyro_count) - (synthetic_gyro_mean ** 2) + epsilon

print(f"Real Acc Mean/Var: {real_acc_mean:.6f}, {real_acc_var:.6f}")
print(f"Synth Acc Mean/Var: {synthetic_acc_mean:.6f}, {synthetic_acc_var:.6f}")
print(f"Real Gyro Mean/Var: {real_gyro_mean:.6f}, {real_gyro_var:.6f}")
print(f"Synth Gyro Mean/Var: {synthetic_gyro_mean:.6f}, {synthetic_gyro_var:.6f}")

# === Model params ===
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 4
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99

# === Load models ===
best_model_dir = './best_models'

# ACC
model_real_acc = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model_synth_acc = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                        num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model_real_acc.load_state_dict(torch.load(os.path.join(best_model_dir, 'best_model_real_acc.pth')))
model_synth_acc.load_state_dict(torch.load(os.path.join(best_model_dir, 'best_model_synth_acc.pth')))

# GYRO
model_real_gyro = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                        num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model_synth_gyro = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                         num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model_real_gyro.load_state_dict(torch.load(os.path.join(best_model_dir, 'best_model_real_gyro.pth')))
model_synth_gyro.load_state_dict(torch.load(os.path.join(best_model_dir, 'best_model_synth_gyro.pth')))

for m in [model_real_acc, model_synth_acc, model_real_gyro, model_synth_gyro]:
    m.eval()

# === Metrics containers ===
def init_metrics():
    return {k: 0 for k in [
        'total_loss_real', 'recon_error_real', 'vq_loss_real', 'perplexity_real',
        'total_loss_synth', 'recon_error_synth', 'vq_loss_synth', 'perplexity_synth',
        'cross_loss'
    ]}

acc_metrics = init_metrics()
gyro_metrics = init_metrics()

# === Evaluate ===
with torch.no_grad():
    loop = tqdm(test_loader, desc="Evaluating", leave=False)
    for batch in loop:
        # ===== ACC =====
        real_acc = torch.stack([s['acc_real'] for s in batch]).to(device).transpose(1, 2)
        synth_acc = torch.stack([s['acc_sim'] for s in batch]).to(device).transpose(1, 2)

        vq_loss_real_acc, recon_real_acc, perplexity_real_acc, _ = model_real_acc(real_acc)
        vq_loss_synth_acc, recon_synth_acc, perplexity_synth_acc, _ = model_synth_acc(synth_acc)

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

        for k, v in zip(
            ['total_loss_real', 'recon_error_real', 'vq_loss_real', 'perplexity_real',
             'total_loss_synth', 'recon_error_synth', 'vq_loss_synth', 'perplexity_synth', 'cross_loss'],
            [total_loss_real_acc, recon_error_real_acc, vq_loss_real_acc, perplexity_real_acc,
             total_loss_synth_acc, recon_error_synth_acc, vq_loss_synth_acc, perplexity_synth_acc, cross_loss_acc]
        ):
            acc_metrics[k] += v.item()

        # ===== GYRO =====
        real_gyro = torch.stack([s['gyro_real'] for s in batch]).to(device).transpose(1, 2)
        synth_gyro = torch.stack([s['gyro_sim'] for s in batch]).to(device).transpose(1, 2)

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

        for k, v in zip(
            ['total_loss_real', 'recon_error_real', 'vq_loss_real', 'perplexity_real',
             'total_loss_synth', 'recon_error_synth', 'vq_loss_synth', 'perplexity_synth', 'cross_loss'],
            [total_loss_real_gyro, recon_error_real_gyro, vq_loss_real_gyro, perplexity_real_gyro,
             total_loss_synth_gyro, recon_error_synth_gyro, vq_loss_synth_gyro, perplexity_synth_gyro, cross_loss_gyro]
        ):
            gyro_metrics[k] += v.item()

# === Average ===
for metrics in [acc_metrics, gyro_metrics]:
    for k in metrics:
        metrics[k] /= len(test_loader)

print("\n=== Test Results (ACC) ===")
print(f"Real Loss: {acc_metrics['total_loss_real']:.6f}, Recon: {acc_metrics['recon_error_real']:.6f}, "
      f"VQ: {acc_metrics['vq_loss_real']:.6f}, Perplexity: {acc_metrics['perplexity_real']:.6f}")
print(f"Synth Loss: {acc_metrics['total_loss_synth']:.6f}, Recon: {acc_metrics['recon_error_synth']:.6f}, "
      f"VQ: {acc_metrics['vq_loss_synth']:.6f}, Perplexity: {acc_metrics['perplexity_synth']:.6f}")
print(f"Cross Loss: {acc_metrics['cross_loss']:.6f}")

print("\n=== Test Results (GYRO) ===")
print(f"Real Loss: {gyro_metrics['total_loss_real']:.6f}, Recon: {gyro_metrics['recon_error_real']:.6f}, "
      f"VQ: {gyro_metrics['vq_loss_real']:.6f}, Perplexity: {gyro_metrics['perplexity_real']:.6f}")
print(f"Synth Loss: {gyro_metrics['total_loss_synth']:.6f}, Recon: {gyro_metrics['recon_error_synth']:.6f}, "
      f"VQ: {gyro_metrics['vq_loss_synth']:.6f}, Perplexity: {gyro_metrics['perplexity_synth']:.6f}")
print(f"Cross Loss: {gyro_metrics['cross_loss']:.6f}")

# === Plotting 5 sequences ===
print("\n=== Plotting sample sequences ===")
vis_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_no_pad, num_workers=1)

seq_count = 0
acc_buf_real, acc_buf_synth, acc_buf_cross = [], [], []
gyro_buf_real, gyro_buf_synth, gyro_buf_cross = [], [], []

with torch.no_grad():
    for batch in vis_loader:
        sample = batch[0]

        # ACC
        real_acc = sample['acc_real'].unsqueeze(0).to(device).transpose(1, 2)
        synth_acc = sample['acc_sim'].unsqueeze(0).to(device).transpose(1, 2)

        z_real_acc = model_real_acc._encoder(real_acc)
        z_real_acc = model_real_acc._pre_vq_conv(z_real_acc).unsqueeze(-1)
        _, quant_real_acc, _, _ = model_real_acc._vq_vae(z_real_acc)
        quant_real_acc = quant_real_acc.squeeze(-1)
        cross_acc = model_synth_acc._decoder(quant_real_acc)

        acc_buf_real.append(real_acc.squeeze().cpu())
        acc_buf_synth.append(synth_acc.squeeze().cpu())
        acc_buf_cross.append(cross_acc.squeeze().cpu())

        # GYRO
        real_gyro = sample['gyro_real'].unsqueeze(0).to(device).transpose(1, 2)
        synth_gyro = sample['gyro_sim'].unsqueeze(0).to(device).transpose(1, 2)

        z_real_gyro = model_real_gyro._encoder(real_gyro)
        z_real_gyro = model_real_gyro._pre_vq_conv(z_real_gyro).unsqueeze(-1)
        _, quant_real_gyro, _, _ = model_real_gyro._vq_vae(z_real_gyro)
        quant_real_gyro = quant_real_gyro.squeeze(-1)
        cross_gyro = model_synth_gyro._decoder(quant_real_gyro)

        gyro_buf_real.append(real_gyro.squeeze().cpu())
        gyro_buf_synth.append(synth_gyro.squeeze().cpu())
        gyro_buf_cross.append(cross_gyro.squeeze().cpu())

        # When we have 4 windows, merge and plot
        if len(acc_buf_real) == 4:
            acc_synth_seq = torch.cat(acc_buf_synth, dim=1)
            acc_cross_seq = torch.cat(acc_buf_cross, dim=1)
            gyro_synth_seq = torch.cat(gyro_buf_synth, dim=1)
            gyro_cross_seq = torch.cat(gyro_buf_cross, dim=1)

            fig, axs = plt.subplots(6, 1, figsize=(12, 10))
            fig.suptitle(f"Sequence {seq_count+1}")

            # ACC channels
            for ch in range(acc_synth_seq.shape[0]):
                axs[ch].plot(acc_synth_seq[ch], label="Original Synth ACC")
                axs[ch].plot(acc_cross_seq[ch], label="From Real ACC", linestyle='--')
                axs[ch].set_ylabel(f"ACC Ch {ch+1}")
                axs[ch].legend()

            # GYRO channels
            for ch in range(gyro_synth_seq.shape[0]):
                axs[ch+3].plot(gyro_synth_seq[ch], label="Original Synth GYRO")
                axs[ch+3].plot(gyro_cross_seq[ch], label="From Real GYRO", linestyle='--')
                axs[ch+3].set_ylabel(f"GYRO Ch {ch+1}")
                axs[ch+3].legend()

            plt.tight_layout()
            plt.show()

            acc_buf_real.clear()
            acc_buf_synth.clear()
            acc_buf_cross.clear()
            gyro_buf_real.clear()
            gyro_buf_synth.clear()
            gyro_buf_cross.clear()

            seq_count += 1
            if seq_count >= 5:
                break
