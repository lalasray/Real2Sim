import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import pandas as pd
from model.model import Model
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
window_size = 24
stride = 24  # can be < window_size for overlap
batch_size = 64  # parallel processing
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 4
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
best_model_dir = './best_models'

# ----------------------------
# Load trained models
# ----------------------------
def load_model(path):
    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

model_real_acc = load_model(os.path.join(best_model_dir, 'best_model_real_acc.pth'))
model_synth_acc = load_model(os.path.join(best_model_dir, 'best_model_synth_acc.pth'))
model_real_gyro = load_model(os.path.join(best_model_dir, 'best_model_real_gyro.pth'))
model_synth_gyro = load_model(os.path.join(best_model_dir, 'best_model_synth_gyro.pth'))

# ----------------------------
# Dataset for real IMU data
# ----------------------------
class RealIMUDataset(Dataset):
    def __init__(self, root_dir, window_size=24, stride=24):
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.samples = []  # (file_index, start_idx, acc_window, gyro_window)
        self.file_list = []

        files = sorted(glob.glob(os.path.join(root_dir, '*.mat')))
        for file_idx, fpath in enumerate(files):
            mat_data = loadmat(fpath)
            if 'd_iner' not in mat_data:
                continue
            data = mat_data['d_iner']
            acc = data[:, 0:3]
            gyro = data[:, 3:6]
            self.file_list.append((fpath, len(acc)))

            # create windows
            for start in range(0, len(acc) - window_size + 1, stride):
                self.samples.append((file_idx, start, acc[start:start+window_size], gyro[start:start+window_size]))

            # leftover samples at the end
            remainder = len(acc) % stride
            if remainder != 0 and len(acc) > window_size:
                start = len(acc) - window_size
                self.samples.append((file_idx, start, acc[start:], gyro[start:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start, acc_win, gyro_win = self.samples[idx]
        return {
            'file_idx': file_idx,
            'start': start,
            'acc_win': torch.tensor(acc_win, dtype=torch.float32),
            'gyro_win': torch.tensor(gyro_win, dtype=torch.float32)
        }

# ----------------------------
# Model inference
# ----------------------------
def generate_batch(real_batch, model_real, model_synth):
    x = real_batch.transpose(1, 2).to(device)  # (B, C, L)
    with torch.no_grad():
        z_e = model_real._encoder(x)
        z_e = model_real._pre_vq_conv(z_e).unsqueeze(-1)
        _, quantized, _, _ = model_real._vq_vae(z_e)
        quantized = quantized.squeeze(-1)
        synth_out = model_synth._decoder(quantized)
        synth_out = synth_out.transpose(1, 2).cpu().numpy()
    return synth_out  # (B, L, C)

# ----------------------------
# Process all files
# ----------------------------
def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = RealIMUDataset(input_dir, window_size=window_size, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Preallocate storage for each file
    file_outputs_acc = {}
    file_outputs_gyro = {}
    file_counts = {}
    for file_idx, (fpath, length) in enumerate(dataset.file_list):
        file_outputs_acc[file_idx] = np.zeros((length, 3), dtype=np.float32)
        file_outputs_gyro[file_idx] = np.zeros((length, 3), dtype=np.float32)
        file_counts[file_idx] = np.zeros(length, dtype=np.int32)

    for batch in loader:
        file_idxs = batch['file_idx'].numpy()
        starts = batch['start'].numpy()
        acc_wins = batch['acc_win']
        gyro_wins = batch['gyro_win']

        synth_acc_batch = generate_batch(acc_wins, model_real_acc, model_synth_acc)
        synth_gyro_batch = generate_batch(gyro_wins, model_real_gyro, model_synth_gyro)

        for i in range(len(file_idxs)):
            fidx = file_idxs[i]
            start = starts[i]
            L = synth_acc_batch.shape[1]
            file_outputs_acc[fidx][start:start+L] += synth_acc_batch[i]
            file_outputs_gyro[fidx][start:start+L] += synth_gyro_batch[i]
            file_counts[fidx][start:start+L] += 1

    # Save each file
    for file_idx, (fpath, length) in enumerate(dataset.file_list):
        counts = file_counts[file_idx]
        counts[counts == 0] = 1  # avoid division by zero
        final_acc = file_outputs_acc[file_idx] / counts[:, None]
        final_gyro = file_outputs_gyro[file_idx] / counts[:, None]

        output_df = pd.DataFrame({
            'synth_acc_x': final_acc[:, 0],
            'synth_acc_y': final_acc[:, 1],
            'synth_acc_z': final_acc[:, 2],
            'synth_gyro_x': final_gyro[:, 0],
            'synth_gyro_y': final_gyro[:, 1],
            'synth_gyro_z': final_gyro[:, 2],
        })

        base_name = os.path.splitext(os.path.basename(fpath))[0] + '_synthetic.csv'
        output_path = os.path.join(output_dir, base_name)
        output_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    input_dir = r"/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/Inertial"  # folder with .mat files
    output_dir = r"/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/real2sim"
    process_dataset(input_dir, output_dir)
