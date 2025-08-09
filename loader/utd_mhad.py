import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch.nn.functional as F
import sys

class UTDMHAD_IMUDataset(Dataset):
    def __init__(self, root_dir, real_dir=None, subjects=None, transform=None, window_size=30, stride=15):
        """
        Dataset that splits each sequence into multiple windows and treats each window as a separate datapoint.

        Args:
            root_dir (str): path to simulated npz data
            real_dir (str or None): path to real .mat IMU data
            subjects (list or None): subjects to include
            transform (callable, optional): transform on sample
            window_size (int): length of each window in timesteps
            stride (int): stride for sliding window
        """
        self.root_dir = root_dir
        self.real_dir = real_dir
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        self.samples = []  # list of tuples: (sim_fpath, mat_path or None, activity)

        # Gather file paths + metadata
        pattern = os.path.join(root_dir, '**', 'wham_output_*_imusim.npz')
        all_files = glob.glob(pattern, recursive=True)

        for fpath in all_files:
            folder = os.path.basename(os.path.dirname(fpath))
            parts = folder.split('_')

            act_part = next((p for p in parts if p.startswith('a')), None)
            subj_part = next((p for p in parts if p.startswith('s')), None)
            if act_part is None or subj_part is None:
                continue

            activity = int(act_part[1:])
            subject = int(subj_part[1:])
            if subjects is not None and subject not in subjects:
                continue

            mat_path = None
            if real_dir is not None:
                mat_filename = folder.replace('_color', '') + '_inertial.mat'
                mat_path = os.path.join(real_dir, mat_filename)
                if not os.path.exists(mat_path):
                    print(f"Skipping (no real data): {mat_path}")
                    continue

            self.samples.append((fpath, mat_path, activity))

        # Precompute all windows for all samples to enable indexing by window
        self.window_indices = []  # list of tuples: (sample_idx, window_start_idx)

        for i, (sim_fpath, mat_path, activity) in enumerate(self.samples):
            data_sim = np.load(sim_fpath)
            acc_sim = torch.tensor(data_sim['accelerometer'], dtype=torch.float32).squeeze()
            seq_len = acc_sim.shape[0]

            # Generate window start indices for this sequence
            starts = list(range(0, seq_len - window_size + 1, stride))
            if len(starts) == 0 and seq_len > 0:
                # Handle short sequences by creating one padded window
                starts = [0]

            for start_idx in starts:
                self.window_indices.append((i, start_idx))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        sample_idx, start_idx = self.window_indices[idx]
        sim_fpath, mat_path, activity = self.samples[sample_idx]

        data_sim = np.load(sim_fpath)
        acc_sim = torch.tensor(data_sim['accelerometer'], dtype=torch.float32).squeeze()
        gyro_sim = torch.tensor(data_sim['gyroscope'], dtype=torch.float32).squeeze()

        # Extract window from simulated data
        acc_sim_window = self.get_window(acc_sim, start_idx)
        gyro_sim_window = self.get_window(gyro_sim, start_idx)

        sample = {
            'acc_sim': acc_sim_window,
            'gyro_sim': gyro_sim_window,
            'activity': activity
        }

        if mat_path is not None:
            mat_data = loadmat(mat_path)
            if 'd_iner' in mat_data:
                d_iner = mat_data['d_iner']
                acc_real = torch.tensor(d_iner[:, 0:3], dtype=torch.float32)
                gyro_real = torch.tensor(d_iner[:, 3:6], dtype=torch.float32)

                # Resample real to simulated length (full sequence)
                target_len = acc_sim.shape[0]
                acc_real_rs = self.resample_signal(acc_real, target_len)
                gyro_real_rs = self.resample_signal(gyro_real, target_len)

                # Extract window from resampled real data
                acc_real_window = self.get_window(acc_real_rs, start_idx)
                gyro_real_window = self.get_window(gyro_real_rs, start_idx)

                sample['acc_real'] = acc_real_window
                sample['gyro_real'] = gyro_real_window
            else:
                print(f"Warning: 'd_iner' key not found in {mat_path}")

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_window(self, data, start_idx):
        # If sequence shorter than window, pad at end
        seq_len = data.shape[0]
        end_idx = start_idx + self.window_size
        if end_idx <= seq_len:
            return data[start_idx:end_idx]
        else:
            # Pad with zeros
            pad_len = end_idx - seq_len
            window = data[start_idx:seq_len]
            pad_tensor = torch.zeros((pad_len, data.shape[1]), dtype=data.dtype)
            return torch.cat([window, pad_tensor], dim=0)

    def resample_signal(self, signal, target_len):
        # signal: (time, channels)
        signal_t = signal.T.unsqueeze(0)  # (1, channels, time)
        resampled = F.interpolate(signal_t, size=target_len, mode='linear', align_corners=False)
        return resampled.squeeze(0).T  # (time, channels)
    
    def print_data_stats(self):
        from collections import Counter

        all_activities = [self.samples[sample_idx][2] for (sample_idx, _) in self.window_indices]
        counts = Counter(all_activities)

        total_classes = len(counts)
        total_datapoints = len(self.window_indices)

        print(f"Total classes: {total_classes}")
        print(f"Total datapoints (windows): {total_datapoints}")
        print("Datapoints per class:")
        for cls, cnt in sorted(counts.items()):
            print(f"  Activity {cls}: {cnt}")

def collate_fn_no_pad(batch):
    return batch


if __name__ == '__main__':

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.plot import plot_sample_signals

    root_sim = '/home/lala/Documents/Data/VQIMU/UTD_MHAD'
    root_real = '/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/Inertial'

    window_size = 24
    stride = 1

    train_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[1, 2, 3, 4, 5], window_size=window_size, stride=stride)
    val_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[6], window_size=window_size, stride=stride)
    test_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[7, 8], window_size=window_size, stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_no_pad)

    for batch in train_loader:
        for sample in batch:
            print("Activity:", sample['activity'])
            print("Simulated acc shape:", sample['acc_sim'].shape)  # (window_size, 3)
            print("Simulated gyro shape:", sample['gyro_sim'].shape)
            if 'acc_real' in sample:
                print("Real acc shape:", sample['acc_real'].shape)
                print("Real gyro shape:", sample['gyro_real'].shape)
            else:
                print("No real IMU data for this sample")
            plot_sample_signals(sample)
        break
    train_dataset.print_data_stats()