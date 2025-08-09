import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

class UTDMHAD_IMUDataset(Dataset):
    def __init__(self, root_dir, real_dir=None, subjects=None, transform=None):
        """
        Args:
            root_dir (str): path to simulated npz data (e.g. UTD_MHAD)
            real_dir (str or None): path to real .mat IMU data (e.g. UTD_MHAD_Inertial)
            subjects (list or None): list of subject numbers to include
            transform (callable, optional): transform on sample
        """
        self.root_dir = root_dir
        self.real_dir = real_dir
        self.transform = transform

        self.file_paths = []
        self.labels = []
        self.subjects = []

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

            if (subjects is None) or (subject in subjects):
                self.file_paths.append(fpath)
                self.labels.append(activity)
                self.subjects.append(subject)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        activity = self.labels[idx]

        # Load simulated data
        data_sim = np.load(fpath)
        acc_sim = data_sim['accelerometer']  # (time, 3)
        gyro_sim = data_sim['gyroscope']     # (time, 3)

        acc_sim = torch.tensor(acc_sim, dtype=torch.float32)
        gyro_sim = torch.tensor(gyro_sim, dtype=torch.float32)

        sample = {'acc_sim': acc_sim, 'gyro_sim': gyro_sim, 'activity': activity}

        # If real_dir given, also load real imu data from .mat
        if self.real_dir is not None:
            # Parse filename for matching .mat file
            # Example npz folder: a1_s1_t1_color
            folder = os.path.basename(os.path.dirname(fpath))
            # Build matching .mat filename: Inertiala1_s1_t1_inertial.mat
            mat_filename = 'Inertial' + folder.replace('_color','') + '_inertial.mat'
            mat_path = os.path.join(self.real_dir, mat_filename)

            if os.path.exists(mat_path):
                mat_data = loadmat(mat_path)
                # Inspect keys to get accelerometer and gyroscope
                # Usually keys like 'acc' or 'acc_data', 'gyro' or 'gyro_data' - adapt as needed
                # Example: assuming keys 'acc' and 'gyro' exist as arrays (time, 3)
                acc_real = mat_data.get('acc') or mat_data.get('acc_data')
                gyro_real = mat_data.get('gyro') or mat_data.get('gyro_data')

                if acc_real is not None and gyro_real is not None:
                    acc_real = torch.tensor(acc_real, dtype=torch.float32)
                    gyro_real = torch.tensor(gyro_real, dtype=torch.float32)
                    sample['acc_real'] = acc_real
                    sample['gyro_real'] = gyro_real
                else:
                    # If keys differ, print keys for debugging
                    print(f"Warning: Could not find 'acc' or 'gyro' keys in {mat_path}. Available keys: {list(mat_data.keys())}")
            else:
                print(f"Warning: Real IMU file not found: {mat_path}")

        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_fn_no_pad(batch):
    return batch


if __name__ == '__main__':
    root_sim = '/home/lala/Documents/Data/VQIMU/UTD_MHAD'
    root_real = '/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial'

    train_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[1, 2, 3, 4, 5])
    val_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[6])
    test_dataset = UTDMHAD_IMUDataset(root_sim, real_dir=root_real, subjects=[7, 8])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_no_pad)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_no_pad)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_no_pad)

    for batch in train_loader:
        for sample in batch:
            print("Activity:", sample['activity'])
            print("Simulated acc shape:", sample['acc_sim'].shape)
            print("Simulated gyro shape:", sample['gyro_sim'].shape)

            if 'acc_real' in sample:
                print("Real acc shape:", sample['acc_real'].shape)
                print("Real gyro shape:", sample['gyro_real'].shape)
            else:
                print("No real IMU data found for this sample")

        break
