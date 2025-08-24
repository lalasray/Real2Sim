import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
from sentence_transformers import SentenceTransformer

class UTDMHAD_IMUDatasetSimOnly(Dataset):
    ACTIONS = [
        "right arm swipe to the left",                  # 1
        "right arm swipe to the right",                 # 2
        "right hand wave",                              # 3
        "two hand front clap",                          # 4
        "right arm throw",                              # 5
        "cross arms in the chest",                      # 6
        "basketball shoot",                             # 7
        "right hand draw x",                            # 8
        "right hand draw circle (clockwise)",           # 9
        "right hand draw circle (counter clockwise)",   # 10
        "draw triangle",                                # 11
        "bowling (right hand)",                         # 12
        "front boxing",                                 # 13
        "baseball swing from right",                    # 14
        "tennis right hand forehand swing",             # 15
        "arm curl (two arms)",                          # 16
        "tennis serve",                                 # 17
        "two hand push",                                # 18
        "right hand knock on door",                     # 19
        "right hand catch an object",                   # 20
        "right hand pick up and throw",                 # 21
        "jogging in place",                             # 22
        "walking in place",                             # 23
        "sit to stand",                                 # 24
        "stand to sit",                                 # 25
        "forward lunge (left foot forward)",           # 26
        "squat (two arms stretch out)"                 # 27
    ]

    def __init__(self, root_dir, subjects=None, transform=None, window_size=30, stride=15, embed_model_name='sentence-transformers/gtr-t5-base'):
        """
        Dataset using only simulated data, split into windows.
        Converts activity description to embedding using SentenceTransformer.

        Args:
            root_dir (str): path to simulated npz data
            subjects (list or None): subjects to include
            transform (callable, optional): transform on sample
            window_size (int): length of each window in timesteps (30, 60, 90)
            stride (int): stride for sliding window
            embed_model_name (str): pre-trained SentenceTransformer model name
        """
        self.root_dir = root_dir
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        self.samples = []  # list of tuples: (sim_fpath, activity)

        # Load embedding model
        self.embed_model = SentenceTransformer(embed_model_name)

        # Compute embeddings for all activity strings
        self.activity_embeddings = torch.tensor(self.embed_model.encode(self.ACTIONS, convert_to_numpy=True), dtype=torch.float32)

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

            self.samples.append((fpath, activity))

        # Precompute all windows for all samples
        self.window_indices = []
        for i, (sim_fpath, activity) in enumerate(self.samples):
            data_sim = np.load(sim_fpath)
            acc_sim = torch.tensor(data_sim['accelerometer'], dtype=torch.float32).squeeze()
            seq_len = acc_sim.shape[0]

            starts = list(range(0, seq_len - window_size + 1, stride))
            if len(starts) == 0 and seq_len > 0:
                starts = [0]

            for start_idx in starts:
                self.window_indices.append((i, start_idx))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        sample_idx, start_idx = self.window_indices[idx]
        sim_fpath, activity = self.samples[sample_idx]

        data_sim = np.load(sim_fpath)
        acc_sim = torch.tensor(data_sim['accelerometer'], dtype=torch.float32).squeeze()
        gyro_sim = torch.tensor(data_sim['gyroscope'], dtype=torch.float32).squeeze()

        acc_sim_window = self.get_window(acc_sim, start_idx)
        gyro_sim_window = self.get_window(gyro_sim, start_idx)

        # 0-based label
        label = activity - 1
        label_str = self.ACTIONS[label]
        label_embed = self.activity_embeddings[label]  # torch tensor

        sample = {
            'acc_sim': acc_sim_window,
            'gyro_sim': gyro_sim_window,
            'activity': label,
            'activity_str': label_str,
            'activity_embed': label_embed
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_window(self, data, start_idx):
        seq_len = data.shape[0]
        end_idx = start_idx + self.window_size
        if end_idx <= seq_len:
            return data[start_idx:end_idx]
        else:
            pad_len = end_idx - seq_len
            window = data[start_idx:seq_len]
            pad_tensor = torch.zeros((pad_len, data.shape[1]), dtype=data.dtype)
            return torch.cat([window, pad_tensor], dim=0)

    def print_data_stats(self):
        from collections import Counter

        all_activities = [self.samples[sample_idx][1] for (sample_idx, _) in self.window_indices]
        counts = Counter(all_activities)

        total_classes = len(counts)
        total_datapoints = len(self.window_indices)

        print(f"Total classes: {total_classes}")
        print(f"Total datapoints (windows): {total_datapoints}")
        print("Datapoints per class:")
        for cls, cnt in sorted(counts.items()):
            label = cls - 1  # 0-based
            print(f"  Activity {cls} ({self.ACTIONS[label]}): {cnt}")

def collate_fn_no_pad(batch):
    return batch


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)

    root_sim = r"C:\Users\DFKILenovo\Desktop\UTD_MHAD"

    window_size = 60
    stride = 1

    train_dataset = UTDMHAD_IMUDatasetSimOnly(root_sim, subjects=[1,2,3,4,5], window_size=window_size, stride=stride)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_no_pad)

    for batch in train_loader:
        for sample in batch:
            print("Activity label:", sample['activity'])
            print("Activity description:", sample['activity_str'])
            print("Activity embedding shape:", sample['activity_embed'].shape)
            print("Simulated acc shape:", sample['acc_sim'].shape)
            print("Simulated gyro shape:", sample['gyro_sim'].shape)
        break

    train_dataset.print_data_stats()
