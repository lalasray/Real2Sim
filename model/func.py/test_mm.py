import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class MMFit_IMUDataset(Dataset):
    POSITIONS = ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]

    def __init__(self, imu_root_dir, label_root_dir, embed_model_name='sentence-transformers/gtr-t5-base'):
        self.imu_root_dir = imu_root_dir
        self.label_root_dir = label_root_dir
        self.embed_model = SentenceTransformer(embed_model_name)
        self.samples = []

        # Iterate over all wXX folders
        for subject_folder in glob.glob(os.path.join(imu_root_dir, "w*")):
            if not os.path.isdir(subject_folder):
                continue

            folder_name = os.path.basename(subject_folder)
            # Look for CSV inside the wXX folder
            label_csv = os.path.join(label_root_dir, folder_name, f"{folder_name}_labels.csv")
            if not os.path.exists(label_csv):
                print(f"Skipping {folder_name}, no label CSV found.")
                continue

            # Load CSV
            label_df = pd.read_csv(label_csv, sep=',', header=None, skipinitialspace=True,
                                   names=["start", "end", "label_id", "activity"])

            # Merge all parts for each position
            merged_imu = {}
            for pos in self.POSITIONS:
                acc_list, gyro_list = [], []
                part_folders = sorted([os.path.join(subject_folder, p) for p in os.listdir(subject_folder)
                                       if os.path.isdir(os.path.join(subject_folder, p))])
                
                default_acc_shape, default_gyro_shape = None, None
                
                for part in part_folders:
                    npz_file = os.path.join(part, f"wham_output_{pos}_imusim.npz")
                    if os.path.exists(npz_file):
                        data = np.load(npz_file)
                        acc = np.squeeze(data['accelerometer'], axis=0)
                        gyro = np.squeeze(data['gyroscope'], axis=0)
                        if default_acc_shape is None:
                            default_acc_shape = acc.shape
                            default_gyro_shape = gyro.shape
                        print(f"[DEBUG] {folder_name} - {part} - {pos} - accel shape: {acc.shape}, gyro shape: {gyro.shape}")
                    else:
                        print(f"[WARNING] Missing {npz_file}, filling with zeros.")
                        if default_acc_shape is not None:
                            acc = np.zeros(default_acc_shape, dtype=np.float32)
                            gyro = np.zeros(default_gyro_shape, dtype=np.float32)
                        else:
                            acc = np.zeros((100, 3), dtype=np.float32)  # default length if first part missing
                            gyro = np.zeros((100, 3), dtype=np.float32)

                    acc_list.append(acc)
                    gyro_list.append(gyro)

                merged_imu[pos] = {
                    "accelerometer": np.concatenate(acc_list, axis=0),
                    "gyroscope": np.concatenate(gyro_list, axis=0)
                }

            # Store each sample
            for idx, row in label_df.iterrows():
                try:
                    start, end = int(row.start), int(row.end)
                except ValueError:
                    print(f"[WARNING] Invalid start/end in {folder_name} row {idx}, skipping")
                    continue
                activity_text = row.activity
                self.samples.append({
                    "folder": folder_name,
                    "merged_imu": merged_imu,
                    "start": start,
                    "end": end,
                    "activity": activity_text,
                    "index": idx
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        merged_imu = sample_info["merged_imu"]
        start, end = sample_info["start"], sample_info["end"]
        activity_text = sample_info["activity"]
        folder_name = sample_info["folder"]
        row_idx = sample_info["index"]

        # Slice merged IMU sequences
        imu_data = {}
        for pos in self.POSITIONS:
            acc = torch.tensor(merged_imu[pos]["accelerometer"][start:end], dtype=torch.float32).unsqueeze(0)
            gyro = torch.tensor(merged_imu[pos]["gyroscope"][start:end], dtype=torch.float32).unsqueeze(0)
            imu_data[pos] = {"accel": acc, "gyro": gyro}

        # Generate embeddings from activity text
        embed = torch.tensor(self.embed_model.encode(activity_text, convert_to_numpy=True),
                             dtype=torch.float32).unsqueeze(0)
        sentence_embedding = embed.clone()
        title_embedding = embed.clone()

        # Unique clip name
        clip_name = f"{folder_name}_{activity_text}_{row_idx}"

        return {
            "clip_name": clip_name,
            "imu": imu_data,
            "sentence_embedding": sentence_embedding,
            "title_embedding": title_embedding
        }

def imu_collate_fn(batch):
    clip_names = [sample["clip_name"] for sample in batch]
    imu_batch = {}
    for pos in MMFit_IMUDataset.POSITIONS:
        accel_list = [sample["imu"][pos]["accel"] for sample in batch]
        gyro_list = [sample["imu"][pos]["gyro"] for sample in batch]
        imu_batch[pos] = {"accel": accel_list, "gyro": gyro_list}
    sentence_embeddings = torch.stack([sample["sentence_embedding"] for sample in batch])
    title_embeddings = torch.stack([sample["title_embedding"] for sample in batch])
    return {
        "clip_name": clip_names,
        "imu": imu_batch,
        "sentence_embedding": sentence_embeddings,
        "title_embedding": title_embeddings
    }

if __name__ == "__main__":
    imu_root = r"/home/lala/Documents/Data/VQIMU/MMFIT_snych/MMFIT/smpl"
    label_root = r"/home/lala/Downloads/mm-fit"
    output_dir = r"/home/lala/Documents/Data/Motion-Xplusplus/processed_dataset"
    os.makedirs(output_dir, exist_ok=True)

    dataset = MMFit_IMUDataset(imu_root, label_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=imu_collate_fn)

    for batch in dataloader:
        clip_name = batch["clip_name"][0]
        imu_sample = {pos: {"accel": batch["imu"][pos]["accel"][0],
                            "gyro": batch["imu"][pos]["gyro"][0]}
                      for pos in MMFit_IMUDataset.POSITIONS}

        save_path = os.path.join(output_dir, f"{clip_name}.pt")
        torch.save({
            "clip_name": clip_name,
            "imu": imu_sample,
            "sentence_embedding": batch["sentence_embedding"][0],
            "title_embedding": batch["title_embedding"][0]
        }, save_path)

        if os.path.exists(save_path):
            print(f"[SUCCESS] Saved {save_path}")
            # Optionally, try loading back
            try:
                loaded = torch.load(save_path)
                print(f"[VERIFY] Loaded {clip_name}, imu keys: {list(loaded['imu'].keys())}")
            except Exception as e:
                print(f"[ERROR] Could not load {save_path}: {e}")
        else:
            print(f"[ERROR] Failed to save {save_path}")