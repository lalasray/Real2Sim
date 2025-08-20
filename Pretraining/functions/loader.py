import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class IMUDataset(Dataset):
    def __init__(self, csv_file, imu_timesteps=100, embedding_dim=512):
        self.data = pd.read_csv(csv_file)
        self.imu_timesteps = imu_timesteps
        self.embedding_dim = embedding_dim

        # Filter out rows with corrupt files
        valid_indices = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            try:
                for pos in ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]:
                    path = row[pos]
                    _ = np.load(path)  # test load
                if row["sentence_embedding"] != "MISSING":
                    _ = torch.load(row["sentence_embedding"])
                if row["title_embedding"] != "MISSING":
                    _ = torch.load(row["title_embedding"])
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping clip {row['clip_name']} due to error: {e}")
        
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.data.iloc[self.valid_indices[idx]]

        imu_data = {}
        for pos in ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]:
            path = row[pos]
            npz_data = np.load(path)
            accel = torch.tensor(npz_data["accelerometer"], dtype=torch.float32)
            gyro = torch.tensor(npz_data["gyroscope"], dtype=torch.float32)
            imu_data[pos] = {"accel": accel, "gyro": gyro}

        # Sentence embedding
        if row["sentence_embedding"] == "MISSING":
            sentence_tensor = torch.zeros(self.embedding_dim)
        else:
            sentence_tensor = torch.load(row["sentence_embedding"])

        # Title embedding
        if row["title_embedding"] == "MISSING":
            title_tensor = torch.zeros(self.embedding_dim)
        else:
            title_tensor = torch.load(row["title_embedding"])

        return {
            "clip_name": row["clip_name"],
            "imu": imu_data,
            "sentence_embedding": sentence_tensor,
            "title_embedding": title_tensor
        }    

def imu_collate_fn(batch):
    """
    Custom collate function for variable-length IMU sequences.
    IMU data stays as list of tensors per body part.
    Embeddings are stacked into tensors.
    """
    clip_names = [sample["clip_name"] for sample in batch]
    
    imu_batch = {}
    for pos in ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]:
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



# Example usage
if __name__ == "__main__":
    output_dir = "/home/lala/Documents/Data/Motion-Xplusplus/processed_dataset"
    os.makedirs(output_dir, exist_ok=True)

    csv_file = "file_paths.csv"  # your CSV
    dataset = IMUDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=imu_collate_fn)

    print("Total number of clips in dataset:", len(dataset))

    for batch in dataloader:
        # Since batch_size=1, get the first item
        clip_name = batch["clip_name"][0]
        
        sample_to_save = {
            "clip_name": clip_name,
            "imu": {pos: {"accel": batch["imu"][pos]["accel"][0],
                        "gyro": batch["imu"][pos]["gyro"][0]} 
                    for pos in ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]},
            "sentence_embedding": batch["sentence_embedding"][0],
            "title_embedding": batch["title_embedding"][0]
        }
        
        save_path = os.path.join(output_dir, f"{clip_name}.pt")
        torch.save(sample_to_save, save_path)

        print(f"Saved {clip_name} to {save_path}")
