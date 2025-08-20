import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ----------------------------
# Collate function
# ----------------------------
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

# ----------------------------------
# Sliding window dataset from .pt files
# ----------------------------------
class IMUSlidingWindowDataset(Dataset):
    def __init__(self, folder, window_size=30, stride=10):
        """
        folder: path to folder containing saved .pt files
        window_size: number of timesteps in a window
        stride: number of timesteps to skip between windows
        """
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pt")]
        if len(self.files) == 0:
            print(f"Warning: No .pt files found in {folder}")
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        # Precompute windows for all clips
        for file_path in self.files:
            try:
                data = torch.load(file_path)
            except Exception as e:
                print(f"Skipping file {file_path} due to load error: {e}")
                continue

            imu_data = data["imu"]

            # Assume all body parts have same length
            first_pos = list(imu_data.keys())[0]
            seq_len = imu_data[first_pos]["accel"].shape[1]  # <-- FIXED
            #print(imu_data[first_pos]["accel"].shape)

            if seq_len < window_size:
                #print(f"Skipping clip {data['clip_name']} (length {seq_len}) < window_size {window_size}")
                continue

            # Generate start indices for sliding windows
            start_indices = list(range(0, seq_len - window_size + 1, stride))
            for start in start_indices:
                self.samples.append({
                    "file_path": file_path,
                    "start_idx": start,
                    "clip_name": data["clip_name"]
                })

        if len(self.samples) == 0:
            print("Warning: No sliding windows generated. Check your window_size and stride!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        data = torch.load(sample_info["file_path"])
        imu_data = data["imu"]
        start = sample_info["start_idx"]
        end = start + self.window_size

        # Extract window for each IMU sensor
        window_imu = {}
        for pos in ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]:
            accel_seq = imu_data[pos]["accel"][:, start:end, :]  # <-- FIXED
            gyro_seq = imu_data[pos]["gyro"][:, start:end, :]    # <-- FIXED
            window_imu[pos] = {"accel": accel_seq, "gyro": gyro_seq}

        return {
            "clip_name": sample_info["clip_name"],
            "imu": window_imu,
            "sentence_embedding": data["sentence_embedding"],
            "title_embedding": data["title_embedding"]
        }

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    folder = "/home/lala/Documents/Data/Motion-Xplusplus/processed_dataset"  # folder containing your saved .pt files

    window_sizes = [30, 60, 90, 120, 150, 210, 300]
    datasets = [IMUSlidingWindowDataset(folder, window_size=ws, stride=10) for ws in window_sizes]

    # Merge all datasets
    merged_dataset = ConcatDataset(datasets)

    print("Total number of sliding windows across all window sizes:", len(merged_dataset))

    print("Precomputing all sliding windows...")
    all_samples = [merged_dataset[i] for i in range(len(merged_dataset))]
    print(f"Total samples precomputed: {len(all_samples)}")

    # Save as a single file
    save_path = os.path.join(folder, "merged_dataset.pt")
    torch.save(all_samples, save_path)
    print(f"Saved merged dataset to {save_path}")