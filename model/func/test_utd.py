import os
import torch
from torch.utils.data import Dataset, DataLoader

class R2SUTDMHADDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        sample = torch.load(file_path)
        return sample

def collate_fn(batch):
    imu_batch = {}
    positions = ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]
    for pos in positions:
        accel_list = [sample["imu"][pos]["accel"] if sample["imu"][pos] is not None else None for sample in batch]
        gyro_list = [sample["imu"][pos]["gyro"] if sample["imu"][pos] is not None else None for sample in batch]
        imu_batch[pos] = {"accel": accel_list, "gyro": gyro_list}

    sentence_embeddings = torch.stack([sample["sentence_embedding"] for sample in batch])
    subject_ids = [sample["subject_id"] for sample in batch]
    activity_ids = [sample["activity_id"] for sample in batch]

    return {
        "imu": imu_batch,
        "sentence_embedding": sentence_embeddings,
        "subject_id": subject_ids,
        "activity_id": activity_ids
    }

if __name__ == "__main__":
    data_dir = r"C:\Users\DFKILenovo\Desktop\UTD_MHAD_R2S"
    dataset = R2SUTDMHADDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print("Total number of clips in dataset:", len(dataset))

    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}")
        for pos in ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]:
            if batch["imu"][pos]["accel"][0] is not None:
                print(f"{pos} accel shape: {batch['imu'][pos]['accel'][0].shape}, gyro shape: {batch['imu'][pos]['gyro'][0].shape}")
            else:
                print(f"{pos}: None")
        print("Sentence embedding shape:", batch["sentence_embedding"].shape)
        print("Subject IDs:", batch["subject_id"])
        print("Activity IDs:", batch["activity_id"])
        
