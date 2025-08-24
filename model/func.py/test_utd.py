import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class UTDMHAD_IMUDatasetFull(Dataset):
    POSITIONS = ["left_thigh", "right_thigh", "left_wrist", "right_wrist"]

    ACTIONS = [
        "right arm swipe to the left", "right arm swipe to the right",
        "right hand wave", "two hand front clap", "right arm throw",
        "cross arms in the chest", "basketball shoot", "right hand draw x",
        "right hand draw circle (clockwise)", "right hand draw circle (counter clockwise)",
        "draw triangle", "bowling (right hand)", "front boxing", "baseball swing from right",
        "tennis right hand forehand swing", "arm curl (two arms)", "tennis serve",
        "two hand push", "right hand knock on door", "right hand catch an object",
        "right hand pick up and throw", "jogging in place", "walking in place",
        "sit to stand", "stand to sit", "forward lunge (left foot forward)",
        "squat (two arms stretch out)"
    ]

    def __init__(self, root_dir, subjects=None, embed_model_name='sentence-transformers/gtr-t5-base'):
        self.root_dir = root_dir
        self.embed_model = SentenceTransformer(embed_model_name)

        # Gather sample paths
        pattern = os.path.join(root_dir, '**', 'wham_output_*_imusim.npz')
        all_files = glob.glob(pattern, recursive=True)
        self.samples = []

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sim_fpath, activity = self.samples[idx]

        # Load full IMU sequence
        data_sim = np.load(sim_fpath)
        acc_sim = torch.tensor(data_sim['accelerometer'], dtype=torch.float32).squeeze().unsqueeze(0)  # [1, seq_len, 3]
        gyro_sim = torch.tensor(data_sim['gyroscope'], dtype=torch.float32).squeeze().unsqueeze(0)    # [1, seq_len, 3]

        # Duplicate single IMU to all positions
        imu_data = {
            pos: {"accel": acc_sim.clone(), "gyro": gyro_sim.clone()}
            for pos in self.POSITIONS
        }

        # Unique clip name from filename
        base_name = os.path.splitext(os.path.basename(sim_fpath))[0]
        clip_name = f"{base_name}"

        # Sentence & title embeddings from activity string
        activity_str = self.ACTIONS[activity - 1]
        sentence_embedding = torch.tensor(
            self.embed_model.encode(activity_str, convert_to_numpy=True), dtype=torch.float32
        ).unsqueeze(0)  # [1, 768]
        title_embedding = sentence_embedding.clone()  # [1, 768]

        return {
            "clip_name": clip_name,
            "imu": imu_data,
            "sentence_embedding": sentence_embedding,
            "title_embedding": title_embedding
        }


def imu_collate_fn(batch):
    clip_names = [sample["clip_name"] for sample in batch]
    imu_batch = {}
    for pos in UTDMHAD_IMUDatasetFull.POSITIONS:
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
    root_sim = r"/home/lala/Documents/Data/VQIMU/UTD_MHAD"
    output_dir = "/home/lala/Documents/Data/Motion-Xplusplus/processed_dataset"
    os.makedirs(output_dir, exist_ok=True)

    dataset = UTDMHAD_IMUDatasetFull(root_sim, subjects=[1,2,3,4,5,6,7])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=imu_collate_fn)

    print("Total number of clips in dataset:", len(dataset))

    for batch in dataloader:
        clip_name = batch["clip_name"][0]

        imu_sample = {pos: {"accel": batch["imu"][pos]["accel"][0],
                            "gyro": batch["imu"][pos]["gyro"][0]} 
                      for pos in UTDMHAD_IMUDatasetFull.POSITIONS}

        save_path = os.path.join(output_dir, f"{clip_name}.pt")
        torch.save({
            "clip_name": clip_name,
            "imu": imu_sample,
            "sentence_embedding": batch["sentence_embedding"][0],
            "title_embedding": batch["title_embedding"][0]
        }, save_path)

        print(f"Saved {clip_name} to {save_path}")
        print(f"IMU shapes:")
        for pos in imu_sample:
            print(f"{pos} accel: {imu_sample[pos]['accel'].shape}, gyro: {imu_sample[pos]['gyro'].shape}")
        print(f"Sentence embedding shape: {batch['sentence_embedding'][0].shape}")
        print(f"Title embedding shape: {batch['title_embedding'][0].shape}")
