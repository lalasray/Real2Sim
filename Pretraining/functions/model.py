import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from windowed_loader import IMUSlidingWindowDataset
folder = "/home/lala/Documents/Data/Motion-Xplusplus/processed_dataset"

window_sizes = [30, 60, 90, 120, 150, 210, 300]
datasets = [IMUSlidingWindowDataset(folder, window_size=ws, stride=10) for ws in window_sizes]

# Merge all datasets
merged_dataset = ConcatDataset(datasets)

print("Total number of sliding windows across all window sizes:", len(merged_dataset))