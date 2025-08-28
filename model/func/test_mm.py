import os
import glob
import numpy as np
import pandas as pd

imu_root_dir = r"C:\Users\DFKILenovo\Desktop\MMFIT_snych\MMFIT"
save_root_dir = r"C:\Users\DFKILenovo\Desktop\MMFIT_snych\MMFIT_R2S"

# Recursively find all _imusim.npz files
npz_files = glob.glob(os.path.join(imu_root_dir, "**", "*_imusim.npz"), recursive=True)

if not npz_files:
    print("No _imusim.npz files found.")
    exit()

for npz_path in npz_files:
    print(f"Processing: {npz_path}")

    # Load data
    data = np.load(npz_path)
    acc = np.squeeze(data['accelerometer'], axis=0)
    gyro = np.squeeze(data['gyroscope'], axis=0)

    # Add tiny noise
    acc_noisy = acc + np.random.normal(0, 0.1, acc.shape)
    gyro_noisy = gyro + np.random.normal(0, 0.5, gyro.shape)

    # Apply rolling average
    acc_smooth = pd.DataFrame(acc_noisy).rolling(window=5, min_periods=1).mean().to_numpy()
    gyro_smooth = pd.DataFrame(gyro_noisy).rolling(window=10, min_periods=1).mean().to_numpy()

    # Prepare save path
    rel_path = os.path.relpath(npz_path, imu_root_dir)
    save_dir = os.path.join(save_root_dir, os.path.dirname(rel_path))
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}_processed.npz")

    # Save processed data
    np.savez(save_path, accelerometer=acc_smooth, gyroscope=gyro_smooth)

    print(f"Saved processed data to: {save_path}")

    # Stop after first file
    #break
