import os
import csv
from collections import defaultdict

# Change this to your directory
imu_dir = "/home/lala/Documents/Data/Motion-Xplusplus/motion/motion_generation/smplx322"
sentence_dir = "/home/lala/Documents/Data/Motion-Xplusplus/text/sentence"
title_dir = "/home/lala/Documents/Data/Motion-Xplusplus/text/Title"
output_csv = "file_paths.csv"

# dictionary: clip_name -> dict of files
clips = defaultdict(dict)

# Step 1: collect imu files
for root, _, files in os.walk(imu_dir):
    for file in files:
        if file.endswith("_imusim.npz"):
            path = os.path.join(root, file)
            parts = file.split("_")
            position = parts[-3] + "_" + parts[-2]   # e.g. left_thigh
            clip_name = "_".join(parts[:-3])        # e.g. Ways_to_Catch_360_clip1
            clips[clip_name][position] = path

# Step 2: collect sentence embeddings
sentence_files = {f[:-3]: os.path.join(root, f)
                  for root, _, files in os.walk(sentence_dir)
                  for f in files if f.endswith(".pt")}

# Step 3: collect title embeddings
title_files = {f[:-3]: os.path.join(root, f)
               for root, _, files in os.walk(title_dir)
               for f in files if f.endswith(".pt")}

# Step 4: link embeddings to clips, write "MISSING" if not found
for clip_name in clips.keys():
    clips[clip_name]["sentence_embedding"] = sentence_files.get(clip_name, "MISSING")
    clips[clip_name]["title_embedding"] = title_files.get(clip_name, "MISSING")

# Step 5: write CSV
columns = ["left_thigh", "right_thigh", "left_wrist", "right_wrist", 
           "sentence_embedding", "title_embedding"]

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["clip_name"] + columns)  # header row
    for clip, files_dict in clips.items():
        row = [clip] + [files_dict.get(col, "MISSING") for col in columns]
        writer.writerow(row)

print(f"Saved {len(clips)} rows to {output_csv}")

