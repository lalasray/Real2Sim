import os
import torch
from model import IMUModel  # adjust import if needed

# ----------------------------
# Settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = r"C:\Users\DFKILenovo\Downloads"
model_path = os.path.join(save_path, "best_model.pth")

# ----------------------------
# Load trained model
# ----------------------------
model = IMUModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded model from {model_path}")

# ----------------------------
# Example individual inference
# ----------------------------
seq_len = 30  # example window length

# ----------------------------
# Left Wrist
accel = torch.randn(1, seq_len, 3).to(device)
gyro = torch.randn(1, seq_len, 3).to(device)
lengths = torch.tensor([seq_len]).to(device)

with torch.no_grad():
    left_wrist_emb = model.encoders["left_wrist"](accel, gyro, lengths)
print("Left Wrist embedding shape:", left_wrist_emb.shape)

# ----------------------------
# Right Wrist
accel = torch.randn(1, seq_len, 3).to(device)
gyro = torch.randn(1, seq_len, 3).to(device)
lengths = torch.tensor([seq_len]).to(device)

with torch.no_grad():
    right_wrist_emb = model.encoders["right_wrist"](accel, gyro, lengths)
print("Right Wrist embedding shape:", right_wrist_emb.shape)

# ----------------------------
# Left Thigh
accel = torch.randn(1, seq_len, 3).to(device)
gyro = torch.randn(1, seq_len, 3).to(device)
lengths = torch.tensor([seq_len]).to(device)

with torch.no_grad():
    left_thigh_emb = model.encoders["left_thigh"](accel, gyro, lengths)
print("Left Thigh embedding shape:", left_thigh_emb.shape)

# ----------------------------
# Right Thigh
accel = torch.randn(1, seq_len, 3).to(device)
gyro = torch.randn(1, seq_len, 3).to(device)
lengths = torch.tensor([seq_len]).to(device)

with torch.no_grad():
    right_thigh_emb = model.encoders["right_thigh"](accel, gyro, lengths)
print("Right Thigh embedding shape:", right_thigh_emb.shape)

# ----------------------------
# Sentence Embedding
sentence_emb = torch.randn(1, 768).to(device)

with torch.no_grad():
    sentence_out = model.sentence_encoder(sentence_emb)
print("Sentence embedding shape:", sentence_out.shape)
