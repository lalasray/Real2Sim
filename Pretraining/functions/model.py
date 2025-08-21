import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
from windowed_loader import IMUSlidingWindowDataset, imu_collate_fn
from encoder import IMUFullWindowEncoder, IdentityEncoder
from info_nce import InfoNCE

# ----------------------------
# Dataset
# ----------------------------
folder = "/home/lala/Documents/Data/Motion-Xplusplus/processed_dataset"
window_sizes = [300]
datasets = [IMUSlidingWindowDataset(folder, window_size=ws, stride=10) for ws in window_sizes]
merged_dataset = ConcatDataset(datasets)

total_len = len(merged_dataset)
train_len = int(0.9 * total_len)
val_len = total_len - train_len
train_dataset, val_dataset = random_split(merged_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=imu_collate_fn, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=imu_collate_fn, num_workers=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# IMU Model
# ----------------------------
class IMUModel(nn.Module):
    def __init__(self, embedding_dim=768, subwindow_len=5, sub_stride=1, use_gyro=True):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "left_wrist": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro),
            "right_wrist": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro),
            "left_thigh": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro),
            "right_thigh": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro)
        })
        self.sentence_encoder = IdentityEncoder()

    def forward(self, imu_batch, sentence_emb):
        sentence_out = self.sentence_encoder(sentence_emb)
        imu_outs = {}
        for key, enc in self.encoders.items():
            acc, gyro = imu_batch[key]
            imu_outs[key] = enc(acc, gyro)
        return sentence_out, imu_outs

# ----------------------------
# Helper: convert list of tensors -> batch tensor
# ----------------------------
def stack_sensor(sensor_dict):
    accel = torch.cat(sensor_dict["accel"], dim=0).to(device)
    gyro = torch.cat(sensor_dict["gyro"], dim=0).to(device)
    return accel, gyro

# ----------------------------
# Initialize model, loss, optimizer
# ----------------------------
model = IMUModel().to(device)
loss_fn = InfoNCE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Loss weights
alpha = 1.0  # sentence ↔ sensor
beta = 0.5   # sensor ↔ sensor

# ----------------------------
# Early Stopping
# ----------------------------
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

# ----------------------------
# Training Loop
# ----------------------------
epochs = 1000
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0.0
    
    # --- Training with tqdm ---
    train_iter = tqdm(train_loader, desc="Training", leave=False)
    for batch in train_iter:
        optimizer.zero_grad()
        imu_batch = {k: stack_sensor(v) for k, v in batch["imu"].items()}
        sentence_emb = batch["sentence_embedding"].to(device).squeeze(1)
        
        sentence_out, imu_outs = model(imu_batch, sentence_emb)

        # Sentence ↔ Sensor loss
        loss = alpha * sum(loss_fn(sentence_out, imu_outs[k]) for k in imu_outs)

        # Sensor ↔ Sensor loss
        imu_keys = list(imu_outs.keys())
        for i in range(len(imu_keys)):
            for j in range(i + 1, len(imu_keys)):
                loss += beta * loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_iter.set_postfix(loss=total_loss/(train_iter.n+1))
    
    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")

    # --- Validation with tqdm ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="Validation", leave=False)
        for batch in val_iter:
            imu_batch = {k: stack_sensor(v) for k, v in batch["imu"].items()}
            sentence_emb = batch["sentence_embedding"].to(device).squeeze(1)
            
            sentence_out, imu_outs = model(imu_batch, sentence_emb)

            # Sentence ↔ Sensor loss
            batch_loss = alpha * sum(loss_fn(sentence_out, imu_outs[k]).item() for k in imu_outs)

            # Sensor ↔ Sensor loss
            imu_keys = list(imu_outs.keys())
            for i in range(len(imu_keys)):
                for j in range(i + 1, len(imu_keys)):
                    batch_loss += beta * loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]]).item()

            val_loss += batch_loss
            val_iter.set_postfix(val_loss=val_loss/(val_iter.n+1))
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # ----------------------------
    # Check Early Stopping
    # ----------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("Validation improved, best model saved to best_model.pth")

    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{patience} epochs.")
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

# Load best model after training
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
    print("Loaded best model state from disk.")
