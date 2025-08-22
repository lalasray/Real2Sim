import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
from windowed_loader import IMUSlidingWindowDataset, imu_collate_fn
from info_nce import InfoNCE
from model import IMUModel, stack_sensor
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ----------------------------
# Dataset
# ----------------------------
folder = "/mimer/NOBACKUP/groups/focs/datasets/MOTIONX/processed_dataset"
# window_sizes = [30, 60, 90, 120, 150,180, 210, 140, 270, 300]
window_sizes = [300]

datasets = [IMUSlidingWindowDataset(folder, window_size=ws, stride=10) for ws in window_sizes]
merged_dataset = ConcatDataset(datasets)

total_len = len(merged_dataset)
train_len = int(0.9 * total_len)
val_len = total_len - train_len
train_dataset, val_dataset = random_split(merged_dataset, [train_len, val_len])

train_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, collate_fn=imu_collate_fn, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=imu_collate_fn, num_workers=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Initialize model, loss, optimizer
# ----------------------------
model = IMUModel().to(device)
loss_fn = InfoNCE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

alpha = 1.0  # sentence ↔ sensor
beta = 0.5   # sensor ↔ sensor
log_name = f"Sim2Real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"/mimer/NOBACKUP/groups/focs/TensorboardLogs/{log_name}")
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
globale_step = 0
global_step_training = 0
global_step_validation = 0
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0.0
    total_alpha_loss = 0.0
    total_beta_loss  = 0.0
    
    train_iter = tqdm(train_loader, desc="Training", leave=False)
    for batch in train_iter:
        optimizer.zero_grad()
        imu_batch = {}

        for k, v in batch["imu"].items():
            accel, gyro, mask, lengths = stack_sensor(v)
            imu_batch[k] = (accel, gyro, lengths)
        
        sentence_emb = batch["sentence_embedding"].to(device).squeeze(1)
        sentence_out, imu_outs = model(imu_batch, sentence_emb)

        alpha_loss = sum(loss_fn(sentence_out, imu_outs[k]) for k in imu_outs)

        # Sentence ↔ Sensor loss
        loss = alpha * sum(loss_fn(sentence_out, imu_outs[k]) for k in imu_outs)

        writer.add_scalar("Loss/alpha_loss", alpha_loss.item(),global_step_training)




        # Sensor ↔ Sensor loss
        beta_loss = 0.0
        imu_keys = list(imu_outs.keys())
        for i in range(len(imu_keys)):
            for j in range(i + 1, len(imu_keys)):
                loss += beta * loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]])
                beta_loss += loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]]).item()
        writer.add_scalar("Loss/beta_loss", beta_loss,global_step_training)
        writer.add_scalar("Loss/total_loss", loss.item(),global_step_training)

        # NaN check before backward
        if torch.isnan(loss):
            print("NaN detected in loss")
            for k, v in imu_outs.items():
                print(f"{k} max: {v.max()}, min: {v.min()}, mean: {v.mean()}")
            print(f"sentence_out max: {sentence_out.max()}, min: {sentence_out.min()}, mean: {sentence_out.mean()}")
            continue  # skip this batch

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        total_alpha_loss += alpha_loss.item()
        total_beta_loss += beta_loss
        train_iter.set_postfix(loss=total_loss / (train_iter.n + 1))
        global_step_training += 1 
    
    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    avg_alpha_loss = total_alpha_loss / len(train_loader)
    avg_beta_loss = total_beta_loss / len(train_loader)
    print(f"Average Alpha Loss: {avg_alpha_loss:.4f}")
    print(f"Average Beta Loss: {avg_beta_loss:.4f}")
    writer.add_scalar("Loss/avg_alpha_loss", avg_alpha_loss, epoch)
    writer.add_scalar("Loss/avg_beta_loss", avg_beta_loss, epoch)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    total_alpha_loss_val = 0.0
    total_beta_loss_val = 0.0
    similarity_scores = {}
    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="Validation", leave=False)
        for batch in val_iter:
            imu_batch = {}
            for k, v in batch["imu"].items():
                accel, gyro, mask, lengths = stack_sensor(v)
                imu_batch[k] = (accel, gyro, lengths)
            
            sentence_emb = batch["sentence_embedding"].to(device).squeeze(1)
            sentence_out, imu_outs = model(imu_batch, sentence_emb)

            alpha_loss = sum(loss_fn(sentence_out, imu_outs[k]) for k in imu_outs)
            batch_loss = alpha * alpha_loss


            def cosine_similarity(a, b):
                a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
                b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
                return (a_norm * b_norm).sum(dim=1).mean().item()

            # Track cosine similarity for each k
            cosine_sims = {k: cosine_similarity(sentence_out, imu_outs[k]) for k in imu_outs}
            for k, sim in cosine_sims.items():
                writer.add_scalar(f"Val_CosineSimilarity/{k}", sim,global_step_validation)
                if k not in similarity_scores:
                    similarity_scores[k] = []
                similarity_scores[k].append(sim)

            writer.add_scalar("Val_Loss/alpha_loss", alpha_loss.item(),global_step_validation)
            imu_keys = list(imu_outs.keys())
            beta_loss = 0.0
            for i in range(len(imu_keys)):
                for j in range(i + 1, len(imu_keys)):
                    batch_loss += beta * loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]])
                    beta_loss += loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]]).item()
            writer.add_scalar("Val_Loss/beta_loss", beta_loss,global_step_validation)
            writer.add_scalar("Val_Loss/total_loss", batch_loss.item(),global_step_validation)
            
            if torch.isnan(batch_loss):
                print("NaN detected in validation loss")
                continue

            val_loss += batch_loss.item()
            total_alpha_loss_val += alpha_loss.item()
            total_beta_loss_val += beta_loss
            val_iter.set_postfix(val_loss=val_loss / (val_iter.n + 1))
            global_step_validation += 1  # advance after each batch
    
    avg_val_loss = val_loss / len(val_loader)
    avg_alpha_loss_val = total_alpha_loss_val / len(val_loader)
    avg_beta_loss_val = total_beta_loss_val / len(val_loader)
    writer.add_scalar("Val_Loss/avg_alpha_loss", avg_alpha_loss_val, epoch)
    writer.add_scalar("Val_Loss/avg_beta_loss", avg_beta_loss_val, epoch)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    # Calculate and log average cosine similarity per sensor key
    for k, sims in similarity_scores.items():
        avg_sim = sum(sims) / len(sims)
        writer.add_scalar(f"Val_CosineSimilarity/avg_{k}", avg_sim, epoch)



    save_path = f"/mimer/NOBACKUP/groups/focs/models/sim2real/"

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        print(f"Epoch {epoch+1}: Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}.")
        print(f"Model saved to {os.path.join(save_path, 'best_model.pth')}")
        print("Validation improved, best model saved to best_model.pth")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{patience} epochs.")
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

# Load best model
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
    print("Loaded best model state from disk.")
