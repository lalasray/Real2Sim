import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, random_split, DistributedSampler
from tqdm import tqdm
from windowed_loader import IMUSlidingWindowDataset, imu_collate_fn
from info_nce import InfoNCE
from model import IMUModel, stack_sensor
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

# ----------------------------
# DDP helpers
# ----------------------------
def setup_ddp(backend="nccl"):
    # torchrun sets these env vars
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(local_rank)  # one process -> one GPU
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def seed_everything(base_seed: int, rank: int):
    torch.manual_seed(base_seed + rank)
    torch.cuda.manual_seed_all(base_seed + rank)
    random.seed(base_seed + rank)

# ----------------------------
# Main
# ----------------------------
def main():
    # <<< DDP: init
    rank, local_rank, world_size = setup_ddp()
    seed_everything(42, rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print("Using device:", device, "| world_size:", world_size)

    # ----------------------------
    # Dataset (same on every rank)
    # ----------------------------
    folder = "/mimer/NOBACKUP/groups/focs/datasets/MOTIONX/processed_dataset/processed_dataset"
    window_sizes = [30, 60, 90, 120, 150,180, 210, 140, 270, 300]
    #window_sizes = [300]
    datasets = [IMUSlidingWindowDataset(folder, window_size=ws, stride=10) for ws in window_sizes]
    merged_dataset = ConcatDataset(datasets)

    total_len = len(merged_dataset)
    # Take 50% of the data for debugging
    debug_len = int(0.5 * total_len)
    g = torch.Generator().manual_seed(1234)
    debug_dataset, _ = random_split(merged_dataset, [debug_len, total_len - debug_len], generator=g)

    # Out of this 50%, take 5% for validation
    val_len = int(0.05 * debug_len)
    train_len = debug_len - val_len
    train_dataset, val_dataset = random_split(debug_dataset, [train_len, val_len], generator=g)

    # <<< DDP: each loader gets a DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # Note: when using a sampler, DO NOT also set shuffle=True in DataLoader.
    num_workers = 4
    # Set the batch size PER GPU (per process)
    bs = int(900 / world_size)  # This is per-GPU batch size. Total effective batch size = bs * world_size

    train_loader = DataLoader(
        train_dataset, batch_size=bs, sampler=train_sampler,
        collate_fn=imu_collate_fn, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, sampler=val_sampler,
        collate_fn=imu_collate_fn, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=4
    )



    # ----------------------------
    # Initialize model, loss, optimizer
    # ----------------------------
    model = IMUModel().to(device)
    # Optional speedups:
    # torch.backends.cudnn.benchmark = True
    # model = torch.compile(model)  # PyTorch 2.x

    # <<< DDP: wrap
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    loss_fn = InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    alpha = 1.0  # sentence ↔ sensor
    beta = 0.5   # sensor ↔ sensor

    # <<< DDP: log per-rank (either only rank 0, or separate dirs). Here: rank 0 only.
    if rank == 0:
        log_name = f"Sim2Real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=f"/mimer/NOBACKUP/groups/focs/TensorboardLogs/{log_name}")
    else:
        writer = None

    # ----------------------------
    # Early Stopping (managed on rank 0)
    # ----------------------------
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # ----------------------------
    # Training Loop
    # ----------------------------
    epochs = 1000
    global_step_training = 0
    global_step_validation = 0

    #scaler = torch.cuda.amp.GradScaler()  # AMP

    for epoch in range(epochs):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")

        # <<< DDP: reshuffle shards each epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_alpha_loss = 0.0
        total_beta_loss  = 0.0

        # Progress bar only on rank 0
        train_iter = train_loader if rank != 0 else tqdm(train_loader, desc="Training", leave=False)
        for batch in train_iter:
            optimizer.zero_grad(set_to_none=True)

            imu_batch = {}
            for k, v in batch["imu"].items():
                accel, gyro, mask, lengths = stack_sensor(v)
                # <<< ensure tensors on the right device (adjust if your model moves them internally)
                accel   = accel.to(device, non_blocking=True)
                gyro    = gyro.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                imu_batch[k] = (accel, gyro, lengths)

            sentence_emb = batch["sentence_embedding"].to(device, non_blocking=True).squeeze(1)

            sentence_out, imu_outs = model(imu_batch, sentence_emb)

            alpha_loss_val = sum(loss_fn(sentence_out, imu_outs[k]) for k in imu_outs)
            loss = alpha * alpha_loss_val

            # Sensor ↔ Sensor loss
            beta_loss_val = 0.0
            imu_keys = list(imu_outs.keys())
            for i in range(len(imu_keys)):
                for j in range(i + 1, len(imu_keys)):
                    pair_loss = loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]])
                    loss += beta * pair_loss
                    beta_loss_val += pair_loss.detach().item()

            if torch.isnan(loss):
                if rank == 0:
                    print("NaN detected in loss; skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.detach().item()
            total_alpha_loss += alpha_loss_val.detach().item()
            total_beta_loss += beta_loss_val

            if rank == 0:
                # Avoid very frequent logging; log every ~100 steps
                if (global_step_training % 100 == 0) and writer is not None:
                    writer.add_scalar("Loss/alpha_loss", alpha_loss_val.item(), global_step_training)
                    writer.add_scalar("Loss/beta_loss",  beta_loss_val,          global_step_training)
                    writer.add_scalar("Loss/total_loss", loss.item(),            global_step_training)
            global_step_training += 1

        # ---- Aggregate train metrics across ranks for printing ----
        # Sum totals and counts then compute global averages
        totals = torch.tensor([total_loss, total_alpha_loss, total_beta_loss, len(train_loader)], dtype=torch.float32, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        avg_loss = (totals[0] / totals[3]).item()
        avg_alpha_loss = (totals[1] / totals[3]).item()
        avg_beta_loss = (totals[2] / totals[3]).item()

        if rank == 0:
            print(f"Train Loss: {avg_loss:.4f}")
            print(f"Average Alpha Loss: {avg_alpha_loss:.4f}")
            print(f"Average Beta Loss: {avg_beta_loss:.4f}")
            if writer is not None:
                writer.add_scalar("Loss/avg_alpha_loss", avg_alpha_loss, epoch)
                writer.add_scalar("Loss/avg_beta_loss",  avg_beta_loss,  epoch)
            path = f"/mimer/NOBACKUP/groups/focs/models/sim2real_multi/training/ckpt_epoch.pth"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "model": model.module.state_dict(),   # strip DDP wrapper
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, path)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        total_alpha_loss_val = 0.0
        total_beta_loss_val = 0.0

        with torch.no_grad():
            val_iter = val_loader if rank != 0 else tqdm(val_loader, desc="Validation", leave=False)
            for batch in val_iter:
                imu_batch = {}
                for k, v in batch["imu"].items():
                    accel, gyro, mask, lengths = stack_sensor(v)
                    accel   = accel.to(device, non_blocking=True)
                    gyro    = gyro.to(device, non_blocking=True)
                    lengths = lengths.to(device, non_blocking=True)
                    imu_batch[k] = (accel, gyro, lengths)

                sentence_emb = batch["sentence_embedding"].to(device, non_blocking=True).squeeze(1)

                sentence_out, imu_outs = model(imu_batch, sentence_emb)
                alpha_loss_batch = sum(loss_fn(sentence_out, imu_outs[k]) for k in imu_outs)
                batch_loss = alpha * alpha_loss_batch

                # Sensor ↔ Sensor during val
                beta_loss_batch = 0.0
                imu_keys = list(imu_outs.keys())
                for i in range(len(imu_keys)):
                    for j in range(i + 1, len(imu_keys)):
                        pair_loss = loss_fn(imu_outs[imu_keys[i]], imu_outs[imu_keys[j]])
                        batch_loss += beta * pair_loss
                        beta_loss_batch += pair_loss.detach().item()

                if torch.isnan(batch_loss):
                    continue

                val_loss += batch_loss.detach().item()
                total_alpha_loss_val += alpha_loss_batch.detach().item()
                total_beta_loss_val += beta_loss_batch

                if rank == 0 and writer is not None and (global_step_validation % 200 == 0):
                    writer.add_scalar("Val_Loss/alpha_loss", alpha_loss_batch.item(), global_step_validation)
                    writer.add_scalar("Val_Loss/beta_loss",  beta_loss_batch,        global_step_validation)
                    writer.add_scalar("Val_Loss/total_loss", batch_loss.item(),      global_step_validation)
                global_step_validation += 1

        # ---- Aggregate val metrics across ranks ----
        vtotals = torch.tensor([val_loss, total_alpha_loss_val, total_beta_loss_val, len(val_loader)],
                               dtype=torch.float32, device=device)
        dist.all_reduce(vtotals, op=dist.ReduceOp.SUM)
        avg_val_loss = (vtotals[0] / vtotals[3]).item()
        avg_alpha_loss_val = (vtotals[1] / vtotals[3]).item()
        avg_beta_loss_val = (vtotals[2] / vtotals[3]).item()

        if rank == 0:
            if writer is not None:
                writer.add_scalar("Val_Loss/avg_alpha_loss", avg_alpha_loss_val, epoch)
                writer.add_scalar("Val_Loss/avg_beta_loss",  avg_beta_loss_val,  epoch)
            print(f"Validation Loss: {avg_val_loss:.4f}")



        # ----------------------------
        # Early Stopping + Save (rank 0 only)
        # ----------------------------
        save_path = f"/mimer/NOBACKUP/groups/focs/models/sim2real_multi/"
        if rank == 0:
            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                os.makedirs(save_path, exist_ok=True)
                # unwrap DDP for saving
                torch.save(model.module.state_dict(), os.path.join(save_path, "best_model.pth"))
                print(f"Epoch {epoch+1}: Validation improved to {avg_val_loss:.4f}. Saved best_model.pth")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{patience} epochs.")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    # Broadcast stop signal to all ranks
                    stop = torch.tensor([1], device=device)
                    dist.broadcast(stop, src=0)
                    break
            # Tell others to continue
            stop = torch.tensor([0], device=device)
            dist.broadcast(stop, src=0)
        else:
            # non-zero ranks receive stop/continue
            stop = torch.tensor([0], device=device)
            dist.broadcast(stop, src=0)
            if stop.item() == 1:
                break

    # Load best model (rank 0 saves; all ranks can optionally load from disk if needed)
    if rank == 0:
        best_path = os.path.join("/mimer/NOBACKUP/groups/focs/models/sim2real_multi/", "best_model.pth")
        if os.path.exists(best_path):
            print("Best model is saved at:", best_path)

    cleanup_ddp()

if __name__ == "__main__":
    # Ensure safe DataLoader start method in DDP
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
