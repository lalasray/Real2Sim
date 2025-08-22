import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.nn as nn
import torch.optim as optim

def setup():
    dist.init_process_group(backend="nccl")  # "nccl" for CUDA, "gloo" for CPU
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

class Net(nn.Module):
    def __init__(self, d=100, h=256, c=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Linear(h, c)
        )
    def forward(self, x): return self.net(x)

def get_dataloader(global_batch_size=256):
    # fake data
    X = torch.randn(50_000, 100)
    y = torch.randint(0, 10, (50_000,))
    ds = TensorDataset(X, y)

    # Distributed sampler makes per-rank shards
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    # Per-process batch size
    world_size = dist.get_world_size()
    assert global_batch_size % world_size == 0
    per_rank_bs = global_batch_size // world_size

    return DataLoader(
        ds,
        batch_size=per_rank_bs,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    ), sampler

def save_ckpt(model_ddp, opt, epoch, path):
    # on rank 0 only
    if dist.get_rank() == 0:
        torch.save({
            "model": model_ddp.module.state_dict(),  # strip DDP wrapper
            "opt": opt.state_dict(),
            "epoch": epoch
        }, path)

def main():
    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    model = Net().to(device)
    model = DDP(model, device_ids=[device], output_device=device)  # static graph → fast

    opt = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()  # mixed precision
    loss_fn = nn.CrossEntropyLoss()

    loader, sampler = get_dataloader(global_batch_size=256)
    epochs = 5

    for epoch in range(epochs):
        # ensure different shard each epoch
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()

        # simple average loss across ranks (for logging)
        loss_t = torch.tensor([running], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"epoch {epoch} | loss_sum {loss_t.item():.3f}")

        save_ckpt(model, opt, epoch, "ckpt.pt")

    cleanup()

if __name__ == "__main__":
    main()
