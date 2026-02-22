import os
import subprocess
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# REVISION: Environment cleanup for TPU v5litepod-8
SCRIPT_REV = "tpu-v5e-final-20260125"

def tpu_worker(index):
    """
    Runs on each of the 8 TPU cores.
    """
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    import torch_xla

    # Initialize device - this is where the hardware handshake happens
    device = torch_xla.device()
    world_size = xr.world_size()
    rank = xr.global_ordinal()

    # if rank == 0:
    print(f"✅ Success! TPU Rank {rank} initialized. World size: {world_size}", flush=True)

    # --- MODEL SETUP ---
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- DATA SETUP ---
    x = torch.randn(4096, 128)
    y = torch.randint(0, 10, (4096,))
    ds = TensorDataset(x, y)

    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        ds,
        batch_size=64,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    device_loader = pl.MpDeviceLoader(loader, device)

    # --- TRAINING LOOP ---
    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for step, (bx, by) in enumerate(device_loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()

            # Synchronizes gradients across all 8 cores
            xm.optimizer_step(optimizer)

            if step % 20 == 0:
                xm.master_print(
                    f"[Rank {rank}] Epoch {epoch} | Step {step} | Loss={loss.item():.4f}",
                    flush=True,
                )

    xm.master_print("🎉 Training finished successfully on all cores.", flush=True)


def main():
    print(f"SCRIPT_REV={SCRIPT_REV}", flush=True)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
        os.environ["TPU_UNINSTALL_TIMM_DONE"] = "1"
    except Exception as exc:
        print(f"WARNING: timm uninstall failed ({exc}); continuing.")
    # 1. GPU Fallback
    if torch.cuda.is_available():
        print("CUDA detected.")
        return
    LOCAL_TIMM = "/kaggle/input/datasets/liucong12601/timm-repos/pytorch-image-models"
    if os.path.isdir(LOCAL_TIMM):
        sys.path.insert(0, LOCAL_TIMM)
    # 2. CRITICAL: Clean Kaggle Environment
    # We remove 'local' addresses to let PJRT auto-configure the 8-core topology
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("TPU_PROCESS_COUNT", None)
    
    # 3. Import XLA Multiprocessing
    # Important: Do NOT import 'torch_xla.core.xla_model' here!
    import torch_xla.distributed.xla_multiprocessing as xmp

    # Check if we are on TPU
    is_tpu = any(k.startswith("TPU") for k in os.environ.keys())

    if is_tpu:
        print("TPU detected. Spawning 8 worker processes...", flush=True)
        # nprocs=None will correctly detect 8 chips on v3-8 or v5litepod-8
        xmp.spawn(tpu_worker, nprocs=None)
    else:
        print("No TPU found. Please enable the Accelerator in Kaggle settings.")


if __name__ == "__main__":
    start = time.time()
    try:
        main()
    except Exception as e:
        print(f"❌ Execution failed: {e}")
    finally:
        print(f"Total time: {time.time() - start:.1f}s", flush=True)