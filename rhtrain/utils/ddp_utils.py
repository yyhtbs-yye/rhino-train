import os, socket, random
from contextlib import ExitStack, nullcontext

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def is_dist_ready():
    return dist.is_available() and dist.is_initialized()

@torch.no_grad()
def broadcast_module_state(module: torch.nn.Module, src_rank: int = 0):
    """Broadcast *all* tensors in state_dict (params + buffers) from src_rank."""
    if not is_dist_ready():
        return
    for t in module.state_dict().values():
        if torch.is_tensor(t):
            dist.broadcast(t, src_rank)

@torch.no_grad()
def copy_buffers(dst: torch.nn.Module, src: torch.nn.Module):
    """Optional: keep non-learnable buffers (e.g., BN running stats) matched to src."""
    for (n1, b1), (n2, b2) in zip(dst.named_buffers(), src.named_buffers(), strict=False):
        if torch.is_tensor(b1) and torch.is_tensor(b2) and b1.shape == b2.shape:
            b1.copy_(b2)

def get_primary_trainable_module(boat):
    # Prefer a DDP-wrapped model if present, else any trainable module
    for m in boat.models.values():
        if isinstance(m, DDP):
            return m
    for m in boat.models.values():
        if isinstance(m, torch.nn.Module) and any(p.requires_grad for p in m.parameters()):
            return m
    return None

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def ddp_setup_env(rank, world_size, addr, port):
    os.environ.setdefault("MASTER_ADDR", addr)
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

def seed_everything(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_no_sync_all(boat, enabled: bool):
    """Enter no_sync() on all DDP-wrapped modules if enabled."""
    if not enabled:
        return nullcontext()
    stack = ExitStack()
    for m in boat.models.values():
        if isinstance(m, DDP):
            stack.enter_context(m.no_sync())
    return stack

def wrap_models_with_ddp(boat, device, *, fup_by_key = None, broadcast_buffers=True):
    """Wrap each trainable nn.Module in boat.models[...] with DDP."""
    fup_by_key = fup_by_key or {}
    for k, module in list(boat.models.items()):
        if not isinstance(module, torch.nn.Module):
            continue
        module.to(device)
        # If nothing requires grad (e.g., EMA), don't wrap in DDP.
        if not any(p.requires_grad for p in module.parameters()):
            boat.models[k] = module
            continue

        ddp = DDP(
            module,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=bool(fup_by_key.get(k, False)),  # default False for perf
            broadcast_buffers=broadcast_buffers,
        )
        boat.models[k] = ddp

def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    if hasattr(batch, "to"):
        return batch.to(device, non_blocking=True)
    return batch

def get_raw_module(m):
    return m.module if isinstance(m, DDP) else m