from __future__ import annotations

import os

import torch
import torch.multiprocessing as mp
from rhtrain.utils.ddp_utils import find_free_port
from rhtrain.workers.ddp_train_worker import ddp_train_worker

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class DDPTrainer:
    """
    One-process-per-GPU rhtrain. No torchrun needed.
    - Trainer owns DDP/AMP/no_sync.
    """
    def __init__(self, config):
        assert config is not None, "config must be provided"
        self.config = config

        self.devices = config['trainer'].get('devices', [0])
        total = len(self.devices)
        world_size = total if self.devices is None else max(1, min(total, total))
        self.config['world_size'] = world_size

    def fit(self, data_module):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure a CUDA-capable device is present and PyTorch is installed with CUDA support.")

        if self.config['world_size'] == 1:

            addr, port = "127.0.0.1", find_free_port()

            ddp_train_worker(rank=0, addr=addr, port=port,
                             config=self.config, data_module=data_module)

            return

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        addr, port = os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"])

        # spawn – pass only picklable state
        mp.start_processes(
            ddp_train_worker,
            nprocs=self.config['world_size'],
            args=(addr, port, self.config, data_module),
            start_method="spawn",
            join=True,
        )

