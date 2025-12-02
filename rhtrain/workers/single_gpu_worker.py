from pathlib import Path
from contextlib import nullcontext

import torch
from torch.utils.data import DistributedSampler  # ok if your datamodule returns one; works with num_replicas=1

from rhcore.utils.build_components import get_class, build_module
from rhtrain.helpers.global_step import GlobalStep
from rhtrain.utils.ddp_utils import (
    seed_everything,
    get_primary_trainable_module,  # still useful to locate the main trainable module
)

def train_single_gpu(config, data_module):
    """
    Single-GPU trainer equivalent of `ddp_train_worker`.

    Expected `config` keys (same as before, minus DDP):
      - boat: { path/name or mpath }
      - trainer: {
            devices: [gpu_index],  # optional; defaults to 0 if CUDA is available
            seed, max_epochs, val_check_epochs, state_save_epochs,
            run_folder, (optional) precision: "bf16-mixed" | "16-mixed" | None,
        }
      - validation: { target_metric_name }
      - resume_from: (optional checkpoint path string)
    The `data_module` must implement:
        make_train_loader(world_size:int, rank:int)
        make_valid_loader(world_size:int, rank:int)
    """

    # ----- Single-GPU setup -----
    config = dict(config)  # shallow copy in case caller reuses it
    config['world_size'] = 1
    config['rank'] = 0

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_idx = (config.get('trainer', {}).get('devices') or [0])[0]
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed_everything(config.get('trainer', {}).get('seed', 42), rank=0)

    # ----- Build boat -----
    boat_conf = config['boat']
    if 'mpath' in boat_conf:
        Boat = get_class(boat_conf['mpath'])
    else:
        Boat = get_class(boat_conf['path'], boat_conf['name'])
    boat = Boat(config)

    # ----- Resume or initialize -----
    if config.get('resume_from'):
        ckpt_path = Path(config['resume_from'])
        boat, meta = boat.load_state(ckpt_path)
        config['trainer']['run_folder'] = Path(config['run_folder']) if config.get('run_folder') else None
        config['trainer']['global_step'] = GlobalStep(meta.get("global_step", 0))
        config['trainer']['start_epoch'] = meta.get("epoch", 0)
    else:
        config['trainer']['run_folder'] = Path(config['run_folder']) if config.get('run_folder') else None
        config['trainer']['global_step'] = GlobalStep(0)
        config['trainer']['start_epoch'] = 0

    boat.attach_global_step(config['trainer']['global_step'])
    boat.to(device)

    # ----- Optional EMA bootstrap (no broadcast needed on single GPU) -----
    ema_keys = [
        k for k, m in getattr(boat, "models", {}).items()
        if isinstance(m, torch.nn.Module) and not any(p.requires_grad for p in m.parameters()) and 'ema' in k
    ]
    primary = get_primary_trainable_module(boat)
    if ema_keys and primary is not None:
        src = primary
        for k in ema_keys:
            try:
                boat.models[k].load_state_dict(src.state_dict(), strict=False)
            except Exception:
                pass  # non-strict copy; ignore missing keys

    # ----- Precision / AMP -----
    precision = config['trainer'].get('precision')
    if precision == "bf16-mixed":
        amp_dtype = torch.bfloat16
    elif precision == "16-mixed":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    if amp_dtype is not None and device.type == "cuda":
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = lambda: nullcontext()

    scaler = torch.amp.GradScaler(enabled=(precision == "16-mixed")) if device.type == "cuda" else None

    # ----- Dataloaders (world_size=1, rank=0) -----
    train_loader = data_module.make_train_loader(world_size=1, rank=0)
    valid_loader = data_module.make_valid_loader(world_size=1, rank=0)

    # ----- Callbacks (always active on single GPU) -----
    callbacks = [build_module(cb) for cb in (config.get('callbacks', []))]
    config['trainer']['valid_epoch_records'] = {}

    # ----- Train loop -----
    start_epoch = config['trainer']['start_epoch']
    max_epochs = config['trainer'].get("max_epochs", 10)
    val_check_epochs = config['trainer'].get("val_check_epochs", 1)
    state_save_epochs = config['trainer'].get("state_save_epochs")

    for epoch in range(start_epoch, max_epochs):
        # If the dataloader gives a DistributedSampler with num_replicas=1, we can still set its epoch
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for cb in callbacks:
            cb.on_epoch_start(config['trainer'], boat, epoch)

        run_train(boat, train_loader, config, epoch, autocast_ctx, scaler)

        if epoch % val_check_epochs == 0:
            target_metric = run_validation(boat, valid_loader, config, epoch)
            config['trainer']['valid_epoch_records'][epoch] = {'target_metric': target_metric.detach().cpu()}

            if state_save_epochs is not None and epoch % state_save_epochs == 0:
                state_path = boat.save_state(config['trainer']['run_folder'], 'boat_state', global_step=config['trainer']['global_step'](), epoch=epoch+1)
                
                if epoch not in config['trainer']['valid_epoch_records']:
                    config['trainer']['valid_epoch_records'][epoch] = {}
                config['trainer']['valid_epoch_records'][epoch]['state_path'] = state_path

        for cb in callbacks:
            cb.on_epoch_end(config['trainer'], boat, epoch)

    for cb in callbacks:
        cb.on_train_end(config['trainer'], boat)


def run_train(boat, train_loader, config, epoch, autocast_ctx, scaler):
    boat.train()

    dataset_length = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader)
    batch_size = getattr(train_loader, 'batch_size', None)
    
    step = 0

    for batch_idx_batch in train_loader:
        step += 1
        # dataloader may yield either (batch_idx, batch) or just batch
        if isinstance(batch_idx_batch, tuple):
            batch_idx, batch = batch_idx_batch
        else:
            batch_idx, batch = step - 1, batch_idx_batch

        # bump global step (mirrors original behavior)
        if boat.get_global_step() > 0 and epoch == 0 and batch_idx == 0:
            boat.global_step -= 1
        boat.global_step += 1

        with autocast_ctx():
            losses = boat.training_all(batch, batch_idx, epoch, scaler=scaler)

        if losses:
            boat.take_a_log(losses, 'train')
            if batch_size is not None:
                progressed = batch_idx * batch_size  # world_size=1
                print(f"Training batch index: {progressed} / {dataset_length}, epoch: {epoch}, step: {step}, global_step: {boat.get_global_step()}")
            else:
                print(f"Training: epoch={epoch}, step={step}, global_step={boat.get_global_step()}")


def run_validation(boat, val_dataloader, config, epoch):
    boat.eval()
    aggr_metrics = {}

    dataset_length = len(val_dataloader.dataset) if hasattr(val_dataloader, 'dataset') else len(val_dataloader)
    batch_size = getattr(val_dataloader, 'batch_size', None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            metrics, named_imgs = boat.validation_step(batch, batch_idx, epoch)

            for key, value in metrics.items():
                if key not in aggr_metrics:
                    aggr_metrics[key] = value
                else:
                    aggr_metrics[key] += value

            boat.save_images(named_imgs, batch_idx)
            if batch_size is not None:
                progressed = batch_idx * batch_size  # world_size=1
                print(f"Validation Completion: {progressed} / {dataset_length}, validation over {batch_idx + 1} batches done at global_step {boat.get_global_step()}")
            else:
                print(f"Validation: batch {batch_idx + 1}, global_step {boat.get_global_step()}")

        # Average across batches (single GPU, no all-reduce)
        if not aggr_metrics:
            raise ValueError("Validation loop produced no losses.")

        for key in aggr_metrics:
            aggr_metrics[key] /= (batch_idx + 1)

    boat.take_a_log(aggr_metrics, 'valid')

    target_metric_name = config['validation']['target_metric_name']
    if target_metric_name not in aggr_metrics:
        raise KeyError(f"'{target_metric_name}' not found in validation metrics: {list(aggr_metrics)}")

    return aggr_metrics[target_metric_name]
