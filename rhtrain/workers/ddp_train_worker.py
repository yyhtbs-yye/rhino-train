import gc
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from rhcore.utils.build_components import get_class, build_module
from rhtrain.helpers.global_step import GlobalStep

from contextlib import nullcontext

from rhtrain.utils.ddp_utils import (
    ddp_setup_env, seed_everything,
    wrap_models_with_ddp, broadcast_module_state,
    is_rank0, is_dist_ready,
    get_primary_trainable_module,
)


def ddp_train_worker(rank, addr, port, config, data_module):
    
    ddp_setup_env(rank, config['world_size'], addr, port)

    # ----- Build boat -----
    boat_conf = config['boat']
    if 'mpath' in boat_conf:
        Boat = get_class(boat_conf['mpath'])
    else:
        Boat = get_class(boat_conf['path'], boat_conf['name'])
    
    config['rank'] = rank

    boat = Boat(config)
    
    # ----- Resume boat -----
    if config.get('resume_from') is not None:
        ckpt_path   = Path(config['resume_from'])
        boat, meta = boat.load_state(ckpt_path)
        config['trainer']['run_folder'] = Path(config['run_folder'])
        config['trainer']['global_step'] = GlobalStep(meta.get("global_step", 0))
        config['trainer']['start_epoch'] = meta.get("epoch", 0)
    else:
        config['trainer']['run_folder']  = Path(config['run_folder']) if config['run_folder'] else None
        config['trainer']['global_step'] = GlobalStep(0)
        config['trainer']['start_epoch'] = 0
    
    use_cuda = True
    device = torch.device(f"cuda:{config['trainer']['devices'][rank]}")
    if use_cuda:
        torch.cuda.set_device(device)

    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=config['world_size'])
    seed_everything(config['trainer'].get('seed', 42), rank)

    boat.attach_global_step(config['trainer']['global_step'])
    
    boat.to(device)

    wrap_models_with_ddp(boat, device, fup_by_key=config['trainer'].get('fup_by_key'), broadcast_buffers=True)

    # Identify gradient-free modules as EMA-like (or mark them in your code with a flag)
    ema_keys = [
        k for k, m in boat.models.items()
        if isinstance(m, torch.nn.Module) and not any(p.requires_grad for p in m.parameters()) and 'ema' in k
    ]

    callbacks = []
    # Build utils for GPU-0 Monitoring
    if is_rank0():
        callbacks   = [build_module(cb) for cb in (config.get('callbacks', []))]

    config['trainer']['valid_epoch_records'] = {}

    primary = get_primary_trainable_module(boat)

    # Optionally initialize EMA from primary, then broadcast once so every rank matches
    if ema_keys and primary is not None and is_dist_ready():
        if is_rank0():
            src = primary.module if isinstance(primary, DDP) else primary
            for k in ema_keys:
                try:
                    # params only; optionally also copy buffers to start identical
                    boat.models[k].load_state_dict(src.state_dict(), strict=False)
                except Exception:
                    pass
        for k in ema_keys:
            broadcast_module_state(boat.models[k], src_rank=0)

    precision = config['trainer'].get('precision', None)
    # AMP context + scaler
    if precision is None:
        amp_dtype = None
    elif precision == "bf16-mixed":
        amp_dtype = torch.bfloat16
    elif precision == "16-mixed":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    if amp_dtype is not None and device.type == "cuda":
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = lambda: nullcontext()

    if precision is not None:
        scaler = torch.amp.GradScaler(enabled=(precision=="16-mixed"))
    else:
        scaler = None
    
    # Dataloaders
    train_loader = data_module.make_train_loader(config['world_size'], rank)
    valid_loader = data_module.make_valid_loader(config['world_size'], rank)
    
    if config['trainer'].get('validate_before_training', False):
        target_metric = run_validation(boat, valid_loader, config, -1)

    # Train
    for epoch in range(config['trainer']['start_epoch'], config['trainer'].get("max_epochs", 10)):
        
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if is_rank0():
            for cb in callbacks: cb.on_epoch_start(config['trainer'], boat, epoch)

        run_train(boat, train_loader, config, epoch, autocast_ctx, scaler)

        if epoch % config['trainer']['val_check_epochs'] == 0:
            target_metric = run_validation(boat, valid_loader, config, epoch)
            config['trainer']['valid_epoch_records'][epoch] = {'target_metric': target_metric.detach().cpu()}

            if is_rank0():
                if config['trainer']['state_save_epochs'] is not None and epoch % config['trainer']['state_save_epochs'] == 0:
                    state_path = boat.save_state(config['trainer']['run_folder'], 'boat_state', global_step=config['trainer']['global_step'](), epoch=epoch+1)
                    
                    if epoch not in config['trainer']['valid_epoch_records']:
                        config['trainer']['valid_epoch_records'][epoch] = {}
                    config['trainer']['valid_epoch_records'][epoch]['state_path'] = state_path

        if ('offline_evaluation' in config
                and epoch % config['trainer']['eval_check_epochs'] == 0):
            run_offline_evaluation(boat, valid_loader, config)

        # Sync EMA after each epoch
        if ema_keys and primary is not None:
            for k in ema_keys:
                broadcast_module_state(boat.models[k], src_rank=0)

        if is_rank0():
            for cb in callbacks: cb.on_epoch_end(config['trainer'], boat, epoch)

    if is_rank0():
        for cb in callbacks: cb.on_train_end(config['trainer'], boat)

    dist.destroy_process_group()


def _format_eta(seconds):
    if seconds is None:
        return "N/A"
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_train(boat, dataloader, config, epoch, autocast_ctx, scaler):

    boat.train()

    dataset_length = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(dataloader)
    batch_size = dataloader.batch_size if hasattr(dataloader, 'batch_size') else None

    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
        steps_per_epoch = None

    step = 0
    epoch_start_time = time.time()

    for batch_idx_batch in dataloader:

        step += 1

        if isinstance(batch_idx_batch, tuple):
            batch_idx, batch = batch_idx_batch
        else:
            batch_idx, batch = step - 1, batch_idx_batch

        # bump global step
        if boat.get_global_step() > 0 and epoch == 0 and batch_idx == 0:
            boat.global_step -= 1
        boat.global_step += 1

        # no_sync across all DDP modules until boundary; AMP is trainer-owned
        with autocast_ctx():
            losses = boat.training_all(batch, batch_idx, epoch, scaler=scaler)

        if is_rank0() and losses:
            boat.take_a_log(losses, 'train')
            eta_seconds = None
            if steps_per_epoch is not None and step > 0:
                elapsed = time.time() - epoch_start_time
                avg_step = elapsed / step
                remaining_steps = max(steps_per_epoch - step, 0)
                eta_seconds = remaining_steps * avg_step
            eta_str = _format_eta(eta_seconds)
            print(
                f"Training batch index: {batch_idx * batch_size * config['world_size']} / {dataset_length}, "
                f"epoch: {epoch}, step: {step}, global_step: {boat.get_global_step()}, "
                f"eta_epoch: {eta_str}"
            )


def run_validation(boat, dataloader, config, epoch):

    boat.eval()
    aggr_metrics = {}

    dataset_length = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(dataloader)

    # Get Batch size from dataloader
    batch_size = dataloader.batch_size if hasattr(dataloader, 'batch_size') else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            metrics, named_imgs = boat.validation_step(batch, batch_idx, epoch)
            for key, value in metrics.items():
                if key not in aggr_metrics:
                    aggr_metrics[key] = metrics[key]
                else:
                    aggr_metrics[key] += metrics[key]
            if is_rank0():
                boat.save_images(named_imgs, batch_idx)
                print(f"Validation Completion: {batch_idx * batch_size * config['world_size']} / {dataset_length}, validation over {batch_idx + 1} batches done at global_step {boat.get_global_step()}")

            # help GC
            del batch, metrics, named_imgs
            gc.collect()
            torch.cuda.empty_cache()

        for key in aggr_metrics:
            aggr_metrics[key] /= (batch_idx + 1)

    if not aggr_metrics: raise ValueError("Validation loop produced no losses.")
    
    # DDP all-reduce and average all metrics
    for key in aggr_metrics:
        aggr_metrics[key] = aggr_metrics[key].to(boat.device)
        dist.all_reduce(aggr_metrics[key], op=dist.ReduceOp.SUM)
        aggr_metrics[key] /= dist.get_world_size()
        aggr_metrics[key] = aggr_metrics[key]

    if is_rank0():
        boat.take_a_log(aggr_metrics, 'valid')

    target_metric_name = config['validation']['target_metric_name']

    if target_metric_name not in aggr_metrics:
        raise KeyError(f"'{target_metric_name}' not found in validation metrics: {list(aggr_metrics)}")

    # after loop:
    gc.collect()
    torch.cuda.empty_cache()

    return aggr_metrics[target_metric_name]


def run_offline_evaluation(boat, dataloader, config):
    boat.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            boat.offline_evaluation_iter(batch, batch_idx)

    # waiting all rank to finish. 
    if is_dist_ready():
        dist.barrier()

    if is_rank0():
        offline_metrics = boat.offline_evaluation_final()
        if offline_metrics is not None and len(offline_metrics) > 0:
            boat.take_a_log(offline_metrics, 'eval')

    if is_dist_ready():
        dist.barrier()

    # after loop:
    gc.collect()
    torch.cuda.empty_cache()

    return None
