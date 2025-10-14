import torch

def save_state(run_folder, prefix="boat_state", boat=None, global_step=None, epoch=None):
    """
    Save the full training state including boat (containing model weights, optimizer states),
    lr_scheduler states, and training metadata.
    
    Args:
        run_folder: Path to the run folder
        boat: The boat being trained (contains models, optimizers, and lr_schedulers dictionaries)
        global_step (int, optional): Current global step
        epoch (int, optional): Current epoch
    """

    run_folder.mkdir(parents=True, exist_ok=True)

    # Determine which identifier to use in the filename
    if global_step is not None and epoch is not None:
        state_path = run_folder / f"{prefix}_step={global_step}_epoch={epoch}.pt"
    elif global_step is not None:
        state_path = run_folder / f"{prefix}_step={global_step}.pt"
    elif epoch is not None:
        state_path = run_folder / f"{prefix}_epoch={epoch}.pt"
    else:
        state_path = run_folder / f"{prefix}_latest.pt"
        print("Warning: Neither global_step nor epoch provided. Using generic filename.")
    
    # Save models individually
    model_states = {}
    for name, model in boat.models.items():
        if hasattr(model, 'state_dict'):
            model_states[name] = model.state_dict()
    
    # Save optimizers individually
    optimizer_states = {}
    for name, optimizer in boat.optimizers.items():
        if optimizer is not None:
            optimizer_states[name] = optimizer.state_dict()
    
    # Save lr_schedulers individually
    scheduler_states = {}
    for name, scheduler in boat.lr_schedulers.items():
        if scheduler is not None:
            scheduler_states[name] = scheduler.state_dict()
    
    # Prepare the state dictionary
    state = {
        'model_states': model_states,
        'optimizer_states': optimizer_states,
        'scheduler_states': scheduler_states,
    }
    
    # Add tracking variables to state
    if global_step is not None:
        state['global_step'] = global_step
    if epoch is not None:
        state['epoch'] = epoch
    
    # Save training configuration, placeholder for now
    state['trainer_config'] = {}
    
    # Save metadata about EMA if it exists
    if hasattr(boat, 'use_ema') and boat.use_ema:
        state['use_ema'] = boat.use_ema
        state['ema_decay'] = getattr(boat, 'ema_decay', 0.999)
        state['ema_start'] = getattr(boat, 'ema_start', 1000)
    
    # Save the state
    torch.save(state, state_path)
    print(f"Full training state saved to {state_path}")
    torch.save(state, run_folder / 'last.pt')
    print(f"Full training state saved to {run_folder / 'last.pt'}")

    return state_path

def strip_module_prefix(s: str) -> str:
    return s[7:] if s.startswith("module.") else s

def strip_module_from_state_dict(sd: dict) -> dict:
    return {strip_module_prefix(k): v for k, v in sd.items()}

def load_state(state_path, boat=None, strict=True):
    """
    Load the full training state including boat model weights, optimizer states,
    lr_scheduler states, and training metadata.
    
    Args:
        state_path (Path): Path to the saved state
        boat: The boat to load weights into (contains models, optimizers, and lr_schedulers dictionaries)
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys
                      returned by this module's state_dict() function
        
    Returns:
        tuple: (boat, metadata) - The boat with loaded state and training metadata
    """

    if not state_path.exists():
        raise FileNotFoundError(f"No state file found at {state_path}")
    
    # Load the state dictionary
    state = torch.load(state_path, map_location=torch.device('cpu'))
    
    # Load model weights
    if boat is not None and 'model_states' in state:
        model_states = state['model_states']
        for name, state_dict in model_states.items():
            if name in boat.models:
                boat.models[name].load_state_dict(strip_module_from_state_dict(state_dict), strict=strict)
            else:
                print(f"Warning: Model {name} in saved state not found in boat")
    
    # Load optimizer states
    if boat is not None and 'optimizer_states' in state:
        optimizer_states = state['optimizer_states']
        for name, state_dict in optimizer_states.items():
            if name in boat.optimizers and boat.optimizers[name] is not None:
                boat.optimizers[name].load_state_dict(state_dict)
            else:
                print(f"Warning: Optimizer {name} in saved state not found in boat")
    
    # Load lr_scheduler states
    if boat is not None and 'scheduler_states' in state:
        scheduler_states = state['scheduler_states']
        for name, state_dict in scheduler_states.items():
            if name in boat.lr_schedulers and boat.lr_schedulers[name] is not None:
                boat.lr_schedulers[name].load_state_dict(state_dict)
            else:
                print(f"Warning: LR Scheduler {name} in saved state not found in boat")
    
    # Load EMA settings if they exist
    if boat is not None and 'use_ema' in state:
        boat.use_ema = state['use_ema']
        if 'ema_decay' in state:
            boat.ema_decay = state['ema_decay']
        if 'ema_start' in state:
            boat.ema_start = state['ema_start']
    
    # Get training metadata
    metadata = {
        'global_step': state.get('global_step', 0),
        'epoch': state.get('epoch', 0),
    }
    
    print(f"Full training state loaded from {state_path}")
    print(f"Resuming from epoch {metadata['epoch']} and step {metadata['global_step']}")
    
    return boat, metadata