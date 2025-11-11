import torch
from rhtrain.utils.load_save_utils import strip_module_from_state_dict
from pathlib import Path

state_path = '/home/yyh/python_workspaces/rhino-compression/work_dirs/sdvae_ffhq_256/run_4/last.pt'
pretrained_models_dir = Path('pretrained/sdvae_ffhq_256_pretrained_models')

# Load the state dictionary
state = torch.load(state_path, map_location=torch.device('cpu'))

if 'model_states' in state:
    model_states = state['model_states']
    for name, state_dict in model_states.items():
        # Save the stripped state dict to pretrained_models
        stripped_state_dict = strip_module_from_state_dict(state_dict)
        save_path = pretrained_models_dir / f"{name}_pretrained.pt"
        torch.save(stripped_state_dict, save_path)
        print(f"Saved pretrained model {name} to {save_path}")