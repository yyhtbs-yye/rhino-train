import torch
from pathlib import Path
from rhtrain.utils.working_path_utils import determine_run_folder
from rhtrain.utils.read_config_utils import load_yaml_config
from rhtrain.ddp_trainer import DDPTrainer as Trainer
from rhcore.utils.build_modules import get_class, build_module

torch.set_float32_matmul_precision('high')

def main(args):
    config = load_yaml_config(args.config)

    root_dir = Path(config['logging']['root_dir']) / config['logging']['name']
    root_dir.mkdir(parents=True, exist_ok=True)
    run_folder = determine_run_folder(root_dir)

    data_module = build_module(config['data'])

    config['run_folder'] = run_folder
    config['resume_from'] = args.resume_from

    trainer = Trainer(config=config)

    trainer.fit(data_module=data_module)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--resume_from', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    main(args)