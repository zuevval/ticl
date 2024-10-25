import socket
import sys
import time

import mlflow
import wandb
import torch
import os

from git import Repo

from mothernet.model_builder import get_model
from mothernet.utils import init_device, get_model_string, synetune_handle_checkpoint, make_training_callback
from mothernet.config_utils import compare_dicts, flatten_dict, update_config
from mothernet.cli_parsing import make_model_level_argparser
from mothernet.model_configs import get_model_default_config
from argparse import Namespace


def main(argv, extra_config=None):
    # extra config is used for testing purposes only
    # this is the generic entry point for training any model, so it has A LOT of options
    parser = make_model_level_argparser()
    args = parser.parse_args(args=argv or ['--help'])
    config = get_model_default_config(args.model_type)

    device, rank, num_gpus = init_device(args.general.gpu_id, args.general.use_cpu)
    # handle syne-tune restarts
    orchestration = args.orchestration
    orchestration.base_path, orchestration.continue_run, orchestration.warm_start_from, report = synetune_handle_checkpoint(orchestration)

    if orchestration.create_new_run and not orchestration.continue_run:
        raise ValueError("Specifying create-new-run makes no sense when not continuing run")
    base_path = orchestration.base_path
    torch.set_num_threads(24)
    for group_name in vars(args):
        if group_name == "model_type":
            # the only non-group argument from the top level parser
            config['model_type'] = args.model_type
            continue
        if group_name not in config:
            config[group_name] = {}
        for k, v in vars(getattr(args, group_name)).items():
            if isinstance(v, Namespace):
                if k not in config[group_name]:
                    config[group_name][k] = {}
                # FIXME we only allow one level of nesting, we should do recursion here really.
                config[group_name][k].update(vars(v))
            else:
                config[group_name][k] = v
        config[group_name].update()
    if args.orchestration.seed_everything:
        import lightning as L
        L.seed_everything(42)

    # promote general group to top level
    config.update(config.pop('general'))
    config['num_gpus'] = 1
    config['device'] = device

    if not config['transformer']['classification_task']:
        print('Setting regression parameters')
        config['prior']['classification']['max_num_classes'] = 0
        config['transformer']['y_encoder'] = 'linear'
        config['mothernet']['decoder_type'] = 'average'

    warm_start_weights = orchestration.warm_start_from
    config['transformer']['nhead'] = config['transformer']['emsize'] // 128

    config['dataloader']['num_steps'] = config['dataloader']['num_steps'] or 1024 * \
        64 // config['dataloader']['batch_size'] // config['optimizer']['aggregate_k_gradients']

    if args.orchestration.extra_fast_test:
        config['prior']['n_samples'] = 2 * 16
        config['transformer']['nhead'] = 1

    if extra_config is not None:
        update_config(config, extra_config)

    save_every = orchestration.save_every

    optimizer_state, scheduler = None, None
    # no warm start
    if config['orchestration']['detect_anomaly']:
        print("ENABLING GRADIENT DEBUGGING (detect-anomaly)! Don't use for training.")
        torch.autograd.set_detect_anomaly(True)

    model_string = get_model_string(config, num_gpus, device, parser)
    save_callback = make_training_callback(save_every, model_string, base_path, report, config, orchestration.no_mlflow,
                                           orchestration.st_checkpoint_dir, classification=config['transformer']['classification_task'], validate=orchestration.validate)

    # no MLFlow and WanDB
    total_loss, model, dl, epoch = get_model(config, device, should_train=True, verbose=1, epoch_callback=save_callback,
                                                 optimizer_state=optimizer_state, scheduler=scheduler,
                                                 load_model_strict=orchestration.continue_run or orchestration.load_strict)
    
    if rank == 0:
        save_callback(model, None, None, "on_exit")
    return {'loss': total_loss, 'model': model, 'dataloader': dl,
            'config': config, 'base_path': base_path,
            'model_string': model_string, 'epoch': epoch}


if __name__ == "__main__":
    main(sys.argv[1:])
