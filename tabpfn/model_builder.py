import math
import os
import subprocess as sp
from functools import partial

import torch

import tabpfn.models.encoders as encoders
from tabpfn.assemble_model import assemble_model
from tabpfn.dataloader import get_dataloader
from tabpfn.train import train
from tabpfn.model_configs import get_base_config
from torch import nn


try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)


def get_criterion(max_num_classes):
    if max_num_classes == 2:
        loss = nn.BCEWthLogitsLoss(reduction='none')
    elif max_num_classes > 2:
        loss = nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError(f"Invalid number of classes: {max_num_classes}")
    return loss


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    optimizer_dict = optimizer.state_dict() if optimizer is not None else None

    import cloudpickle
    torch.save((model.state_dict(), optimizer_dict, scheduler, config_sample), os.path.join(path, filename), pickle_module=cloudpickle)


def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info


@cache
def load_model(path, device, verbose=False):
    states = torch.load(path, map_location='cpu')
    model_state = states[0]
    config_sample = states[-1]
    if 'y_encoder' not in config_sample and 'onehot' in path:
        # workaround for the single model that was saved without y_encoder
        # that happens to be my reference model.
        config_sample['y_encoder'] = 'one_hot'
    _, model, *_ = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model_state.pop("criterion.weight", None)

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, config_sample


def get_encoder(config):
    if (('nan_prob_no_reason' in config and config['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config and config['nan_prob_a_reason'] > 0.0) or
            ('nan_prob_unknown_reason' in config and config['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder(config['num_features'], config['emsize'])
    else:
        encoder = encoders.Linear(config['num_features'], config['emsize'], replace_nan_by_zero=True)

    if 'encoder' in config and config['encoder'] == 'featurewise_mlp':
        encoder = encoders.FeaturewiseMLP
    return encoder


def get_y_encoder(config):
    if config['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config['max_num_classes'], emsize=config['emsize'])
    elif config['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config['emsize'])
    else:
        raise ValueError(f"Unknown y_encoder: {config['y_encoder']}")
    return y_encoder


def old_config_to_new(old_config, new_config):
    old_config['learning_rate'] = old_config.pop('lr')
    old_config['n_samples'] = old_config.pop('bptt')
    if "y_encoder" not in old_config:
        old_config['y_encoder'] = 'linear'
    if "model_maker" in old_config:
        old_config['model_type'] = old_config.pop('model_maker')
    if "model_type" not in old_config:
        old_config['model_type'] = 'tabpfn'
    ignored_configs = ['seq_len_used', 'verbose', 'noise_type', 'normalize_to_ranking', 'normalize_by_used_features', 'num_categorical_features_sampler_a', 'differentiable',
                       'flexible', 'bptt_extra_samples', 'dynamic_batch_size', 'new_mlp_per_example', 'batch_size_per_gp_sample', 'normalize_ignore_label_too',
                       'differentiable_hps_as_style', 'rotate_normalized_labels', 'canonical_y_encoder', 'total_available_time_in_s', 'normalize_with_sqrt', 'done_part_in_training']
    for k in ignored_configs:
        old_config.pop(k)
    for k, v in new_config.items():
        if k in old_config:
            new_config[k] = old_config.pop(k)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if k3 in old_config:
                            new_config[k][k2][k3] = old_config.pop(k3)
                elif k2 in old_config:
                    new_config[k][k2] = old_config.pop(k2)
    assert len(old_config) == 0
    return new_config


def get_model(config, device, should_train=True, verbose=False, model_state=None, optimizer_state=None, scheduler=None, epoch_callback=None, load_model_strict=True):
    # copy config. Maybe should be a deepcopy?
    passed_config = config.copy()
    config = get_base_config()
    config.update(passed_config)
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    criterion = get_criterion(config['max_num_classes'])

    # backwards compatibility for cases where absence of parameter doesn't correspond to current default
    if 'n_samples' not in passed_config:
        config['n_samples'] = config['bptt']
    if 'y_encoder' not in passed_config:
        config['y_encoder'] = 'linear'
    if 'model_type' not in passed_config:
        if 'model_maker' in passed_config:
            config['model_type'] = config['model_maker']
        else:
            config['model_type'] = 'tabpfn'

    dl = get_dataloader(config=config, steps_per_epoch=config['num_steps'], batch_size=config['batch_size'], n_samples=config['n_samples'], device=device,
                        prior_type=config['prior_type'])
    y_encoder = get_y_encoder(config)

    encoder = get_encoder(config)
    model = assemble_model(encoder_layer=encoder, y_encoder_layer=y_encoder, num_features=config['num_features'], emsize=config['emsize'], nhead=config['nhead'],
                           nhid=config['emsize'] * config['nhid_factor'], nlayers=config['nlayers'], dropout=config['dropout'],
                           input_normalization=config['input_normalization'],  model_type=config['model_type'], max_num_classes=config['max_num_classes'],
                           predicted_hidden_layer_size=config['predicted_hidden_layer_size'],
                           model_state=model_state, load_model_strict=load_model_strict,
                           decoder_embed_dim=config['decoder_embed_dim'], decoder_two_hidden_layers=config['decoder_two_hidden_layers'],
                           decoder_hidden_size=config['decoder_hidden_size'], no_double_embedding=config['no_double_embedding'],
                           verbose=verbose_train, pre_norm=config['pre_norm'], efficient_eval_masking=config['efficient_eval_masking'],
                           output_attention=config['output_attention'], predicted_hidden_layers=config['predicted_hidden_layers'],
                           special_token=config['special_token'], weight_embedding_rank=config['weight_embedding_rank'] if config['low_rank_weights'] else None,
                           num_latents=config['num_latents'], input_bin_embedding=config['input_bin_embedding'], factorized_output=config['factorized_output'], output_rank=config['output_rank'],
                           bin_embedding_rank=config['bin_embedding_rank'], low_rank_weights=config['low_rank_weights'])

    if model_state is not None:
        if not load_model_strict:
            for k, v in model.state_dict().items():
                if k in model_state and model_state[k].shape != v.shape:
                    model_state.pop(k)
        model.load_state_dict(model_state, strict=load_model_strict)

    if verbose:
        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    if 'losses' in config:
        # for continuing training
        model.losses = config['losses']
        model.learning_rates = config['learning_rates']
        model.wallclock_times = config.get('wallclock_times', [])
    if should_train:
        model = train(dl,
                      model, criterion=criterion,
                      optimizer_state=optimizer_state, scheduler=scheduler, epochs=config['epochs'], stop_after_epochs=config['stop_after_epochs'],
                        warmup_epochs=config['warmup_epochs'], device=device, aggregate_k_gradients=config['aggregate_k_gradients'], epoch_callback=epoch_callback,
                        learning_rate=config['lr'], min_lr=config['min_lr'],
                        learning_rate_schedule=config['learning_rate_schedule'], lr_decay=config['lr_decay'], verbose=verbose_train, train_mixed_precision=config['train_mixed_precision'],
                        weight_decay=config['weight_decay'], adaptive_batch_size=config['adaptive_batch_size'],
                        reduce_lr_on_spike=config['reduce_lr_on_spike'], adam_beta1=config['adam_beta1'], spike_tolerance=config['spike_tolerance']
                        )

    return model
