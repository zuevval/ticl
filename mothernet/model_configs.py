import torch

from mothernet.config_utils import merge_dicts


def get_optimizer_config():
    optimizer = {
        "aggregate_k_gradients": 1,
        "learning_rate": 0.00003,
        "epochs": 4000,
        "train_mixed_precision": True,
        'stop_after_epochs': None,
        'reduce_lr_on_spike': False,
        'warmup_epochs': 20,
        'learning_rate_schedule': 'cosine',
        'min_lr': 1e-8,
        'adam_beta1': 0.9,
        'spike_tolerance': 4,
        'weight_decay': 0.0,
        'lr_decay': 0.99,
        'adaptive_batch_size': True
    }
    return {'optimizer': optimizer}


def get_transformer_config():
    transformer = {
        "emsize": 512,
        "nlayers": 12,
        "dropout": 0.0,
        "nhid_factor": 2,
        'nhead': 512 // 128,
        'init_method': None,
        'recompute_attn': True,
        'pre_norm': False,
        'y_encoder': "one_hot",
        'classification_task': True,
        'efficient_eval_masking': True,
        'input_normalization': False,
        'tabpfn_zero_weights': True,

    }
    return {'transformer': transformer}


def get_prior_config(max_features=100, n_samples=1024+128):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """

    prior = {
        "num_features": max_features,
        "n_samples": n_samples,
        "eval_positions": [n_samples * 0.95],
        'heterogeneous_batches': False,
        'multiclass_loss_type': 'nono',  # 'compatible'
        'prior_type': 'prior_bag',
        'prior_bag': {'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0}}}

    mlp_prior_config = {"pre_sample_causes": True,
                        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
                        'prior_mlp_scale_weights_sqrt': True,
                        'random_feature_rotation': True,
                        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True, 'lower_bound': 2},
                        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
                        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
                        # This mustn't be too high since activations get too large otherwise
                        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
                        "noise_std": {'distribution': 'log_uniform', 'min': 1e-4, 'max': .5},
                        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                       'lower_bound': 2},
                        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        # "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
                        "prior_mlp_activations": {'distribution': 'meta_choice', 'choice_values': [
                            torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU
                        ]},
                        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        'add_uninformative_features': False}

    prior['mlp'] = mlp_prior_config

    gp_prior_config = {
        'outputscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'lengthscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]},
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
    }

    prior['gp'] = gp_prior_config

    prior['step_function'] = {
        'max_steps': 1,
        'sampling': 'uniform',
    }

    max_num_classes = 10
    classsification_prior = {
        "max_num_classes": max_num_classes,
        "num_classes": {'distribution': 'uniform_int', 'min': 2, 'max': max_num_classes},
        # "noise_type": "Gaussian",  # NN unused?!
        "balanced": False,
        'output_multiclass_ordered_p': 0.,
        'multiclass_max_steps': 10,
        "multiclass_type": 'rank',
        'categorical_feature_p': .2,  # diff: .0
        'nan_prob_no_reason': 0.0,
        'nan_prob_a_reason': 0.0,
        'set_value_to_nan': .9,
        'num_features_sampler': 'uniform',
        'pad_zeros': True,
        'feature_curriculum': False,
    }
    prior['classification'] = classsification_prior

    dataloader = {
        "batch_size": 8,
        "num_steps": 8192,
        'min_eval_pos': 2}

    prior['boolean'] = {
        'max_fraction_uninformative': 0.5,
        'p_uninformative': 0.5}

    return {'prior': prior, 'dataloader': dataloader}



def get_shared_defaults():
    config = get_prior_config()
    config.update(get_optimizer_config())
    config.update(get_transformer_config())
    return config


def get_model_default_config(model_type):
    if model_type == 'tabpfn':
        config = get_shared_defaults()
    else:
        raise ValueError(f"Unknown model type {model_type}")
    config['model_type'] = model_type
    return config
