import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext


import torch
from torch import nn
from torch import autograd

import tabpfn.utils as utils

from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
import tabpfn.priors as priors
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
from tabpfn.utils import init_dist
from torch.cuda.amp import autocast, GradScaler
from torch import nn

class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')


def train(dl, model, 
          epochs=10, steps_per_epoch=100, bptt=10, lr=None, weight_decay=0.0, warmup_epochs=10, scheduler=get_cosine_schedule_with_warmup,
          validation_period=10, single_eval_pos_gen=None, bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, epoch_callback=None, train_mixed_precision=False,
          ):
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    using_dist, rank, device = init_dist(device)

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    dl.model = model


    # learning rate
    print(f"learning rate:{lr}")
    print(f"steps_per_epoch:{steps_per_epoch}")
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                if bptt_extra_samples is None:
                    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                else:
                    single_eval_pos = targets.shape[0] - bptt_extra_samples
                with autocast(enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    output = model(tuple(e.to(device) if torch.is_tensor(e) else e for e in data) if isinstance(data, tuple) else data.to(device)
                                   , single_eval_pos=single_eval_pos)

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    if isinstance(model.criterion, nn.GaussianNLLLoss):
                        assert output.shape[-1] == 2, \
                            'need to write a little bit of code to handle multiple regression targets at once'

                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = model.criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
                    elif isinstance(model.criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        losses = model.criterion(output.flatten(), targets.to(device).flatten())
                    elif isinstance(model.criterion, nn.CrossEntropyLoss):
                        losses = model.criterion(output.reshape(-1, model.n_out), targets.to(device).long().flatten())
                    else:
                        losses = model.criterion(output, targets)
                    losses = losses.view(*output.shape[0:2])
                    loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                    loss = loss / aggregate_k_gradients

                if scaler: loss = scaler.scale(loss)
                loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    try:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    optimizer.zero_grad()

                step_time = time.time() - before_forward

                if torch.isnan(loss):
                    print("NAN loss encountered")
                else:
                    total_loss += losses.mean().cpu().detach().item()
                    total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                        losses[:bptt-single_eval_pos].mean().cpu().detach()

                    total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                nan_steps += nan_share
                ignore_steps += (targets == -100).float().mean()


            before_get_batch = time.time()
        return total_loss / steps_per_epoch, (total_positional_losses / total_positional_losses_recorded).tolist(),\
               time_to_get_batch, forward_time, step_time, nan_steps.cpu().item()/(batch+1),\
               ignore_steps.cpu().item()/(batch+1)

    total_loss = float('inf')
    total_positional_losses = float('inf')
    try:
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
            if verbose:
                print(f"start of epoch {epoch}")

            epoch_start_time = time.time()
            total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                train_epoch()
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            if verbose:
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.4f} | ')
                if verbose > 2:
                    print(f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])},")
                print(
                    f' lr {scheduler.get_last_lr()[0]} data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' forward time {forward_time:5.2f}' 
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)

            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch / epochs)
            scheduler.step()
    except KeyboardInterrupt:
        pass

    if rank == 0: # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        return total_loss, total_positional_losses, model.to('cpu'), dl
