#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the original graph2nn github repo.

# File modifications and additions by Rowan AI Lab, licensed under the Creative Commons Zero v1.0 Universal
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""


import os
import torch

from collections import OrderedDict
from pycls.config import cfg

import pycls.utils.distributed as du


# Common prefix for checkpoint file names
_NAME_PREFIX = 'model_epoch_'

# Checkpoints directory name
_DIR_NAME = 'checkpoints'


def get_checkpoint_dir():
    """Get location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def got_checkpoint_dir():
    """Get location for storing checkpoints for inference time."""
    return os.path.join(cfg.CHECKPT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Get the full path to a checkpoint file."""
    name = '{}{:04d}.pyth'.format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)

def got_checkpoint(epoch):
    """Get the full path to a checkpoint file for inference time."""
    name = '{}{:04d}.pyth'.format(_NAME_PREFIX, epoch)
    return os.path.join(got_checkpoint_dir(), name)


def get_checkpoint_last():
    d = get_checkpoint_dir()
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if _NAME_PREFIX in f]
    assert len(names), 'No checkpoints found in \'{}\'.'.format(d)
    name = sorted(names)[-1]
    return os.path.join(d, name)


def got_checkpoint_last():
    d = got_checkpoint_dir()
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if _NAME_PREFIX in f]
    assert len(names), 'No checkpoints found in \'{}\'.'.format(d)
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint():
    """Determines if the given directory contains a checkpoint."""
    d = get_checkpoint_dir()
    print("checkpoint directory =", d)
    files = os.listdir(d) if os.path.exists(d) else []
    return any(_NAME_PREFIX in f for f in files)


def had_checkpoint():
    """Determines if the given directory contains a checkpoint for inference time."""
    d = got_checkpoint_dir()
    print("checkpoint directory =", d)
    files = os.listdir(d) if os.path.exists(d) else []
    return any(_NAME_PREFIX in f for f in files)


def is_checkpoint_epoch(cur_epoch):
    """Determines if a checkpoint should be saved on current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(model, optimizer, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not du.is_master_proc():
        return
    os.makedirs(get_checkpoint_dir(), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'cfg': cfg.dump()
    }
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), \
        'Checkpoint \'{}\' not found'.format(checkpoint_file)
#     if cfg.IS_INFERENCE and cfg.IS_DDP:
#         state_dict = torch.load(checkpoint_file, map_location='cpu')
#         new_state_dict = OrderedDict()
#         print("state_dict.items() :", state_dict)
#         for k, v in state_dict.items():
#             name = k[7:] # remove `module.`
#             new_state_dict[name] = v
#         # load params
#         epoch = state_dict['epoch']
#         model.load_state_dict(new_state_dict['model_state'])
#         if optimizer:
#             optimizer.load_state_dict(new_state_dict['optimizer_state'])
    if cfg.IS_INFERENCE:
        print("Mapping model to CPU")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
#         print(checkpoint)
    else:
        checkpoint = torch.load(checkpoint_file)
    epoch = checkpoint['epoch']
    print("Epochs from checkpoint = ", epoch)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    return epoch
