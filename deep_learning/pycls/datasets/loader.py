#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the original graph2nn github repo.

# File modifications and additions by Rowan AI Lab, licensed under the Creative Commons Zero v1.0 Universal
# LICENSE file in the root directory of this source tree.

"""Data loader."""

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import torch

from pycls.config import cfg
from pycls.datasets.cifar10 import Cifar10
from pycls.datasets.cifar100 import Cifar100
from pycls.datasets.tinyimagenet200 import TinyImageNet200
from pycls.datasets.imagenet import ImageNet

import pycls.datasets.paths as dp

# Supported datasets
_DATASET_CATALOG = {
    'cifar10': Cifar10,
    'cifar100': Cifar100,
    'tinyimagenet200': TinyImageNet200,
    'imagenet': ImageNet
}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    assert dataset_name in _DATASET_CATALOG.keys(), \
        'Dataset \'{}\' not supported'.format(dataset_name)
    assert dp.has_data_path(dataset_name), \
        'Dataset \'{}\' has no data path'.format(dataset_name)
    # Retrieve the data path for the dataset
    data_path = dp.get_data_path(dataset_name)
    # Construct the dataset
    dataset = _DATASET_CATALOG[dataset_name](data_path, split, batch_size)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False
    )



def construct_test_loader_adv():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), \
        'Sampler type \'{}\' not supported'.format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
