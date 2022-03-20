#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the original graph2nn github repo.

# File modifications and additions by Rowan AI Lab, licensed under the Creative Commons Zero v1.0 Universal
# LICENSE file in the root directory of this source tree.

"""CIFAR100 dataset."""

import numpy as np
import os
import pickle
import torch
import torch.utils.data

import pycls.datasets.transforms as transforms
from torchvision import datasets
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [129.3, 124.1, 112.4]
_SD = [68.2,  65.4,  70.4]


class Cifar100(torch.utils.data.Dataset):
    """CIFAR-100 dataset."""

    def __init__(self, data_path, split, batch_size):
        assert os.path.exists(data_path), \
            'Data path \'{}\' not found'.format(data_path)
        assert split in ['train', 'test'], \
            'Split \'{}\' not supported for cifar'.format(split)
        logger.info('Constructing CIFAR-100 {}...'.format(split))
        self._data_path = data_path
        self._split = split
        self._batch_size = batch_size
        # Data format:
        #   self._inputs - (split_size, 3, 32, 32) ndarray
        #   self._labels - split_size list
        self._inputs, self._labels = self._load_data()

    def _load_batch(self, batch_path):
        with open(batch_path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        return d[b'data'], d[b'fine_labels']
#         return d[b'data'], d[b'labels']

    def _load_data(self):
        """Loads data in memory."""
        logger.info('{} data path: {}'.format(self._split, self._data_path))
        # Compute data batch names
        if self._split == 'train':
            batch_names = ['train']
#             datasets.CIFAR100(self._data_path, train=True)
#             batch_names = ['data_batch_{}'.format(i) for i in range(1, 6)]
        else:
            batch_names = ['test']
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            inputs_batch, labels_batch = self._load_batch(batch_path)
            inputs.append(inputs_batch)
            labels += labels_batch
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, 32, 32))
        return inputs, labels

    def _transform_image(self, image):
        """Transforms the image for network input."""
        if self._batch_size != 1:
            image = transforms.color_normalization(image, _MEAN, _SD)
        if self._split == 'train':
            image = transforms.horizontal_flip(image=image, prob=0.5)
            image = transforms.random_crop(image=image, size=32, pad_size=4)
        return image

    def __getitem__(self, index):
        image, label = self._inputs[index, ...], self._labels[index]
        image = self._transform_image(image)
        return image, label

    def __len__(self):
        return self._inputs.shape[0]
