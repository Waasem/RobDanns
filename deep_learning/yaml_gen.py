#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the original graph2nn github repo.

# File modifications and additions by Rowan AI Lab, licensed under the Creative Commons Zero v1.0 Universal
# LICENSE file in the root directory of this source tree.

"""Generate yaml files for experiment configurations."""

import yaml
# import math
import os
import re
import argparse
import numpy as np
import shutil


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        dest='task',
        help='Generate configs for the given tasks: e.g., mlp_cifar, cnn_cifar, cnn_imagenet, resnet18_tinyimagenet, resenet18_imagenet',
        default='mlp_cifar10',
        type=str
    )
    return parser.parse_args()


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def gen(dir_in, dir_out, fname_base, vars_label, vars_alias, vars_value):
    '''Generate yaml files'''
    with open(dir_in + fname_base + '.yaml') as f:
        data_base = yaml.unsafe_load(f)

    for vars in vars_value:
        data = data_base.copy()
        fname_new = fname_base
        for id, var in enumerate(vars):
            if vars_label[id][0] in data:  # if key1 exist
                data[vars_label[id][0]][vars_label[id][1]] = var
            else:
                data[vars_label[id][0]] = {vars_label[id][1]: var}
            if vars_label[id][1] == 'TRANS_FUN':
                var = var.split('_')[0]
            fname_new += '_{}{}'.format(vars_alias[id], var)
        with open(dir_out + fname_new + '.yaml', "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def gen_single(dir_in, dir_out, fname_base, vars_label, vars_alias, vars, comment='best'):
    '''Generate yaml files for a single experiment'''
    with open(dir_in + fname_base + '.yaml') as f:
        data_base = yaml.unsafe_load(f)

    data = data_base.copy()
    fname_new = '{}_{}'.format(fname_base, comment)
    for id, var in enumerate(vars):
        if vars_label[id][0] in data:  # if key1 exist
            data[vars_label[id][0]][vars_label[id][1]] = var
        else:
            data[vars_label[id][0]] = {vars_label[id][1]: var}
    with open(dir_out + fname_new + '.yaml', "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def grid2list(grid):
    '''grid search to list'''
    list_in = [[i] for i in grid[0]]
    grid.pop(0)
    for grid_temp in grid:
        list_out = []
        for val in grid_temp:
            for list_temp in list_in:
                list_out.append(list_temp + [val])
        list_in = list_out
    return list_in


args = parse_args()

# Format for all experiments
# Note: many arguments are deprecated, they are kept to be consistent with existing experimental results
vars_value = []
vars_label = [['RESNET', 'TRANS_FUN'], ['RGRAPH', 'TALK_MODE'], ['RGRAPH', 'GROUP_NUM'],
              ['RGRAPH', 'MESSAGE_TYPE'], ['RGRAPH', 'SPARSITY'], ['RGRAPH', 'P'], ['RGRAPH', 'AGG_FUNC'],
              ['RGRAPH', 'SEED_GRAPH'], ['RGRAPH', 'SEED_TRAIN_START'], ['RGRAPH', 'SEED_TRAIN_END'],
              ['RGRAPH', 'KEEP_GRAPH'],
              ['RGRAPH', 'ADD_1x1'], ['RGRAPH', 'UPPER'], ['TRAIN', 'AUTO_MATCH'], ['OPTIM', 'MAX_EPOCH'], ['TRAIN', 'CHECKPOINT_PERIOD']]
vars_alias = ['trans', 'talkmode', 'num',
              'message', 'sparsity', 'p', 'agg',
              'graphseed', 'starttrainseed', 'endtrainseed', 'keep',
              'add1x1', 'upper', 'match', 'epoch', 'chkpt'
              ]


## Note: (1) how many relational graphs used to run: graphs_n64_54, graphs_n64_441, graphs_n64_3854
## (2): "best_id" is to be discovered based on experimental results. Given best_id is for graph2nn experiments
## (3): Each ImageNet experiment provides with 1 seed. One can change SEED_TRAIN_START and SEED_TRAIN_END
## to get results for multiple seeds


### 5 layer 64 dim MLP, CIFAR-10
# usage: python yaml_gen.py --task mlp_cifar10
if args.task == 'mlp_cifar10':
#     best_id = 3552 # best_id is for graph2nn experiments.
    fname_bases = ['mlp_bs128_1gpu_layer3']
#     graphs = np.load('analysis/graphs_n64_53.npy')
    
    # To load the .npy file
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    graphs = np.load('analysis/graphs_n64_53.npy')
    # restore np.load for future normal usage
    np.load = np_load_old
    
    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['talklinear_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 6, True,
                        0, True, True, 200, 200]]
    vars_value += [['linear_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 6, True,
                    0, True, True, 200, 200]]


### CNN, CIFAR-10
# usage : python yaml_gen.py --task cnn_cifar10
if args.task == 'cnn_cifar10':
    # best_id = 3552  # best_id is for graph2nn experiments. 
    fname_bases = ['cnn6_bs1024_8gpu_64d']
    # graphs = np.load('analysis/graphs_n64_53.npy')

    # To load the .npy file
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    graphs = np.load('analysis/graphs_n64_53.npy')
    # restore np.load for future normal usage
    np.load = np_load_old

    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['convtalk_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 6, True,
                        0, True, True, 100, 100]]
    vars_value += [['convbasic_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 6, True,
                    0, True, True, 100, 100]]    
    
### CNN, CIFAR-100
# uage python yaml_gen.py --task cnn_cifar100
elif args.task == 'cnn_cifar100':
    # best_id = 3552  # best_id is for graph2nn experiments. 
    fname_bases = ['cnn6_bs640_1gpu_64d']
    # graphs = np.load('analysis/graphs_n64_53.npy')

    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    graphs = np.load('analysis/graphs_n64_53.npy')
    # restore np.load for future normal usage
    np.load = np_load_old

    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['convtalk_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 6, True,
                        0, True, True, 100, 100]]
    vars_value += [['convbasic_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 6, True,
                    0, True, True, 100, 100]] 


### Res18, tinyimagenet
# usage: python yaml_gen.py --task resnet18_tinyimagenet
elif args.task == 'resnet18_tinyimagenet':
    fname_bases = ['R-18_tiny_bs256_1gpu']
    # graphs = np.load('analysis/graphs_n64_53.npy')
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    graphs = np.load('analysis/graphs_n64_53.npy')
    # restore np.load for future normal usage
    np.load = np_load_old

    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['groupbasictalk_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 2, True,
                        0, True, True, 75, 25]]
    vars_value += [['channelbasic_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 2, True,
                    0, True, True, 75, 25]]


### CNN, imagenet
elif args.task == 'cnn_imagenet':
#     best_id = 27  # best_id is for graph2nn experiments.
    fname_bases = ['cnn6_bs32_1gpu_64d', 'cnn6_bs256_8gpu_64d']
    # graphs = np.load('analysis/graphs_n64_53.npy')

    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    graphs = np.load('analysis/graphs_n64_53.npy')
    # restore np.load for future normal usage
    np.load = np_load_old
    
    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['convtalk_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 2, True,
                        0, True, True, 100, 100]]
    vars_value += [['convbasic_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 2, True,
                    0, True, True, 100, 100]]


### Res18, ImageNet
# usage : python yaml_gen.py --task resnet18_imagenet
elif args.task == 'resnet18_imagenet':
    # best_id = 37  # best_id is for graph2nn experiments.
    fname_bases = ['R-18_bs450_1gpu']
    # graphs = np.load('analysis/graphs_n64_53.npy')
    
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    graphs = np.load('analysis/graphs_n64_53.npy')
    np.load = np_load_old
    
    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['groupbasictalk_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 2, True,
                        0, True, True, 75, 25]]
    vars_value += [['channelbasic_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 2, True,
                    0, True, True, 75, 25]]
    
    
### Res34, ImageNet
# usage: python yaml_gen.py --task resnet34_imagenet
elif args.task == 'resnet34_imagenet':
    # best_id = 37  # best_id is for graph2nn experiments.
    fname_bases = ['R-34_bs32_1gpu', 'R-34_bs256_8gpu']
    # graphs = np.load('analysis/graphs_n64_52.npy')
    
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    graphs = np.load('analysis/graphs_n64_53.npy')
    np.load = np_load_old

    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['groupbasictalk_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 2, True,
                        0, True, True, 100, 25]]
    vars_value += [['channelbasic_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 2, True,
                    0, True, True, 100, 25]]


### Res34-sep, ImageNet
elif args.task == 'resnet34sep_imagenet':
#     best_id = 36 # best_id is for graph2nn experiments.
    fname_bases = ['R-34_bs32_1gpu', 'R-34_bs256_8gpu']
    graphs = np.load('analysis/graphs_n64_53.npy')
    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['groupseptalk_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 2, True,
                        0, True, True, 100, 25]]
    vars_value += [['channelsep_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 2, True,
                    0, True, True, 100, 25]]



### Res50, ImageNet
elif args.task == 'resnet50_imagenet':
#     best_id = 22  # best_id is for graph2nn experiments.
    fname_bases = ['R-50_bs32_1gpu', 'R-50_bs256_8gpu']
    graphs = np.load('analysis/graphs_n64_53.npy')
    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['talkbottleneck_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 2, True,
                        0, True, True, 100, 25]]
    vars_value += [['bottleneck_transform', 'dense', 64,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 2, True,
                    0, True, True, 100, 25]]




### Efficient net, ImageNet
# uage : python yaml_gen.py --task efficient_imagenet
elif args.task == 'efficient_imagenet':
#     best_id = 42  # best_id is for graph2nn experiments.
    fname_bases = ['EN-B0_bs64_1gpu_nms', 'EN-B0_bs512_8gpu_nms']
    # graphs = np.load('analysis/graphs_n64_53.npy'))
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    graphs = np.load('analysis/graphs_n64_53.npy')
    np.load = np_load_old
    
    for graph in graphs:
        sparsity = float(round(graph[1], 6))
        randomness = float(round(graph[2], 6))
        graphseed = int(graph[3])
        vars_value += [['mbtalkconv_transform', 'dense', int(graph[0]),
                        'ws', sparsity, randomness, 'sum',
                        graphseed, 1, 2, True,
                        0, True, True, 100, 25]]
    vars_value += [['mbconv_transform', 'dense', 16,
                    'ws', 1.0, 0.0, 'sum',
                    1, 1, 2, True,
                    0, True, True, 100, 25]]


# ### MLP, cifar10, bio
# elif args.task == 'mlp_cifar10_bio':
#     fname_bases = ['mlp_bs128_1gpu_layer3']
#     for graph_type in ['mcwholeraw']:
#         vars_value += [['talklinear_transform', 'dense', 71,
#                         graph_type, 1.0, 0.0, 'sum',
#                         1, 1, 6, True,
#                         0, True, True, 200]]
#     for graph_type in ['mcvisualraw']:
#         vars_value += [['talklinear_transform', 'dense', 30,
#                         graph_type, 1.0, 0.0, 'sum',
#                         1, 1, 6, True,
#                         0, True, True, 200]]
#     for graph_type in ['catraw']:
#         vars_value += [['talklinear_transform', 'dense', 52,
#                         graph_type, 1.0, 0.0, 'sum',
#                         1, 1, 6, True,
#                         0, True, True, 200]]
#     vars_value += [['linear_transform', 'dense', 64,
#                     'ws', 1.0, 0.0, 'sum',
#                     1, 1, 6, True,
#                     0, True, True, 200]]

if args.task == 'cifar0':
    dir_name = 'cifar10'
elif 'cifar100' in args.task:
    dir_name = 'cifar100'
elif 'tinyimagenet' in args.task:
    dir_name = 'tinyimagenet200'
else:
    dir_name = 'imagenet'
dir_in = 'configs/baselines/{}/'.format(dir_name)
dir_out = 'configs/baselines/{}/{}/'.format(dir_name, args.task)
dir_out_all = 'configs/baselines/{}/{}/all/'.format(dir_name, args.task)
dir_out_best = 'configs/baselines/{}/{}/best/'.format(dir_name, args.task)
# makedirs_rm_exist(dir_out)
# makedirs_rm_exist(dir_out_all)
# makedirs_rm_exist(dir_out_best)

# print(vars_value)
for fname_base in fname_bases:
    if 'bio' not in args.task:
        gen(dir_in, dir_out_all, fname_base, vars_label, vars_alias, vars_value)
        # gen_single(dir_in, dir_out_best, fname_base, vars_label, vars_alias, vars_value[best_id], comment='best')
        gen_single(dir_in, dir_out_best, fname_base, vars_label, vars_alias, vars_value[-1], comment='baseline')
    else:
        gen(dir_in, dir_out_best, fname_base, vars_label, vars_alias, vars_value)
