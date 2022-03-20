#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the original graph2nn github repo.

# File modifications and additions by Rowan AI Lab, licensed under the Creative Commons Zero v1.0 Universal
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import argparse
import pickle
import numpy as np
import os
import sys
import torch
import math
import torchvision
import torchattacks

from pycls.config import assert_cfg
from pycls.config import cfg
from pycls.config import dump_cfg
from pycls.datasets import loader
from pycls.models import model_builder
from pycls.utils.meters import TestMeter

import pycls.models.losses as losses
import pycls.models.optimizer as optim
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu
import pycls.utils.net as nu
import pycls.datasets.transforms as transforms

from datetime import datetime
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from skimage.util import random_noise

print("Using GPU :", torch.cuda.current_device())
logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See pycls/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
            (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def log_model_info(model, writer_eval=None):
    """Logs model info"""
    logger.info('Model:\n{}'.format(model))
    params = mu.params_count(model)
    flops = mu.flops_count(model)
    logger.info('Params: {:,}'.format(params))
    logger.info('Flops: {:,}'.format(flops))
    logger.info('Number of node: {:,}'.format(cfg.RGRAPH.GROUP_NUM))
    # logger.info('{}, {}'.format(params,flops))
    if writer_eval is not None:
        writer_eval.add_scalar('Params', params, 1)
        writer_eval.add_scalar('Flops', flops, 1)
    return params, flops


@torch.no_grad()
def eval_epoch(test_loader, model, test_meter, cur_epoch, writer_eval=None, params=0, flops=0, is_master=False):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    # val_input_imgs,
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch, writer_eval, params, flops, model, is_master=is_master)
    stats = test_meter.get_epoch_stats(cur_epoch)
    test_meter.reset()
    if cfg.RGRAPH.SAVE_GRAPH:
        adj_dict = nu.model2adj(model)
        adj_dict = {**adj_dict, 'top1_err': stats['top1_err']}
        os.makedirs('{}/graphs/{}'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN), exist_ok=True)
        np.savez('{}/graphs/{}/{}.npz'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN, cur_epoch), **adj_dict)


def save_noisy_image(img, name):
    if img.size(2) == 32:
        img = img.view(img.size(0), 3, 32, 32)
        save_image(img, name)
    else:
        img = img.view(img.size(0), 3, 224, 224)
        save_image(img, name)        

## Functions to save noisy images.

# def gaussian_noise(test_loader):
#     print("Adding gaussian_noise")
#     for data in test_loader:
#         img, _ = data[0], data[1]
#         gaussian_img_05 = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True))
#         gaussian_img_2 = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.2, clip=True))
#         gaussian_img_4 = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.4, clip=True))
#         gaussian_img_6 = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.6, clip=True))
#         save_noisy_image(gaussian_img_05, r"noisy-images/gaussian_05.png")
#         save_noisy_image(gaussian_img_2, r"noisy-images/gaussian_2.png") 
#         save_noisy_image(gaussian_img_4, r"noisy-images/gaussian_4.png") 
#         save_noisy_image(gaussian_img_6, r"noisy-images/gaussian_6.png") 
#         break

# def salt_pepper_noise(test_loader):
#     print("Adding salt_pepper_noise")
#     for data in test_loader:
#         img, _ = data[0], data[1]
#         s_vs_p_5 = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.5, clip=True))
#         s_vs_p_6 = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.6, clip=True))
#         s_vs_p_7 = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.7, clip=True))
#         save_noisy_image(s_vs_p_5, r"noisy-images/s&p_5.png")
#         save_noisy_image(s_vs_p_6, r"noisy-images/s&p_6.png")
#         save_noisy_image(s_vs_p_7, r"noisy-images/s&p_7.png")
#         break        

# def speckle_noise(test_loader):
#     print("Adding speckle_noise")
#     for data in test_loader:
#         img, _ = data[0], data[1]
#         speckle_img_05 = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.05, clip=True))
#         speckle_img_2 = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.2, clip=True))
#         speckle_img_4 = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.4, clip=True))
#         speckle_img_6 = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.6, clip=True))
#         save_noisy_image(speckle_img_05, r"noisy-images/speckle_05.png")
#         save_noisy_image(speckle_img_2, r"noisy-images/speckle_2.png")
#         save_noisy_image(speckle_img_4, r"noisy-images/speckle_4.png")
#         save_noisy_image(speckle_img_6, r"noisy-images/speckle_6.png")
#         break

        
def train_model(writer_eval=None, is_master=False):
    """Trains the model."""
    # Fit flops/params
    if cfg.TRAIN.AUTO_MATCH and cfg.RGRAPH.SEED_TRAIN == cfg.RGRAPH.SEED_TRAIN_START:
        mode = 'flops'  # flops or params
        if cfg.TRAIN.DATASET == 'cifar10':
            pre_repeat = 15
            if cfg.MODEL.TYPE == 'resnet':  # ResNet20
                stats_baseline = 40813184
            elif cfg.MODEL.TYPE == 'mlpnet':  # 5-layer MLP. cfg.MODEL.LAYERS exclude stem and head layers
                if cfg.MODEL.LAYERS == 3:
                    if cfg.RGRAPH.DIM_LIST[0] == 256:
                        stats_baseline = 985600
                    elif cfg.RGRAPH.DIM_LIST[0] == 512:
                        stats_baseline = 2364416
                    elif cfg.RGRAPH.DIM_LIST[0] == 1024:
                        stats_baseline = 6301696
            elif cfg.MODEL.TYPE == 'cnn':
                if cfg.MODEL.LAYERS == 3:
                    if cfg.RGRAPH.DIM_LIST[0] == 64:
                        stats_baseline = 48957952
                    elif cfg.RGRAPH.DIM_LIST[0] == 512:
                        stats_baseline = 806884352
                    elif cfg.RGRAPH.DIM_LIST[0] == 16:
                        stats_baseline = 1216672
                elif cfg.MODEL.LAYERS == 6:
                    if '64d' in cfg.OUT_DIR:
                        stats_baseline = 48957952
                    elif '16d' in cfg.OUT_DIR:
                        stats_baseline = 3392128
        elif cfg.TRAIN.DATASET == 'cifar100':
            pre_repeat = 15
            if cfg.MODEL.TYPE == 'resnet':  # ResNet20
                if cfg.MODEL.DEPTH == 20:
                    stats_baseline = 40813184   # ResNet20
                elif cfg.MODEL.DEPTH == 26:
                    stats_baseline = 56140000     # ResNet26
                elif cfg.MODEL.DEPTH == 34:
                    stats_baseline = 71480000   # ResNet34
                elif cfg.MODEL.DEPTH == 38:
                    stats_baseline = 86819000     # ResNet38
                elif cfg.MODEL.DEPTH == 50:
                    stats_baseline = 130000000    # ResNet50
            elif cfg.MODEL.TYPE == 'mlpnet':  # 5-layer MLP. cfg.MODEL.LAYERS exclude stem and head layers
                if cfg.MODEL.LAYERS == 3:
                    if cfg.RGRAPH.DIM_LIST[0] == 256:
                        stats_baseline = 985600
                    elif cfg.RGRAPH.DIM_LIST[0] == 512:
                        stats_baseline = 2364416
                    elif cfg.RGRAPH.DIM_LIST[0] == 1024:
                        stats_baseline = 6301696
            elif cfg.MODEL.TYPE == 'cnn':
                if cfg.MODEL.LAYERS == 3:
                    if cfg.RGRAPH.DIM_LIST[0] == 512:
                        stats_baseline = 806884352
                    elif cfg.RGRAPH.DIM_LIST[0] == 16:
                        stats_baseline = 1216672
                elif cfg.MODEL.LAYERS == 6:
                    if '64d' in cfg.OUT_DIR:
                        stats_baseline = 48957952
                    elif '16d' in cfg.OUT_DIR:
                        stats_baseline = 3392128
        elif cfg.TRAIN.DATASET == 'tinyimagenet200':
            pre_repeat = 9
            if cfg.MODEL.TYPE == 'resnet':
                if 'basic' in cfg.RESNET.TRANS_FUN and cfg.MODEL.DEPTH == 18:  # ResNet18
                    stats_baseline = 1820000000
                elif 'basic' in cfg.RESNET.TRANS_FUN and cfg.MODEL.DEPTH == 34:  # ResNet34
                    stats_baseline = 3663761408
                elif 'sep' in cfg.RESNET.TRANS_FUN:  # ResNet34-sep
                    stats_baseline = 553614592
                elif 'bottleneck' in cfg.RESNET.TRANS_FUN:  # ResNet50
                    stats_baseline = 4089184256
            elif cfg.MODEL.TYPE == 'efficientnet':  # EfficientNet
                stats_baseline = 385824092
            elif cfg.MODEL.TYPE == 'cnn':  # CNN
                if cfg.MODEL.LAYERS == 6:
                    if '64d' in cfg.OUT_DIR:
                        stats_baseline = 166438912
        elif cfg.TRAIN.DATASET == 'imagenet':
            pre_repeat = 9
            if cfg.MODEL.TYPE == 'resnet':
                if 'basic' in cfg.RESNET.TRANS_FUN:  # ResNet34
                    stats_baseline = 3663761408
                elif 'sep' in cfg.RESNET.TRANS_FUN:  # ResNet34-sep
                    stats_baseline = 553614592
                elif 'bottleneck' in cfg.RESNET.TRANS_FUN:  # ResNet50
                    stats_baseline = 4089184256
            elif cfg.MODEL.TYPE == 'efficientnet':  # EfficientNet
                stats_baseline = 385824092
            elif cfg.MODEL.TYPE == 'cnn':  # CNN
                if cfg.MODEL.LAYERS == 6:
                    if '64d' in cfg.OUT_DIR:
                        stats_baseline = 166438912
        cfg.defrost()
        stats = model_builder.build_model_stats(mode)
        if stats != stats_baseline:
            # 1st round: set first stage dim
            for i in range(pre_repeat):
                scale = round(math.sqrt(stats_baseline / stats), 2)
                first = cfg.RGRAPH.DIM_LIST[0]
                ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                first = int(round(first * scale))
                cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                stats = model_builder.build_model_stats(mode)
            flag_init = 1 if stats < stats_baseline else -1
            step = 1
            while True:
                first = cfg.RGRAPH.DIM_LIST[0]
                ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                first += flag_init * step
                cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                stats = model_builder.build_model_stats(mode)
                flag = 1 if stats < stats_baseline else -1
                if stats == stats_baseline:
                    break
                if flag != flag_init:
                    if cfg.RGRAPH.UPPER == False:  # make sure the stats is SMALLER than baseline
                        if flag < 0:
                            first = cfg.RGRAPH.DIM_LIST[0]
                            ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                            first -= flag_init * step
                            cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                        break
                    else:
                        if flag > 0:
                            first = cfg.RGRAPH.DIM_LIST[0]
                            ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                            first -= flag_init * step
                            cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                        break
            # 2nd round: set other stage dim
            first = cfg.RGRAPH.DIM_LIST[0]
            ratio_list = [int(round(dim / first)) for dim in cfg.RGRAPH.DIM_LIST]
            stats = model_builder.build_model_stats(mode)
            flag_init = 1 if stats < stats_baseline else -1
            if 'share' not in cfg.RESNET.TRANS_FUN:
                for i in range(1, len(cfg.RGRAPH.DIM_LIST)):
                    for j in range(ratio_list[i]):
                        cfg.RGRAPH.DIM_LIST[i] += flag_init
                        stats = model_builder.build_model_stats(mode)
                        flag = 1 if stats < stats_baseline else -1
                        if flag_init != flag:
                            cfg.RGRAPH.DIM_LIST[i] -= flag_init
                            break
        stats = model_builder.build_model_stats(mode)
        print('FINAL', cfg.RGRAPH.GROUP_NUM, cfg.RGRAPH.DIM_LIST, stats, stats_baseline, stats < stats_baseline)
    # Build the model (before the loaders to ease debugging)
    
    model = model_builder.build_model()
    params, flops = log_model_info(model, writer_eval)
    
    if cfg.IS_INFERENCE and cfg.IS_DDP:
        model = torch.nn.parallel.DataParallel(model)
        
#     for name, param in model.named_parameters():
#         print(name, param.shape)

    # Define the loss function
    loss_fun = losses.get_loss_fun()
    # Construct the optimizer
    optimizer = optim.construct_optimizer(model)

    # Load a checkpoint if applicable
    start_epoch = 0
    if cu.had_checkpoint():
        print("Checking for a checkpoint")
        last_checkpoint = cu.get_checkpoint_last()
        print("Last Checkpoint : ", last_checkpoint)
        checkpoint_epoch = cu.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info('Loaded checkpoint from: {}'.format(last_checkpoint))
        if checkpoint_epoch == cfg.OPTIM.MAX_EPOCH:
            exit()
            start_epoch = checkpoint_epoch
        else:
            start_epoch = checkpoint_epoch + 1
    print("Epoch = ", start_epoch)
    # Create data loaders
    test_loader = loader.construct_test_loader()
    # Create meters
    test_meter = TestMeter(len(test_loader))

    if cfg.ONLINE_FLOPS:
        model_dummy = model_builder.build_model()
        IMAGE_SIZE = 224
        n_flops, n_params = mu.measure_model(model_dummy, IMAGE_SIZE, IMAGE_SIZE)
        logger.info('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
        del (model_dummy)

    # Perform the training loop
    logger.info('Start epoch: {}'.format(start_epoch + 1))

    if start_epoch == cfg.OPTIM.MAX_EPOCH:
        cur_epoch = start_epoch - 1
        eval_epoch(test_loader, model, test_meter, cur_epoch,
                   writer_eval, params, flops, is_master=is_master)        
            
    noise_mode = ['gaussian', 'speckle', 's&p']
    noise_var = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # change the variance values as desired.
    model.eval()
    accuracies_gaussian = []
    accuracies_saltpepper = []
    accuracies_speckle = []
    for mode in noise_mode:
        for level in noise_var:
            print("Adding noise={} at level={} to images".format(mode, level))
            ctr = 0
            correct = 0
            total = 0
            for cur_iter, (inputs, labels) in enumerate(test_loader):
                if not 's&p' in mode:
                    noisy_img = torch.tensor(random_noise(inputs, mode=mode, mean=0, var=level, clip=True))
                else:
                    noisy_img = torch.tensor(random_noise(inputs, mode=mode, salt_vs_pepper=0.5, clip=True))
                
                noisy_img, labels = noisy_img.cuda(), labels.cuda(non_blocking=True)
                outputs = model(noisy_img.float())
                _, predicted = torch.max(outputs.data, 1)  
                ctr += 1
                total += labels.size(0)
                correct += (predicted == labels).sum()
                if total > X: # replace X with the number of images to be generated for adversarial attacks.
                    break
            acc = 100 * float(correct) / total
            print("acc =", round(acc, 2), "correct =", float(correct), "total =", total)
            
            if 'gaussian' in mode:
                print('Robust Accuracy = {:.3f} with level = {:.2f}'.format(acc, level))
                accuracies_gaussian.append(round(acc, 2))
                print("Guassian Accuracies after append :", accuracies_gaussian)            
            elif 'speckle' in mode:
                print('Robust Accuracy = {:.3f} with level = {:.2f}'.format(acc, level))
                accuracies_speckle.append(round(acc, 2))
                print("Speckle Accuracies after append :", accuracies_speckle)
            elif 's&p' in mode:
                print('Robust Accuracy = {:.3f} with level = {:.2f}'.format(acc, level))
                accuracies_saltpepper.append(round(acc, 2))
                print("Salt&Pepper Accuracies after append :", accuracies_saltpepper)
                break
            else:
                print("noise mode not supported")
                  
#     gaussian_noise(test_loader)
#     salt_pepper_noise(test_loader)
#     speckle_noise(test_loader)    

    # Change the number of variable as desired number of outputs.
    gaus_001, gaus_01, gaus_05, gaus_1, gaus_2, gaus_3, gaus_4, gaus_5, gaus_6 = (items for items in accuracies_gaussian)
    speck_001, speck_01, speck_05, speck_1, speck_2, speck_3, speck_4, speck_5, speck_6 = (items for items in accuracies_speckle)
    saltpepper = accuracies_saltpepper[0]
    
    # load the top1 error and top5 error from the evaluation results
    f = open("{}/results_epoch{}.txt".format(cfg.OUT_DIR, cfg.OPTIM.MAX_EPOCH), "r")
    c_ids = []
    for i in f.readlines():
        sub_id = list(map(float, i.split(",")))
        c_ids.append(sub_id[3:5])
    topK_errors = [sum(i) / len(c_ids) for i in zip(*c_ids)]
    top1_error, top5_error = topK_errors[0], topK_errors[1]
    
    result_gaussian = ', '.join(
        [str(cfg.RGRAPH.GROUP_NUM), str(cfg.RGRAPH.P), str(cfg.RGRAPH.SPARSITY),
         '{:.3f}'.format(top1_error), '{:.3f}'.format(top5_error),
         str(gaus_001), str(gaus_01), str(gaus_05), str(gaus_1), str(gaus_2), str(gaus_3), str(gaus_4), str(gaus_5), str(gaus_6)])
    result_speck = ', '.join(
        [str(cfg.RGRAPH.GROUP_NUM), str(cfg.RGRAPH.P), str(cfg.RGRAPH.SPARSITY),
         '{:.3f}'.format(top1_error), '{:.3f}'.format(top5_error),
         str(speck_001), str(speck_01), str(speck_05), str(speck_1), str(speck_2), str(speck_3), str(speck_4), str(speck_5), str(speck_6)])
    result_sp = ', '.join(
        [str(cfg.RGRAPH.GROUP_NUM), str(cfg.RGRAPH.P), str(cfg.RGRAPH.SPARSITY),
         '{:.3f}'.format(top1_error), '{:.3f}'.format(top5_error),
         str(saltpepper)])
    
    
    with open("{}/gaus_noise_stats.txt".format(cfg.OUT_DIR), "a") as text_file:
        print(" Writing Text File with accuracies Gaussian:{} ".format(accuracies_gaussian))
        text_file.write(result_gaussian + '\n')
    
    with open("{}/saltpepper_noise_stats.txt".format(cfg.OUT_DIR), "a") as text_file:
        print(" Writing Text File with accuracies Salt & Pepper:{} ".format(accuracies_saltpepper))
        text_file.write(result_sp + '\n')
    
    with open("{}/speckle_noise_stats.txt".format(cfg.OUT_DIR), "a") as text_file:
        print(" Writing Text File with accuracies Speckle:{} ".format(accuracies_speckle))
        text_file.write(result_speck + '\n')


def single_proc_train():
    """Performs single process training."""

    # Setup logging
    lu.setup_logging()

    # Show the config
    logger.info('Config:\n{}'.format(cfg))
    # Setup tensorboard if provided
    writer_train = None
    writer_eval = None
    # If use tensorboard
    if cfg.TENSORBOARD and du.is_master_proc() and cfg.RGRAPH.SEED_TRAIN == cfg.RGRAPH.SEED_TRAIN_START:
        comment = ''
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir_train = os.path.join(cfg.OUT_DIR,
                                    'runs', current_time + comment + '_train')
        logdir_eval = os.path.join(cfg.OUT_DIR,
                                   'runs', current_time + comment + '_eval')
        if not os.path.exists(logdir_train):
            os.makedirs(logdir_train)
        if not os.path.exists(logdir_eval):
            os.makedirs(logdir_eval)
        writer_train = SummaryWriter(logdir_train)
        writer_eval = SummaryWriter(logdir_eval)

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RGRAPH.SEED_TRAIN)
    torch.manual_seed(cfg.RGRAPH.SEED_TRAIN)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # Launch inference + adversarial run
    train_model(writer_eval, is_master=du.is_master_proc())

    if writer_eval is not None:
        # writer_train.close()
        writer_eval.close()


def check_seed_exists(i):
    fname = "{}/results_epoch{}.txt".format(cfg.OUT_DIR, cfg.OPTIM.MAX_EPOCH)
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        if len(lines) > i:
            return True
    return False


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    # cfg.freeze()

    # Ensure that the output dir exists
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Save the config
    dump_cfg()

    for i, cfg.RGRAPH.SEED_TRAIN in enumerate(range(cfg.RGRAPH.SEED_TRAIN_START, cfg.RGRAPH.SEED_TRAIN_END)):
        # check if a seed has been run
        if not check_seed_exists(i):
            print("Launching inference for seed {}".format(i))
            single_proc_train()
        else:
            print('Inference seed {} already exists, stopping inference'.format(cfg.RGRAPH.SEED_TRAIN))

if __name__ == '__main__':
    main()
