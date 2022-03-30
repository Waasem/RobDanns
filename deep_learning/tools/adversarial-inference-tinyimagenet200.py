#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the original graph2nn github repo.

# File modifications and additions by Rowan AI Lab, licensed under the Creative Commons Zero v1.0 Universal
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""
from __future__ import print_function
import argparse
import numpy as np
import os
import sys
import torch
import multiprocessing as mp
import math
import pdb
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pycls.config import assert_cfg
from pycls.config import cfg
from pycls.config import dump_cfg
from pycls.datasets import loader
from pycls.models import model_builder
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from PIL import Image

import pycls.models.losses as losses
import pycls.models.optimizer as optim
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu
import pycls.utils.net as nu
import pycls.datasets.paths as dp
import time

from datetime import datetime
from tensorboardX import SummaryWriter

print("Let's use GPU :", torch.cuda.current_device())
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

# TEST/VAL DATA_LOADER FOR TINY_IMAGENET200
def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0, len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames, classes

def load_allimages(dir):
    images = []
    if not os.path.isdir(dir):
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            # if datasets.folder.is_image_file(fname):
            if datasets.folder.has_file_allowed_extension(fname,['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images

class TinyImageNet(torch.utils.data.Dataset):
    """ TinyImageNet200 validation dataloader."""
    
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])
    
    def __getitem__(self, index):
        """inputs: Index, retrns: tuple(im, label)"""
        
        img = None
        with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        label = self.classidx[index]
        return img, label
    
    def __len__(self):
        return len(self.imgs)
    

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
    # test_meter.log_epoch_stats(cur_epoch,writer_eval,params,flops)
    test_meter.log_epoch_stats(cur_epoch, writer_eval, params, flops, model, is_master=is_master)
    eval_stats = test_meter.get_epoch_stats(cur_epoch)
    test_meter.reset()
    if cfg.RGRAPH.SAVE_GRAPH:
        adj_dict = nu.model2adj(model)
        adj_dict = {**adj_dict, 'top1_err': eval_stats['top1_err']}
        os.makedirs('{}/graphs/{}'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN), exist_ok=True)
        np.savez('{}/graphs/{}/{}.npz'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN, cur_epoch), **adj_dict)
#     return eval_stats

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1,3,1,1)
        std = self.std.reshape(1,3,1,1)
        norm_img = (input - mean) / std
        return norm_img

# Helper class for printing model layers
class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x


def train_model(writer_train=None, writer_eval=None, is_master=False):
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
    data_path = dp.get_data_path(cfg.TRAIN.DATASET)     # Retrieve the data path for the dataset
    traindir = os.path.join(data_path, cfg.TRAIN.SPLIT)
    valdir = os.path.join(data_path, cfg.TEST.SPLIT, 'images')
    valgtfile = os.path.join(data_path, cfg.TEST.SPLIT, 'val_annotations.txt')    
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # create training dataset and loader
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=True)

    # create validation dataset
    test_dataset = TinyImageNet(
        valdir,
        valgtfile,
        class_to_idx=train_loader.dataset.class_to_idx.copy(),
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize]))
    
    # create validation loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False)
    
    # create adversarial dataset
    adv_dataset = TinyImageNet(
        valdir,
        valgtfile,
        class_to_idx=train_loader.dataset.class_to_idx.copy(),
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()]))
    
    # create adversarial loader
    test_loader_adv = torch.utils.data.DataLoader(
        adv_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False)


    # Create meters
    test_meter = TestMeter(len(test_loader))
    test_meter_adv = TestMeter(len(test_loader_adv))

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

    # when epsilon=0 --> PGD, epsilon=1 --> CW, otherwise FGSM-->replace eps1, eps2, ... with required epsilon of attack versions
    epsilons = [0, eps1, eps2, ... epsN, 1]
    
    # Per-channel mean and SD values in BGR order for TinyImageNet dataset
    tinyimagenet_MEAN = [0.485, 0.456, 0.406]
    tinyimagenet_SD = [0.229, 0.224, 0.225]
    
    accuracies = []
    
    # add normalization layer to the model
    norm_layer = Normalize(mean=tinyimagenet_MEAN, std=tinyimagenet_SD)
    net = torch.nn.Sequential(norm_layer, model).cuda()
    net = net.eval()
    for epsilon in epsilons:
        if epsilon == 0:
            print("Running PGD Attack")
            atk = torchattacks.PGD(net, eps=1/510, alpha=2/225, steps=7) # for relevant dataset, use parameters from torchattacks official notebook
        elif epsilon == 1:
            print("Running CW Attack")
            atk = torchattacks.CW(net, c=0.1, kappa=0, steps=100, lr=0.01) # choose suitable values for c, kappa, steps, and lr.
        else:
            print("Running FGSM Attacks on epsilon :", epsilon)
            atk = torchattacks.FGSM(net, eps=epsilon)
        ctr = 0
        correct = 0
        total = 0
        for cur_iter, (inputs, labels) in enumerate(test_loader_adv):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            adv_images = atk(inputs, labels)
            outputs = net(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            ctr += 1
            total += 1
            correct += (predicted == labels).sum()
            
            if ctr > X: # replace X with the number of images to be generated for adversarial attacks.
                print(ctr, " images done for epsilon:", epsilon)
                break
        acc = 100 * float(correct) / total
        print("acc =", round(acc, 2), "correct =", float(correct), "total =", total)

        accuracies.append(round(acc, 2))
        print('Attack Accuracy = {:.3f} with epsilon = {:.4f}'.format(acc, epsilon))
        print("accuracies after apend :", accuracies)        

    # save items inside accuracies list to separate float objects, update the # of variables according to requirement.
    accPGD, accFGSM1, accFGSM2, accFGSM3, accFGSM4, accFGSM5, accFGSM6, accFGSM7, accCW = (items for items in accuracies)
    
    # load the top1 error and top5 error from the evaluation results
    f = open("{}/results_epoch{}.txt".format(cfg.OUT_DIR, cfg.OPTIM.MAX_EPOCH), "r")
    c_ids = []
    for i in f.readlines():
        sub_id = list(map(float, i.split(",")))
        c_ids.append(sub_id[3:5])
    topK_errors = [sum(i) / len(c_ids) for i in zip(*c_ids)]
    top1_error, top5_error = topK_errors[0], topK_errors[1]
    
    result_info = ', '.join(
        [str(cfg.RGRAPH.GROUP_NUM), str(cfg.RGRAPH.P), str(cfg.RGRAPH.SPARSITY),
         '{:.3f}'.format(top1_error), '{:.3f}'.format(top5_error),
         str(accPGD), str(accFGSM1), str(accFGSM2), str(accFGSM3), str(accFGSM4), str(accFGSM5),
         str(accFGSM6), str(accFGSM7), str(accCW)])
    with open("{}/stats.txt".format(cfg.OUT_DIR), "a") as text_file:
        print(" Writing Text File with accuracies {} ".format(accuracies))
        text_file.write(result_info + '\n')


def single_proc_train():
    """Performs single process training."""
  
    # Setup logging
    lu.setup_logging()

    # Show the config
    logger.info('Config:\n{}'.format(cfg))
    # Setup tensorboard if provided
    writer_train = None
    writer_eval = None
    ## If use tensorboard
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
    train_model(writer_train, writer_eval, is_master=du.is_master_proc())
    if writer_train is not None and writer_eval is not None:
        writer_train.close()
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
