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

import foolbox as fb
import art
import art.attacks.evasion as evasion
from art.estimators.classification import PyTorchClassifier

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
    # test_meter.log_epoch_stats(cur_epoch,writer_eval,params,flops)
    test_meter.log_epoch_stats(cur_epoch, writer_eval, params, flops, model, is_master=is_master)
    stats = test_meter.get_epoch_stats(cur_epoch)
    test_meter.reset()
    if cfg.RGRAPH.SAVE_GRAPH:
        adj_dict = nu.model2adj(model)
        adj_dict = {**adj_dict, 'top1_err': stats['top1_err']}
        os.makedirs('{}/graphs/{}'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN), exist_ok=True)
        np.savez('{}/graphs/{}/{}.npz'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN, cur_epoch), **adj_dict)
        
        
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
            if cfg.MODEL.TYPE == 'resnet':
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
    test_loader_adv = loader.construct_test_loader_adv()
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

    # when epsilon=0, 1 --> PGD, epsilon=2, 3 --> CW, otherwise FGSM-->replace eps1, eps2, ... with required epsilon of attack versions
    epsilons = [0, 1, eps1, eps2, ... epsN, 2, 3]

    # Per-channel mean and SD values in BGR order for ImageNet dataset
    cifar10_MEAN = [0.491, 0.482, 0.4465]
    cifar10_SD = [0.247, 0.243, 0.262]
    cifar100_MEAN = [0.507, 0.487, 0.441]
    cifar100_SD = [0.267, 0.256, 0.276]
    imagenet_MEAN = [0.406, 0.456, 0.485]
    imagenet_SD = [0.225, 0.224, 0.229]

    accuracies = []
    # replace the MEAN and SD variable in the following line for the relevant dataset.
    norm_layer = Normalize(mean=cifar10_MEAN, std=cifar10_SD)
    net = torch.nn.Sequential(norm_layer, model).cuda()
#     net = torch.nn.Sequential(norm_layer, PrintLayer(), model).cuda()
    net = net.eval()
    print("Adversarial Loader Batch Size =", test_loader_adv.batch_size)
    for epsilon in epsilons:
        if epsilon == 0:
            print("Running PGD Attack")
            atk_ta = torchattacks.PGD(net, eps=6/255, alpha=2/255, steps=7)  # for relevant dataset, use parameters from torchattacks official notebook
        elif epsilon == 1:
            print("Running PGD Attack")
            atk_ta = torchattacks.PGD(net, eps=9/255, alpha=2/255, steps=7)  # for relevant dataset, use parameters from torchattacks official notebook
        elif epsilon == 2:
            print("Running Torchattacks.CW")
            atk_ta = torchattacks.CW(net, c=0.15, kappa=0, steps=100, lr=0.01) # replace the values of c and steps according to hyperparameters reported in the paper.
        elif epsilon == 3:
            print("Running Torchattacks.CW")
            atk_ta = torchattacks.CW(net, c=0.25, kappa=0, steps=100, lr=0.01) # replace the values of c and steps according to hyperparameters reported in the paper.
            
            # For Foolbox or ART attacks, uncomment the following lines.
#             print("-> FoolBox.CW")
#             fmodel = fb.PyTorchModel(net, bounds=(0, 1))
#             atk_fb = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=1, initial_const=0.05,
#                                            confidence=0, steps=100, stepsize=0.01)
            
#             print("-> Adversarial Robustness Toolbox.CW")
#             classifier = PyTorchClassifier(model=net, clip_values=(0, 1),
#                                    loss=loss_fun,
#                                    optimizer=optimizer,
#                                    input_shape=(3, 32, 32), nb_classes=10)
#             atk_art = evasion.CarliniL2Method(batch_size=1, classifier=classifier, 
#                                   binary_search_steps=1, initial_const=0.05,
#                                   confidence=0, max_iter=100,
#                                   learning_rate=0.01)
        else:
            print("Running FGSM Attacks on epsilon :", epsilon)
            atk_ta = torchattacks.FGSM(net, eps=epsilon)
        ctr = 0
        correct_ta = 0
#         correct_fb = 0
#         correct_art = 0
        total = 0
        for cur_iter, (inputs, labels) in enumerate(test_loader_adv):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.float().div(255)
            adv_images_ta = atk_ta(inputs, labels)
#             _, adv_images_fb, _ = atk_fb(fmodel, inputs, labels, epsilons=1)
#             adv_images_art = torch.tensor(atk_art.generate(inputsnp, labelsnp)).cuda()
            
          
            adv_inputs_ta = adv_images_ta.float()
#             adv_inputs_fb = adv_images_fb.float()
#             adv_inputs_art = adv_images_art.float()
            
            outputs_ta = net(adv_inputs_ta)
#             outputs_fb = net(adv_inputs_fb)
#             outputs_art = net(adv_inputs_art)

            _, predicted_ta = torch.max(outputs_ta.data, 1)
#             _, predicted_fb = torch.max(outputs_fb.data, 1)
#             _, predicted_art = torch.max(outputs_art.data, 1)

            ctr += 1
            total += 1
            correct_ta += (predicted_ta == labels).sum()
#             correct_fb += (predicted_fb == labels).sum()
#             correct_art += (predicted_art == labels).sum()
    
            if ctr > X: # replace X with the number of images to be generated for adversarial attacks.
                print(ctr, " images done for epsilon:", epsilon)
                break
        acc_ta = 100 * float(correct_ta) / total
#         acc_fb = 100 * float(correct_fb) / total
#         acc_art = 100 * float(correct_art) / total

        print("ta acc =", round(acc_ta, 2), ", ta correct =", float(correct_ta), ", total =", total)
#         print("fb acc =", round(acc_fb, 2), ", fb correct =", float(correct_fb), ", total =", total)
#         print("art acc =", round(acc_art, 2), ", art correct =", float(correct_art), ", total =", total)

        accuracies.append(round(acc_ta, 2))
        print('Attack Accuracy = {:.3f} with epsilon = {:.2f}'.format(acc_ta, epsilon))
        print("accuracies after apend :", accuracies)

    # save items inside accuracies list to separate float objects, update the # of variables according to requirement.
    accPGD_6by255, accPGD_9by255, accFGSM1, accFGSM2, accFGSM3, accFGSM4, accFGSM5, accCW_15, accCW_25 = (items for items in accuracies)
    
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
         str(accPGD_6by255), str(accPGD_9by255), str(accFGSM1), str(accFGSM2), str(accFGSM3), str(accFGSM4), str(accFGSM5),
         str(accCW_15), str(accCW_25)])
    # 
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
            print('Trained seed {} already exists, stopping inference'.format(cfg.RGRAPH.SEED_TRAIN))

if __name__ == '__main__':
    main()
