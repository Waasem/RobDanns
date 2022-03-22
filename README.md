# Under construction site, shall be updated shortly for the Nature Machine Intelligence article submission.

# RobDanns
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4.0-blue)](https://pytorch.org/get-started/previous-versions/#v140)
[![Matlab](https://img.shields.io/badge/MATLAB-R2020a-green)](https://www.mathworks.com/products/new_products/release2020a.html)
[![DOI](https://img.shields.io/badge/DOI-%3Cpending%3E-red)]()
[![Predecessor](https://img.shields.io/badge/Predecessor-graph2nn-yellow)](https://github.com/facebookresearch/graph2nn)
[![Pycls](https://img.shields.io/badge/pycls-FAIR-orange)](https://github.com/facebookresearch/pycls)


This repository is the official implementation of our article, `Exploring Robust Architectures for Deep Artificial Neural Networks`, under review at Nature Machine Intelligence. 

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [License](./LICENSE)
- [Issues](https://github.com/ebridge2/lol/issues)
- [Citation](#citation)

## Overview

Deep artificial neural networks (DANNs) have varying predictive performance when their architectures change. How robustness is related to the architecture of DANN is still under-explored topic. In this work, we study the robustness of DANNs vis-a-vis their underlying graph architectures or structures. `RobDANNs` (Robust DANNs) are the architecture of neural networks that have high robustness properties against natural and malicious noise in the input data. This study encompasses three parts:

- Explore the design space of architectures of DANNs using graph-theoretic robustness measures.
- Transform the graphs to DANN architectures to train/validate/test on various image classification tasks.
- Explore the relationship between the robustness of trained DANNs against noise and adversarial attacks and the robustness of their underlying architectures estimated via graph-theoretic measures.

This work shows that the topological entropy and Olivier-Ricci curvature of the underlying graphs can quantify the robustness of DANNs. This relationship is stronger for complex tasks and large/deep DANNs. Our work offers a novel approach for autoML and neural architecture search community to quantify robustness of DANNs through graph curvature and entropy. We believe that this work will have a high impact on the acceptability of safety-critical AI applications.

**TLDR**: We explore neural networks for their robustness against adversarial attacks through the lens of graph theory. The graph theoretic tools successfully employed in network science (NetSci) have been used to quantify robustness of artificial neural networks (ANNs).

<div align="center">
        <img src="./deep_learning/docs/figs/overview.png" width="1100px" />
        <p align="center"><b>Overview of our research.</b> Graph measures that have successfully quantified network robustness in NetSci domain (a), can also quantify robustness of Aritifical Neural Networks in Deep Learning domain (b).</p>
</div>

## Repo Contents

The repository has two parts:
-	[Deep learning](deep_learning): `Deep learning code` for exploring the DANNs using [WS-flex](https://arxiv.org/abs/2007.06559) random graph generator.
-	[Graph theory](graph_theory): `Graph theory code` where graph theoretical properties of architectures of DANNs are calculated.

## Code setup

### 1 WS-flex / Relational Graphs

Given in [wsflex](wsflex) directory. The repository is heavily built upon **[pycls](https://github.com/facebookresearch/pycls)**, an image classification codebase built by FAIR. The repo itself is a modified version of **[graph2nn]( https://github.com/facebookresearch/graph2nn)** linked to the paper [Graph Structure of Neural Networks](https://arxiv.org/abs/2007.06559).

**Requirements:**

- NVIDIA GPU, Linux, Python3
- PyTorch, various Python packages; Instructions for installing these dependencies are found below

**Python environment:**
We recommend using Conda or virtualenv package manager

```bash
conda create -n graph2robnn python=3.6
conda activate graph2robnn
```

**Pytorch:**
Manually install [PyTorch](https://pytorch.org/) with **CUDA** support (CPU version is not supported). 
We have verified under PyTorch 1.4.0 and torchvision 0.5.0. For example:
```bash
pip install torch==1.4.0 torchvision==0.5.0 torchattacks==2.6
``` 

**Clone graph2robnn repository and install:**

```bash
git clone https://github.com/Waasem/Exploring-Robustness-of-NNs-through-Graph-Measures.git
cd Exploring-Robustness-of-NNs-through-Graph-Measures/wsflex
pip install -r requirements.txt
python setup.py develop
```
**Download datasets:**

Download [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and/or [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and/or [TinyImageNet dataset]( http://cs231n.stanford.edu/tiny-imagenet-200.zip).
Uncompress the datasets then link the datasets with our code base. We have already downloaded the TinyImageNet dataset in the [wsflex/pycls/datasets/data/ tinyimagenet200](wsflex/pycls/datasets/data/ tinyimagenet200) directory. For the rest of the datasets, use following commands.

```bash
# CIFAR-10
mkdir -p pycls/datasets/data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz # or other data directory 
tar -xvf cifar-10-python.tar.gz
DIR=$(readlink -f cifar-10-batches-py)
ln -s $DIR pycls/datasets/data/cifar10 # symlink
# CIFAR-100 (optional)
mkdir -p pycls/datasets/data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz # or other data directory 
tar -xvf cifar-100-python.tar.gz
DIR=$(readlink -f cifar-100-python)
ln -s $DIR pycls/datasets/data/cifar100 # symlink
# TinyImageNet (optional)
ln -s path/tinyimagenet200 pycls/datasets/data/tinyimagenet200 # symlink
```

## Run the code

A clean repo for reproducing multiple experimental setups related to WS-flex models in our paper is provided here.
We have made all the raw experimental configs related to Cifar-10 and Cifar-100 datasets available in [wsflex/configs/baselines/](wsflex/configs/baselines) directory. For individual experiments on TinyImageNet, the config.yaml files in each sub-directory of [results/training-outputs/wsflex/resnet18_tinyimagenet/](results/training-outputs/wsflex/resnet18_tinyimagenet/) can be used as training configuration. The results for these results are available in [results/training-outputs/wsflex](results/training-outputs/wsflex), . Therefore, you may directly skip this step to play with these results.

###  Run the model for training
Running the models training through following commands
```bash
# Explaining the arguments:
# Tasks: mlp_cifar10, cnn_ cifar10, cnn_cifar100, resnet18_tinyimagenet
# Division: all
# GPU: 1
bash launch_MLP_CIFAR10.sh mlp_cifar10 all 1
bash launch_CNN_CIFAR10.sh cnn_cifar10 all 1
bash launch_CNN_CIFAR100.sh cnn_cifar100 all 1
bash launch-ResNet-TinyImageNet.sh resnet18_tinyimagenet all 1
```
### 2 RandWire Networks

Given in [randwire](randwire) directory. The repository is based upon the paper **[RandWire](https://arxiv.org/abs/1904.01569)** that implements neural networks using ER, BA, and WS graph generators. There are separate script files in the project directory for multiple runs of each of the graph type:
-	BA Model : [multi_runs_BA.sh](randwire/multi_runs_BA.sh)
-	ER Model : [multi_runs_ER.sh](randwire/multi_runs_ER.sh)
-	WS Model : [multi_runs_WS_1.sh](randwire/multi_runs_WS_1.sh), [multi_runs_WS_2.sh](randwire/multi_runs_WS_2.sh), [multi_runs_WS_3.sh](randwire/multi_runs_WS_3.sh), [multi_runs_WS_4.sh](randwire/multi_runs_WS_4.sh)

**Requirements:**

We recommend using Conda or virtualenv package manager

```bash
conda create -n randwire python=3.8
conda activate randwire
```

**Pytorch:**
Manually install [PyTorch](https://pytorch.org/) with **CUDA** support.

**Clone graph2robnn repository and install:**

```bash
git clone https://github.com/Waasem/Exploring-Robustness-of-NNs-through-Graph-Measures.git
cd Exploring-Robustness-of-NNs-through-Graph-Measures/randwire
pip install -r requirements.txt
```

### 3 Graph Calculations
Given in [graphcalcs](graphcalcs) directory. This directory is based on Matlab. Here we calculate the graph measures of our neural networks. We recommend using [Matlab R_2020a]( https://www.mathworks.com/products/new_products/release2020a.html).  Because of the large number of experiments and huge space requirements for files containing the Adjacency Matrices, we have not uploaded these files. However, here we present a method to find the adjacency matrices of the graphs.

**Code Setup:**
Before running this code part, we need to find and store the Adjacency Matrices of our graphs. For the Relational Graphs, we have placed a notebook file [generate_graphs.ipynb](wsflex/analysis/generate_graphs.ipynb) that saves the Adjacency matrices for all of our WS-flex graphs. For other graphs, similar methodology can be followed.

Once we have Adjacency Matrices of our graphs, we can feed them as input to the file [
main.m](graphcalcs/main.m). For each of the graph models, users can comment / uncomment the code blocks in this file and run the code. Note: this can be time consuming process.

## Results

All results of our experiments are placed in the directory [results](results/training-outputs/). Users can view the training results of each category of neural networks in this directory for reference.

## Plots

The plots reported in our paper are located in the directory [plots](plots).
-	CIFAR-10 Plots: For this dataset, the reported plots can be found in the notebook [Cifar-10_plots](plots/cifar-10/Cifar-10_results_reported.ipynb).
-	CIFAR-100 Plots: For this dataset, the reported plots can be found in the notebook [Cifar-100_plots](plots/cifar-100/Cifar-100_results_reported.ipynb).
-	Tiny ImageNet-200 Plots: For this dataset, the reported plots can be found in the notebook [Tiny-ImagetNet-200_plots](plots/tiny-imagenet-200/TinyImageNet_results_reported.ipynb).

## Thank You !

