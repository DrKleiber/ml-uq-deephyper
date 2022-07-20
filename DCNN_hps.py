# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:37:29 2022

@author: Yang
"""
import ray
import json
import os
import pathlib
import shutil
from glob import glob
import random

import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.DenseNet_2D import DenseNet
from utils.DataLoader_2D import StarDataset

is_gpu_available = torch.cuda.is_available()
n_gpus = torch.cuda.device_count() - 1

# if is_gpu_available:
#     print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
# else:
#     print("No GPU available")
#
# if not(ray.is_initialized()):
#     if is_gpu_available:
#         ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
#     else:
#         ray.init(num_cpus=4, log_to_driver=False)

seed = 1

"perform HPS on a reduced (10%) dataset"
input_filelist = sorted(glob('../Datasets_coarse/input_*0.pt'))
output_filelist = sorted(glob('../Datasets_coarse/output_*0.pt'))

np.random.seed(seed)
random.seed(seed)

" select data files for training, randomly "
train_ratio = 0.7

full_index = range(len(input_filelist))
train_index = random.sample(range(len(input_filelist)),int(train_ratio*len(input_filelist)))
test_index = list(set(full_index) - set(train_index))

input_filelist_train = [
                         input_filelist[i] for i in sorted(train_index)
                         ]
output_filelist_train = [
                         output_filelist[i] for i in sorted(train_index)
                          ]
input_filelist_test = [
                         input_filelist[i] for i in sorted(test_index)
                         ]
output_filelist_test = [
                         output_filelist[i] for i in sorted(test_index)
                         ]

input_train_mean = torch.load('input_train_mean.pt')
input_train_std = torch.load('input_train_std.pt')
output_train_var = torch.load('output_train_var.pt')
output_test_var = torch.load('output_test_var.pt')

train_dataset = StarDataset(input_filelist_train, output_filelist_train)
test_dataset = StarDataset(input_filelist_test, output_filelist_test)

def build_and_train_model(config:dict):

    default_config = {
        'block_1':3,
        'block_2':6,
        'block_3':3,
        'growth_rate':16,
        'init_features':32,
        'dropout_rate': 0.0,
        'out_activation': 'relu',
        'lr': 1e-3,
        'batch_size': 16,
        'weight_decay': 1e-3
    }
    default_config.update(config)

    torch.manual_seed(seed)
    kwargs = {'num_workers': 8,
                  'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=int(default_config['batch_size']), **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=int(default_config['batch_size']), **kwargs)

    n_out_pixels_train = len(train_index) * train_loader.dataset[0][1].numel()
    n_out_pixels_test = len(test_index) * test_loader.dataset[0][1].numel()

    model = DenseNet(in_channels=4, out_channels=1,
                    blocks=(int(default_config['block_1']), int(default_config['block_2']), int(default_config['block_3'])),
                    growth_rate=int(default_config['growth_rate']),
                    init_features=int(default_config['init_features']),
                    drop_rate=default_config['dropout_rate'],
                    bn_size=4,
                    bottleneck=False,
                    out_activation=default_config['out_activation'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=default_config['lr'],
                     weight_decay=default_config['weight_decay'])

    epoch = 100

    logger = {}
    logger['rmse_train'] = []
    logger['rmse_test'] = []

    for i in range(1, epoch+1):

        model.train()
        mse = 0.
        for _, (input_features, output_qois) in enumerate(train_loader):
            "normalize each input feature over the 2-D geometry"
            input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            input_features, output_qois = input_features.to(device), output_qois.to(device)
            model.zero_grad()
            output = model(input_features)
            loss = F.mse_loss(output, output_qois, reduction = 'sum')
            loss.backward()
            optimizer.step()
            mse += loss.item()
        rmse_train = np.sqrt(mse / n_out_pixels_train)
        logger['rmse_train'].append(rmse_train)

        with torch.no_grad():
            model.eval()
            mse = 0.
            for _, (input_features, output_qois) in enumerate(test_loader):
                "normalize each input feature over the 3-D geometry"
                input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                input_features, output_qois = input_features.to(device), output_qois.to(device)
                output = model(input_features)
                mse += F.mse_loss(output, output_qois, reduction = 'sum').item()

            rmse_test = np.sqrt(mse / n_out_pixels_test)
            logger['rmse_test'].append(rmse_test)

    torch.save(model.module.state_dict(), "/model.pth")
    return logger

def run(config):
    logger = build_and_train_model(config)
    return -logger['rmse_test'][-1]

from deephyper.problem import HpProblem

problem = HpProblem()
problem.add_hyperparameter((1e-4, 1e-2, "log-uniform"), "lr", default_value=1e-3)
problem.add_hyperparameter((8, 64), "batch_size", default_value=16)
problem.add_hyperparameter((1e-4, 1e-2, "log-uniform"), "weight_decay", default_value=1e-3)
problem.add_hyperparameter(["sigmoid", "softplus", "relu"], "out_activation", default_value="relu")
problem.add_hyperparameter((0.0, 0.5), "dropout_rate", default_value=0.0)
problem.add_hyperparameter((3,5),"block_1",default_value=3)
problem.add_hyperparameter((4,6),"block_2",default_value=6)
problem.add_hyperparameter((3,5),"block_3",default_value=3)
problem.add_hyperparameter((8, 64),'init_features', default_value = 32)
problem.add_hyperparameter((4, 32),'growth_rate', default_value = 16)

default_config = {
    'block_1':3,
    'block_2':6,
    'block_3':3,
    'growth_rate':16,
    'init_features':32,
    'dropout_rate': 0.0,
    'out_activation': 'relu',
    'lr': 1e-3,
    'batch_size': 16,
    'weight_decay': 1e-3
}

from deephyper.evaluator import Evaluator

evaluator = Evaluator.create(
    run, method="ray", method_kwargs={
        "address": "auto", # tells the Ray evaluator to connect to the already started cluster
        "num_cpus_per_task": 1, #
        "num_gpus_per_task": 1 # automatically compute the number of workers
    }
)

from deephyper.search.hps import CBO
search = CBO(problem, evaluator, initial_points=[problem.default_configuration], log_dir="hps_cbo_results", random_state=42)
results = search.search(max_evals=128)
