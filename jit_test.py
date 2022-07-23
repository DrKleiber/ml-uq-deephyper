# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:53:04 2022

@author: Yang
"""
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

model = DenseNet(in_channels=4, out_channels=1,
                blocks=(3,6,3),
                growth_rate=16,
                init_features=16,
                drop_rate=0.0,
                bn_size=4,
                bottleneck=False,
                out_activation='relu')

input_features = torch.Tensor(8, 4, 32,128)

torch.jit.save(torch.jit.trace(model, input_features), "./{}.pth".format(1))

loaded_model = torch.jit.load("1.pth")

b = torch.Tensor(2, 4, 32,128)

loaded_model.eval()

a = loaded_model(b)