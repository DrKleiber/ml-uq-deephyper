# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:37:29 2022

@author: Yang
"""

import json
import os
import pathlib
import shutil

import torch
from torch import nn
import numpy as np

is_gpu_available = torch.cuda.is_available()
n_gpus = torch.cuda.device_count() - 1

if is_gpu_available:
    print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
else:
    print("No GPU available")
    
import ray

if not(ray.is_initialized()):
    if is_gpu_available:
        ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
    else:
        ray.init(num_cpus=4, log_to_driver=False)
        
        