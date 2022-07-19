# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:29:17 2022

@author: Yang
"""
import torch

input_train_mean = torch.load('input_train_mean.pt')
input_train_std = torch.load('input_train_std.pt')
output_train_var = torch.load('output_train_var.pt')
output_test_var = torch.load('output_test_var.pt')