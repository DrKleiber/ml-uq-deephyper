#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:00:02 2020

@author: yang.liu
"""
import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.DenseNet_ensemble import DenseNet
from utils.DataLoader_2D import StarDataset
# from utils.plot import plot_prediction_det, save_stats
import json
from time import time
import random
from args import args, device
import pickle
import torch.nn.functional as F

#args.train_dir = args.run_dir + "/training"
#args.pred_dir = args.train_dir + "/predictions"
#mkdirs([args.train_dir, args.pred_dir])
  
# load data
input_filelist = sorted(glob('./tensor_sam/input_*.pt'))
output_filelist = sorted(glob('./tensor_sam/output_*.pt'))

random.seed(args.seed)
" select data files for training, randomly "
full_index = range(len(input_filelist))

train_index = random.sample(range(len(input_filelist)),int(0.8*len(input_filelist)))

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

# "calculate the mean and std of each feature in the training dataset"

# input_train_tensor = []
# output_train_tensor = []
# output_test_tensor = []

# input_idx = [0,2,3,4]  # remove velocity j component from inputs

# for i in input_filelist_train:
#     tmp = torch.load(i)
#     tmp = tmp[input_idx,16,:,:]
#     input_train_tensor.append(tmp)

# for i in output_filelist_train:
#     tmp = torch.load(i)
#     tmp = tmp[0,16,:,:]
#     output_train_tensor.append(tmp)

# for i in output_filelist_test:
#     tmp = torch.load(i)
#     tmp = tmp[0,16,:,:]
#     output_test_tensor.append(tmp)
     
# input_train_tensor = torch.stack(input_train_tensor)
# output_train_tensor = torch.stack(output_train_tensor)  
# output_test_tensor = torch.stack(output_test_tensor)

# input_train_mean = input_train_tensor.mean(dim=[0,2,3])
# input_train_std = input_train_tensor.std(dim=[0,2,3])

# output_train_var = output_train_tensor.var()
# output_test_var = output_test_tensor.var()

# torch.save(input_train_mean, 'input_train_mean.pt')
# torch.save(input_train_std, 'input_train_std.pt')

# torch.save(output_train_var, 'output_train_var.pt')
# torch.save(output_test_var, 'output_test_var.pt')

# del input_train_tensor
# del output_train_tensor
# del output_test_tensor

input_train_mean = torch.load('sam_input_train_mean.pt')
input_train_std = torch.load('sam_input_train_std.pt')
output_train_var = torch.load('sam_output_train_var.pt')
output_test_var = torch.load('sam_output_test_var.pt')

train_dataset = StarDataset(input_filelist_train, output_filelist_train)
test_dataset = StarDataset(input_filelist_test, output_filelist_test)

kwargs = {'num_workers': 8,
              'pin_memory': True} if torch.cuda.is_available() else {}

torch.manual_seed(args.seed)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

torch.manual_seed(args.seed)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

print('Loaded data!')

n_out_pixels_train = len(train_index) * train_loader.dataset[0][1].numel()
n_out_pixels_test = len(test_index) * test_loader.dataset[0][1].numel()

logger = {}
logger['nll_train'] = []
logger['nll_test'] = []
logger['nll_adv'] = []
logger['r2_train'] = []
logger['r2_test'] = []
logger['r2_adv'] = []
logger['rmse_train'] = []
logger['rmse_test'] = []
logger['rmse_adv'] = []

# initialize model
torch.manual_seed(args.seed)
model = DenseNet(in_channels=args.in_channels, out_channels=2, 
                blocks=args.blocks,
                growth_rate=args.growth_rate, 
                init_features=args.init_features,
                drop_rate=args.drop_rate,
                bn_size=args.bn_size,
                bottleneck=args.bottleneck,
                out_activation='Softplus')
print(model)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.to(device)


# define training process
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                 weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-6)

" Proper scoring rule using negative log likelihood scoring rule"

nll_loss = lambda mu, sigma, y: torch.log(sigma)/2 + ((y-mu)**2)/(2*sigma)
sp = torch.nn.Softplus()

def adv_example(model, X, y, epsilon=0.1, alpha=0.02, num_iter=10):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        model.zero_grad()
        y_adv = model(X + delta)        
        mu, sig = y_adv[:,0,:,:], sp(y_adv[:,1,:,:]) + 1e-6 
        adv_loss = torch.sum(nll_loss(mu,sig, y.squeeze(1)))  
        adv_loss.backward()
        delta.data = (delta + alpha*delta.grad.sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def train(epoch):
    model.train()
    nll_total = 0.
    mse  = 0.
    for input_features, output_qois in train_loader:
        
        "normalize each input feature over the 2-D geometry"           
        input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        input_features, output_qois = input_features.to(device), output_qois.to(device)

        optimizer.zero_grad()        
        output = model(input_features)        
        mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6 
        loss = torch.sum(nll_loss(mu,sig, output_qois.squeeze(1)))        
        loss.backward()
        
        optimizer.step()
        nll_total += loss.item()
        mse += F.mse_loss(mu, output_qois.squeeze(1), reduction = 'sum').item()

    nll_train = nll_total / n_out_pixels_train
    scheduler.step(nll_train)

    rmse_train = np.sqrt(mse / n_out_pixels_train)
    r2_train = 1 - mse /n_out_pixels_train/ output_train_var[0]
    r2_train = r2_train.numpy()
    
    print("epoch: {}, train nll: {:.6f}".format(epoch, nll_train))
    print("epoch: {}, train r2: {:.6f}".format(epoch, r2_train))
    print("epoch: {}, train rmse: {:.6f}".format(epoch, rmse_train))

    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_train)
        logger['nll_train'].append(nll_train)
        logger['rmse_train'].append(rmse_train)
        f = open(args.run_dir + '/' + 'nll_train.pkl',"wb")
        pickle.dump(logger['nll_train'],f)
        f.close()
        f = open(args.run_dir + '/' + 'r2_train.pkl',"wb")
        pickle.dump(logger['r2_train'],f)
        f.close()
        f = open(args.run_dir + '/' + 'rmse_train.pkl',"wb")
        pickle.dump(logger['rmse_train'],f)
        f.close()
    # save model
    if epoch % args.ckpt_freq == 0:
        tic2 = time()
        print("Trained 100 epochs with using {} seconds".format(tic2 - tic))
        torch.save(model.module.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))
        
def train_adversarial(epoch):
    model.train()
    nll_total = 0.
    mse = 0.
    for input_features, output_qois in train_loader:
        
        "normalize each input feature over the 2-D geometry"           
        input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features, output_qois = input_features.to(device), output_qois.to(device)        
        
        optimizer.zero_grad()        
        output = model(input_features)        
        mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6 
        # loss = torch.sum(nll_loss(mu,sig, output_qois.squeeze(1)))        
        # loss.backward(retain_graph=True)
        
        "generate adversarial sample"
        delta = adv_example(model, input_features, output_qois)

        model.zero_grad()
        output_adv = model(input_features + delta)        
        mu_adv, sig_adv = output_adv[:,0,:,:], sp(output_adv[:,1,:,:]) + 1e-6 
        "Compute loss as sum of l(x) + l(x')"        
        loss = torch.sum(nll_loss(mu,sig, output_qois.squeeze(1))) + torch.sum(nll_loss(mu_adv,sig_adv, output_qois.squeeze(1)))
        
        loss.backward()
        optimizer.step()
        nll_total += loss.item()
        mse += F.mse_loss(mu, output_qois.squeeze(1), reduction = 'sum').item()

    nll_train = nll_total / n_out_pixels_train
    scheduler.step(nll_train)
    
    rmse_train = np.sqrt(mse / n_out_pixels_train)
    r2_train = 1 - mse /n_out_pixels_train/ output_train_var[0]
    r2_train = r2_train.numpy()
    print("epoch: {}, train adversarial nll: {:.6f}".format(epoch, nll_train))
    print("epoch: {}, train adversarial r2: {:.6f}".format(epoch, r2_train))
    print("epoch: {}, train adversarial rmse: {:.6f}".format(epoch, rmse_train))


    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_train)
        logger['nll_train'].append(nll_train)
        logger['rmse_train'].append(rmse_train)
        f = open(args.run_dir + '/' + 'nll_train.pkl',"wb")
        pickle.dump(logger['nll_train'],f)
        f.close()
        f = open(args.run_dir + '/' + 'r2_train.pkl',"wb")
        pickle.dump(logger['r2_train'],f)
        f.close()
        f = open(args.run_dir + '/' + 'rmse_train.pkl',"wb")
        pickle.dump(logger['rmse_train'],f)
        f.close()
    # save model
    if epoch % args.ckpt_freq == 0:
        tic2 = time()
        print("Trained 100 epochs with using {} seconds".format(tic2 - tic))
        torch.save(model.module.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))

def test(epoch):
    model.eval()
    nll_total = 0.
    mse = 0.
    for input_features, output_qois in test_loader:
        
        "normalize each input feature over the 2-D geometry"           
        input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        input_features, output_qois = input_features.to(device), output_qois.to(device)

        output = model(input_features)
        
        mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6 
        loss = torch.sum(nll_loss(mu,sig, output_qois.squeeze(1)))        

        nll_total += loss.item()
        mse += F.mse_loss(mu, output_qois.squeeze(1), reduction = 'sum').item()


    nll_test = nll_total / n_out_pixels_test
    
    rmse_test = np.sqrt(mse / n_out_pixels_test)

    r2_test = 1 - mse/n_out_pixels_test/output_test_var[0]
    r2_test = r2_test.numpy()
    print("epoch: {}, test nll: {:.6f}".format(epoch, nll_test))
    print("epoch: {}, test r2: {:.6f}".format(epoch, r2_test))
    print("epoch: {}, test rmse: {:.6f}".format(epoch, rmse_test))

    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_test)
        logger['nll_test'].append(nll_test)
        logger['rmse_test'].append(rmse_test)
        f = open(args.run_dir + '/' + 'nll_test.pkl',"wb")
        pickle.dump(logger['nll_test'],f)
        f.close()
        f = open(args.run_dir + '/' + 'r2_test.pkl',"wb")
        pickle.dump(logger['r2_test'],f)
        f.close()
        f = open(args.run_dir + '/' + 'rmse_test.pkl',"wb")
        pickle.dump(logger['rmse_test'],f)
        f.close()
        
def test_adversarial(epoch):
    model.eval()
    nll_total = 0.
    mse = 0.
    for input_features, output_qois in test_loader:
        "normalize each input feature over the 3-D geometry"
        input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features, output_qois = input_features.to(device), output_qois.to(device)
                
        delta = adv_example(model, input_features, output_qois)
        
        output_adv = model(input_features + delta)        
        mu_adv, sig_adv = output_adv[:,0,:,:], sp(output_adv[:,1,:,:]) + 1e-6 
        loss = torch.sum(nll_loss(mu_adv,sig_adv, output_qois.squeeze(1)))        

        nll_total += loss.item()
        mse += F.mse_loss(mu_adv, output_qois.squeeze(1), reduction = 'sum').item()

    nll_adv = nll_total /n_out_pixels_test
    rmse_adv = np.sqrt(mse / n_out_pixels_test)

    r2_adv = 1 - mse/n_out_pixels_test/output_test_var[0]
    r2_adv = r2_adv.numpy()
    print("epoch: {}, test adversarial nll: {:.6f}".format(epoch, nll_adv))
    print("epoch: {}, test adversarial r2: {:.6f}".format(epoch, r2_adv))
    print("epoch: {}, test adversarial rmse: {:.6f}".format(epoch, rmse_adv))

    if epoch % args.log_freq == 0:
        logger['r2_adv'].append(r2_adv)
        logger['nll_adv'].append(nll_adv)
        logger['rmse_adv'].append(rmse_adv)
        f = open(args.run_dir + '/' + 'nll_adv.pkl',"wb")
        pickle.dump(logger['nll_adv'],f)
        f.close()
        f = open(args.run_dir + '/' + 'r2_adv.pkl',"wb")
        pickle.dump(logger['r2_adv'],f)
        f.close()
        f = open(args.run_dir + '/' + 'rmse_adv.pkl',"wb")
        pickle.dump(logger['rmse_adv'],f)
        f.close()

print('Start training........................................................')
tic = time()
for epoch in range(1, args.epochs + 1):
    train_adversarial(epoch)
#    train(epoch)
    test_adversarial(epoch)
    with torch.no_grad():
        test(epoch)
tic2 = time()
print("Finished training {} epochs with {} data using {} seconds"
      .format(args.epochs, args.ntrain, tic2 - tic))

args.training_time = tic2 - tic
args.n_params, args.n_layers = model._num_parameters_convlayers()

with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)



