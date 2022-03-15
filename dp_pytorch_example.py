# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:44:53 2022

@author: Yang
"""

import ray
import json
import pandas as pd
from functools import partial

import torch

from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from torch import nn

from torchtext.datasets import AG_NEWS

def load_data(train_ratio):
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * train_ratio)
    split_train, split_valid = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    return split_train, split_valid, test_dataset

train_iter = AG_NEWS(split='train')
num_class = 4

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


def collate_batch(batch, device):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def train(model, criterion, optimizer, dataloader):
    model.train()

    for _, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


def run(config: dict):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  embed_dim = 64

  collate_fn = partial(collate_batch, device=device)
  split_train, split_valid, _ = load_data(0.3)
  train_dataloader = DataLoader(split_train, batch_size=int(config["batch_size"]),
                              shuffle=True, collate_fn=collate_fn)
  valid_dataloader = DataLoader(split_valid, batch_size=int(config["batch_size"]),
                              shuffle=True, collate_fn=collate_fn)

  model = TextClassificationModel(vocab_size, int(embed_dim), num_class).to(device)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

  for _ in range(1, int(config["num_epochs"]) + 1):
      train(model, criterion, optimizer, train_dataloader)

  accu_test = evaluate(model, valid_dataloader)
  return accu_test


# quick_run = get_run(train_ratio=0.3)
# perf_run = get_run(train_ratio=0.95)

# We define a dictionnary for the default values
default_config = {
    "num_epochs": 10,
    "batch_size": 64,
    "learning_rate": 5,
}

# We launch the Ray run-time and execute the `run` function
# with the default configuration

is_gpu_available = torch.cuda.is_available()
n_gpus = torch.cuda.device_count()

if is_gpu_available:
    if not(ray.is_initialized()):
        ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)

    run_default = ray.remote(num_cpus=1, num_gpus=1)(run(default_config))
    objective_default = ray.get(run_default.remote())
else:
    if not(ray.is_initialized()):
        ray.init(num_cpus=1, log_to_driver=False)
    objective_default = run(default_config)

print(f"Accuracy Default Configuration:  {objective_default:.3f}")



if __name__ == "__main__":
    import os
    from deephyper.problem import HpProblem
    from deephyper.search.hps import AMBS
    from deephyper.evaluator.evaluate import Evaluator

    problem = HpProblem()
    # Discrete hyperparameter (sampled with uniform prior)
    problem.add_hyperparameter((5, 20), "num_epochs")
    # Discrete and Real hyperparameters (sampled with log-uniform)
    problem.add_hyperparameter((8, 256, "log-uniform"), "batch_size")
    problem.add_hyperparameter((0.5, 5, "log-uniform"), "learning_rate")
    
    # Add a starting point to try first
    problem.add_starting_point(**default_config)

    evaluator = Evaluator.create(
        run, method="ray", method_kwargs={
            "address": "auto", # tells the Ray evaluator to connect to the already started cluster
            "num_cpus_per_task": 1, #
            "num_gpus_per_task": 1 # automatically compute the number of workers
        }
    )

    search = AMBS(problem, evaluator)

    search.search(max_evals=10)