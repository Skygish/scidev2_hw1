from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.optim as optim

import time

def train_without_ddp(model):
    logger = pl.loggers.CSVLogger("logs", name="ToyModel")

    trainer = pl.Trainer(max_epochs=10, logger=logger)
    trainer.log_every_n_steps = 1

    trainer.fit(model)


def train_with_ddp_on_cpu(model, num_processes):
    logger = pl.loggers.CSVLogger("logs", name="ToyModel")
    trainer = pl.Trainer(
        max_epochs=10,
        strategy='ddp',
        accelerator='cpu',
        devices=num_processes,
        logger=logger
    )