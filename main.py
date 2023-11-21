from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.optim as optim

import time

from model import ToyModel
from utils import train_with_ddp_on_cpu, train_without_ddp

if __name__ == '__main__':
    
    for batch_size in [16, 64, 128]:
        model = ToyModel(batch_size=batch_size)
        train_without_ddp(model)
    
    for num_processes in [1, 2, 4]:
        for batch_size in [16, 32, 64]:
            model = ToyModel(batch_size=batch_size)
            train_with_ddp_on_cpu(model, num_processes)