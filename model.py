from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.optim as optim

import time

class ToyModel(pl.LightningModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.start_epoch = time.time()
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.batch_size=batch_size

    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(self.flatten(x))))
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.train_loss = loss
        self.train_accuracy = self.accuracy(logits, y)
        self.start_epoch = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.start_epoch
        self.log('start', self.start_epoch)
        self.log('end', epoch_end_time)
        self.log('epoch_duration', epoch_duration)
        self.log('train_loss', self.train_loss)
        self.log('train_accuracy', self.train_accuracy)
        self.log('val_loss', loss)
        self.log('val_accuracy', self.accuracy(logits, y))

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_dataset = datasets.FashionMNIST(
            root='./data', train=True, transform=transform, download=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.val_dataset = datasets.FashionMNIST(
            root='./data', train=False, transform=transform)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)