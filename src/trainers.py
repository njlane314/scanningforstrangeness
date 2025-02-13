import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from models import UResNet, ResNetEncoder

class BaseTrainer:
    def __init__(self, config, model, dataloader, optimizer, criterion):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = config.get("train.epochs")
        self.checkpoint_dir = config.get("train.checkpoint_dir")
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.__class__.__name__}_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def train(self):
        raise NotImplementedError("Subclasses must implement the train method.")

class SegmentationTrainer(BaseTrainer):
    def __init__(self, config, dataset):
        model = UResNet(
            in_dim=config.get("model.in_channels"),
            n_classes=config.get("model.n_classes")
        )
        optimizer = optim.Adam(model.parameters(), lr=config.get("train.learning_rate"))
        criterion = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=config.get("train.batch_size"), shuffle=True)
        super().__init__(config, model, dataloader, optimizer, criterion)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            start = time.time()
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.dataloader)
            elapsed = time.time() - start
            print(f"Segmentation Epoch {epoch}: Loss = {avg_loss:.4f}, Time = {elapsed:.2f}s")
            self.save_checkpoint(epoch, avg_loss)

class ContrastiveTrainer(BaseTrainer):
    def __init__(self, config, dataset):
        model = ResNetEncoder(
            in_dim=config.get("model.in_channels"),
            feature_dim=config.get("model.feature_dim"),
            n_filters=config.get("model.n_filters"),
            drop_prob=config.get("model.drop_prob")
        )
        optimizer = optim.Adam(model.parameters(), lr=config.get("train.learning_rate"))
        criterion = None
        dataloader = DataLoader(dataset, batch_size=config.get("train.batch_size"), shuffle=True)
        super().__init__(config, model, dataloader, optimizer, criterion)
        self.temperature = config.get("train.temperature", 0.1)
        self.induction_idx = config.get("dataset.induction_plane_index", 2)
    def nt_xent_loss(self, features):
        batch_size, n_planes, feat_dim = features.size()
        anchor = features[:, self.induction_idx, :]
        mask = torch.ones(n_planes, dtype=torch.bool, device=features.device)
        mask[self.induction_idx] = False
        positive = features[:, mask, :].mean(dim=1)
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        sim_matrix = torch.matmul(anchor, positive.t()) / self.temperature
        labels = torch.arange(batch_size).to(features.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            start = time.time()
            for event_images in self.dataloader:
                batch_size, n_planes, C, H, W = event_images.size()
                event_images = event_images.to(self.device)
                flat_images = event_images.view(batch_size * n_planes, C, H, W)
                features = self.model(flat_images)
                features = features.view(batch_size, n_planes, -1)
                loss = self.nt_xent_loss(features)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.dataloader)
            elapsed = time.time() - start
            print(f"Contrastive Epoch {epoch}: Loss = {avg_loss:.4f}, Time = {elapsed:.2f}s")
            self.save_checkpoint(epoch, avg_loss)