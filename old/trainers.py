import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.models import UResNetFull, UResNetEncoder

class BaseTrainer:
    def __init__(self, config, model, dataloader, optimizer, criterion):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = config.get("train.num_epochs")
        self.ckpt_dir = config.get("train.ckpt_dir")
        if self.ckpt_dir and not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.ckpt_dir, f"{self.__class__.__name__}_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    def train(self):
        raise NotImplementedError("Subclasses must implement the train method.")
    
class WeightedRecallLoss(nn.Module):
    def __init__(self, class_weights, smooth=1e-6):
        super(WeightedRecallLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.smooth = smooth
    def forward(self, inputs, target):
        probs = F.softmax(inputs, dim=1)
        recall_sum = 0.0
        total_weight = self.class_weights.sum()
        for c in range(probs.shape[1]):
            p_c = probs[:, c, :, :]
            target_c = target[:, c, :, :]
            TP = (p_c * target_c).sum()
            FN = ((1 - p_c) * target_c).sum()
            recall_c = TP / (TP + FN + self.smooth)
            recall_sum += self.class_weights[c] * recall_c
        weighted_recall = recall_sum / total_weight
        return 1 - weighted_recall

class SegmentationTrainer(BaseTrainer):
    def __init__(self, config, full_dataset):
        total = len(full_dataset)
        train_ratio = config.get("dataset.train_split", 0.8)
        train_size = int(total * train_ratio)
        val_size = total - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=config.get("train.batch_size"), shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.get("val.batch_size", config.get("train.batch_size")), shuffle=False)
        model = UResNet(
            in_dim=config.get("model.in_channels"),
            n_classes=config.get("model.num_classes"),
            n_filters=config.get("model.filters"),
            drop_prob=config.get("model.dropout"),
            y_range=None
        )
        optimizer = optim.Adam(model.parameters(), lr=config.get("train.lr"))
        num_classes = config.get("model.num_classes")
        self.alpha = config.get("train.alpha", 0.1)
        self.epsilon = 1e-6
        self.global_counts = torch.zeros(num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        criterion = WeightedRecallLoss(class_weights=torch.ones(num_classes), smooth=self.epsilon)
        super().__init__(config, model, train_dataloader, optimizer, criterion)
        self.batch_loss_history = []
        self.epoch_loss_history = []

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            train_losses = []
            start = time.time()
            print(f"\n**** Epoch {epoch} Training")
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                print(f" Batch {batch_idx:03d} | Images: {images.shape} | Labels: {labels.shape}")
                self.optimizer.zero_grad()
                outputs = self.model(images)
                print(f" Batch {batch_idx:03d} | Outputs: {outputs.shape}")
                batch_counts = labels.sum(dim=(0, 2, 3))
                self.global_counts = self.alpha * batch_counts + (1 - self.alpha) * self.global_counts
                global_weights = 1.0 / (self.global_counts + self.epsilon)
                global_weights = global_weights / global_weights.sum()
                self.criterion.class_weights = global_weights
                print(f" Batch {batch_idx:03d} | Global Weights: {global_weights.cpu().numpy()}")
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                self.batch_loss_history.append((epoch, batch_idx, loss.item()))
                print(f" Batch {batch_idx:03d} | Loss: {loss.item():.4f}")
            avg_train_loss = np.mean(train_losses)
            train_loss_std = np.std(train_losses)
            train_loss_error = train_loss_std / np.sqrt(len(train_losses))
            self.model.eval()
            val_losses = []
            print(f"\n**** Epoch {epoch} Validation")
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(self.val_dataloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_losses.append(loss.item())
                    print(f" Val Batch {batch_idx:03d} | Loss: {loss.item():.4f}")
            avg_val_loss = np.mean(val_losses)
            val_loss_std = np.std(val_losses)
            val_loss_error = val_loss_std / np.sqrt(len(val_losses))
            elapsed = time.time() - start
            print(f"\n**** Epoch {epoch} Summary")
            print(f" Train Loss: {avg_train_loss:.4f} ± {train_loss_error:.4f}")
            print(f" Val Loss:   {avg_val_loss:.4f} ± {val_loss_error:.4f}")
            print(f" Elapsed:    {elapsed:.2f}s")
            self.epoch_loss_history.append((epoch, avg_train_loss, train_loss_error, avg_val_loss, val_loss_error))
            self.save_checkpoint(epoch, avg_train_loss)

class ContrastiveTrainer(BaseTrainer):
    def __init__(self, config, dataset):
        model = ResNetEncoder(
            in_dim=config.get("model.in_channels"),
            feat_dim=config.get("model.feat_dim"),
            filters=config.get("model.filters"),
            dropout=config.get("model.dropout")
        )
        optimizer = optim.Adam(model.parameters(), lr=config.get("train.lr"))
        criterion = None
        dataloader = DataLoader(dataset, batch_size=config.get("train.batch_size"), shuffle=True)
        super().__init__(config, model, dataloader, optimizer, criterion)
        self.temp = config.get("train.temp", 0.1)
        self.induction_idx = config.get("dataset.ind_plane_idx", 2)
    def nt_xent_loss(self, features):
        batch_size, n_planes, feat_dim = features.size()
        anchor = features[:, self.induction_idx, :]
        mask = torch.ones(n_planes, dtype=torch.bool, device=features.device)
        mask[self.induction_idx] = False
        positive = features[:, mask, :].mean(dim=1)
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        sim_matrix = torch.matmul(anchor, positive.t()) / self.temp
        labels = torch.arange(batch_size).to(features.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    def train(self):
        for epoch in range(1, self.num_epochs + 1):
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