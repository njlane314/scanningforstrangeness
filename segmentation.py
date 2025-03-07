#!/usr/bin/env python
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import Dataset
from src.models import UResNetFull

def main():
    config_path = "cfg/default.yaml"
    config = Config(config_path)
    dataset = Dataset(config)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = config.get("model.in_channels", 1)
    n_classes = config.get("model.n_classes", 4)
    num_epochs = config.get("train.num_epochs", 1)
    learning_rate = config.get("train.learning_rate", 1e-3)
    model = UResNetFull(in_channels, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_epoch = time.time()
        for batch_idx, (images, one_hot_targets, run, subrun, event) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = one_hot_targets.argmax(dim=1).to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds, Average Loss: {avg_loss:.4f}")
        checkpoint_path = f"uresnet_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at: {checkpoint_path}")
    
    print("Training complete.")

if __name__ == "__main__":
    main()