#!/usr/bin/env python
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import Dataset
from src.models import ResNet 

def main():
    config_path = "cfg/default.yaml"
    config = Config(config_path)
    dataset = Dataset(config)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    

if __name__ == "__main__":
    main()