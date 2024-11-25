import torch.nn as nn
import time
from line_profiler import profile
from torch.cuda.amp import autocast, GradScaler

import os
import torch
import numpy as np
import uproot
import argparse
from datetime import datetime
from tqdm import tqdm 
import random

from lib.dataset import ImageDataLoader
from lib.model import UNet
from lib.loss import FocalLoss
from lib.common import set_seed
from lib.config import ConfigLoader

from sklearn.metrics import precision_score, recall_score

def create_model(n_classes, weights, device):
    model = UNet(1, n_classes=n_classes, depth=4, n_filters=16)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    #loss_fn = FocalLoss(alpha=torch.tensor(weights, dtype=torch.float32, device=device), gamma=3, reduction='mean')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, loss_fn, optim

def get_class_weights(stats):
    if np.any(stats == 0.):
        idx = np.where(stats == 0.)
        stats[idx] = 1
        weights = 1. / stats
        weights[idx] = 0
    else:
        weights = 1. / stats
    return [weight / sum(weights) for weight in weights]

def save_class_weights(weights, path="class_weights.npy"):
    np.save(path, weights)
    print("\033[34m-- Class weights saved\033[0m")

def load_class_weights(path="class_weights.npy"):
    if os.path.exists(path):
        print("\033[34m-- Loading saved class weights\033[0m")
        return np.load(path)
    else:
        print("\033[35m-- Class weights file not found; calculating class weights\033[0m")
        return None

def calculate_metrics(pred, target, n_classes):
    pred = pred.argmax(dim=1).flatten().cpu().numpy()  
    target = target.flatten().cpu().numpy()  

    accuracy = (pred == target).mean()
    precision = precision_score(target, pred, labels=[1, 2], average='macro', zero_division=0)
    recall = recall_score(target, pred, labels=[1, 2], average='macro', zero_division=0)

    return accuracy, precision, recall

def train(config):
    print("\033[34m-- Initialising training configuration\033[0m")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)
    print(f"\033[34m-- Using device: {device}\033[0m")

    batch_size = config.batch_size
    train_pct = config.train_pct
    valid_pct = config.valid_pct
    n_classes = config.n_classes
    n_epochs = config.n_epochs

    input_dir = config.input_dir
    target_dir = config.target_dir
    output_dir = config.output_dir
    model_name = config.model_name
    model_save_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    print("\033[34m-- Loading data\033[0m")
    data_loader = ImageDataLoader(
        input_dir=input_dir,
        target_dir=target_dir,
        batch_size=batch_size,
        train_pct=train_pct,
        valid_pct=valid_pct,
        device=device
    )

    print("\033[34m-- Loading or calculating class weights\033[0m")
    class_weights = load_class_weights()  
    if class_weights is None:  
        train_stats = data_loader.count_classes(n_classes)
        class_weights = get_class_weights(train_stats)
        save_class_weights(class_weights)

    print("\033[34m-- Initialising model, loss function, and optimiser\033[0m")
    model, loss_fn, optim = create_model(n_classes, class_weights, device)
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)

    train_losses = []
    valid_losses = []

    output_path = os.path.join(output_dir, f"{model_name}_metrics.root")
    with uproot.recreate(output_path) as output:
        print("\033[34m-- Starting training loop\033[0m")

        train_loss = []
        valid_loss = []
        for epoch in range(n_epochs):
            model.train()
            epoch_train_loss = []
            epoch_valid_loss = []
            
            for i, batch in enumerate(tqdm(data_loader.train_dl, desc=f"Training Epoch {epoch+1}")):
                x, y = batch
                x, y = x.to(device), y.to(device)

                y = y.to(torch.long)
                optim.zero_grad()

                pred = model(x)
                loss = loss_fn(pred, y)
                        
                loss.backward()
                optim.step()

                train_loss.append(loss.item())
                tqdm.write(f"\033[34m--- Epoch {epoch+1}, Batch {i+1} - Training Loss: {loss.item()}\033[0m")

                model.eval()
                with torch.no_grad():
                    valid_iterator = iter(data_loader.valid_signature_dl)
                    try:
                        random_val_batch = next(valid_iterator) 
                    except StopIteration:
                        valid_iterator = iter(data_loader.valid_signature_dl)
                        random_val_batch = next(valid_iterator)

                    val_x, val_y = random_val_batch
                    val_x, val_y = val_x.to(device), val_y.to(device)

                    val_pred = model(val_x)
                    val_loss = loss_fn(val_pred, val_y)

                    valid_loss.append(val_loss.item())
                    tqdm.write(f"\033[34m--- Epoch {epoch+1}, Batch {i+1} - Validation Loss (Random Batch): {val_loss.item()}\033[0m")
            
                model.train()

            model_save_path = os.path.join(model_save_dir, f"{model_name}_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"\033[32m-- Model saved at {model_save_path}\033[0m")

        output["training"] = {
            "train_loss": train_losses,
            "valid_loss": valid_losses,
        }

    print("\033[32m-- Training complete\033[0m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    
    config = ConfigLoader(args.config) 
    train(config)
