import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from model import UNet
import csv
import os

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_class_weights(stats):
    if np.any(stats == 0.):
        idx = np.where(stats == 0.)
        stats[idx] = 1
        weights = 1. / stats
        weights[idx] = 0
    else:
        weights = 1. / stats
    return [weight / sum(weights) for weight in weights]

def load_model(filename, num_classes, device):
    model = UNet(1, n_classes = num_classes, depth = 4, n_filters = 16, y_range = (0, num_classes - 1))
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

def save_model(model, input, filename):
    pkl_path = f"{filename}.pkl"
    torch.save(model.state_dict(), pkl_path)

    pt_path = f"{filename}.pt"
    torch.save(model.state_dict(), pt_path, _use_new_zipfile_serialization=False)

def accuracy(pred, truth, nearby=False):
    target = truth.squeeze(1)
    pred_cls = pred.argmax(dim=1)
    mask = (target != 0)
    if nearby:
        result = abs(pred_cls[mask] - target[mask]) <= 1
    else:
        result = pred_cls[mask] == target[mask]
    return result.float().mean()

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.argmax(dim=1) 
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum().float()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice

def intersection_over_union(pred, target, smooth=1e-6):
    pred = pred.argmax(dim=1)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum().float()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def create_model(num_classes, weights, device):
    model = UNet(1, n_classes=num_classes, depth=4, n_filters=16)
    if weights is not None:
        loss_fn = nn.CrossEntropyLoss(torch.as_tensor(weights, device=device, dtype=torch.float))
    else:
        loss_fn = nn.CrossEntropyLoss() 
    optim = torch.optim.Adam(model.parameters())
    
    return model, loss_fn, optim


def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, train_dice_scores, val_dice_scores, train_iou_scores, val_iou_scores, n_epochs, output_dir):
    epochs = np.arange(1, n_epochs + 1)

    train_losses = torch.tensor(train_losses).cpu().numpy()
    val_losses = torch.tensor(val_losses).cpu().numpy()
    train_accs = torch.tensor(train_accs).cpu().numpy()
    val_accs = torch.tensor(val_accs).cpu().numpy()
    train_dice_scores = torch.tensor(train_dice_scores).cpu().numpy()
    val_dice_scores = torch.tensor(val_dice_scores).cpu().numpy()
    train_iou_scores = torch.tensor(train_iou_scores).cpu().numpy()
    val_iou_scores = torch.tensor(val_iou_scores).cpu().numpy()

    csv_path = os.path.join(output_dir, 'loss_accuracy_data.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy', 'Train Dice', 'Validation Dice', 'Train IoU', 'Validation IoU'])
        for epoch, tl, vl, ta, va, td, vd, ti, vi in zip(epochs, train_losses, val_losses, train_accs, val_accs, train_dice_scores, val_dice_scores, train_iou_scores, val_iou_scores):
            writer.writerow([epoch, tl, vl, ta, va, td, vd, ti, vi])

    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='orange', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_dice_scores, label='Training Dice Coefficient', color='green', marker='o')
    plt.plot(epochs, val_dice_scores, label='Validation Dice Coefficient', color='red', marker='o')
    plt.plot(epochs, train_iou_scores, label='Training IoU', color='purple', marker='o')
    plt.plot(epochs, val_iou_scores, label='Validation IoU', color='brown', marker='o')
    plt.title('Dice Coefficient and IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_accuracy_plot.png")
    plt.close()
