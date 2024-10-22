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
    model = create_model(num_classes)  
    model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))  
    model = model.to(device)
    return model


def save_model(model, input, filename):
    pkl_path = f"{filename}.pkl"
    torch.save(model.state_dict(), pkl_path)

    pt_path = f"{filename}.pt"
    torch.save(model.state_dict(), pt_path, _use_new_zipfile_serialization=False)

    print(f"Model saved as: {pkl_path} and {pt_path}")

def accuracy(pred, truth, nearby=False):
    target = truth.squeeze(1)
    pred_cls = pred.argmax(dim=1)
    mask = (target != 0)
    if nearby:
        result = abs(pred_cls[mask] - target[mask]) <= 1
    else:
        result = pred_cls[mask] == target[mask]
    return result.float().mean()

def create_model(num_classes, weights, device):
    model = UNet(1, n_classes=num_classes, depth=4, n_filters=16, y_range=(0, num_classes - 1))
    loss_fn = nn.CrossEntropyLoss(torch.as_tensor(weights, device=device, dtype=torch.float))
    optim = opt.Adam(model.parameters())
    return model, loss_fn, optim

def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, n_epochs, output_dir):
    epochs = np.arange(1, n_epochs + 1)

    train_losses = torch.tensor(train_losses).cpu().numpy()
    val_losses = torch.tensor(val_losses).cpu().numpy()
    train_accs = torch.tensor(train_accs).cpu().numpy()
    val_accs = torch.tensor(val_accs).cpu().numpy()

    csv_path = os.path.join(output_dir, 'loss_accuracy_data.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'])
        for epoch, tl, vl, ta, va in zip(epochs, train_losses, val_losses, train_accs, val_accs):
            writer.writerow([epoch, tl, vl, ta, va])

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='orange', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_accuracy_plot.png")
    plt.close()

def plot_class_performance(iou_scores, dice_scores, class_names, n_epochs, output_dir):
    epochs = np.arange(1, n_epochs + 1)

    for i, class_name in enumerate(class_names):
        plt.plot(epochs, np.array(iou_scores)[:, i], label=f'{class_name}')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU per Class')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f"{output_dir}/iou_plot.png")
    plt.close()

    for i, class_name in enumerate(class_names):
        plt.plot(epochs, np.array(dice_scores)[:, i], label=f'{class_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient per Class')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f"{output_dir}/dice_plot.png")
    plt.close()
