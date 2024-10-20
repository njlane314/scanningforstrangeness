import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from model import UNet

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
    model = UNet(1, n_classes=num_classes, depth=4, n_filters=16, y_range=(0, num_classes - 1))
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

def save_model(model, input, filename):
    torch.save(model.state_dict(), f"{filename}.pkl")
    torch.save(model.state_dict(), filename + ".pt", _use_new_zipfile_serialization=False)

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
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accs, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='red', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_plot.png")
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
