import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from model import UNet
import csv
import os
import torch.nn.functional as F

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss * 1000

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def create_model(num_classes, weights, device):
    model = UNet(1, n_classes=num_classes, depth=4, n_filters=16)
    
    if weights is not None:
        loss_fn = FocalLoss(alpha=torch.as_tensor(weights, device=device, dtype=torch.float), gamma=2.0)
    else:
        loss_fn = FocalLoss(gamma=2.0)
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    return model, loss_fn, optim

def visualise_predictions(model, loader, device, output_dir, num_samples=3):
    model.eval()
    samples = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            preds = model(x).argmax(dim=1)

            x = x.cpu()
            y = y.cpu()
            preds = preds.cpu()

            for i in range(min(num_samples, x.size(0))):
                input_img = x[i].squeeze(0)
                ground_truth = y[i]
                prediction = preds[i]

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                ax[0].imshow(input_img, cmap='gray')
                ax[0].set_title('Input Image')
                ax[0].axis('off')

                ax[1].imshow(ground_truth, cmap='jet', vmin=0, vmax=loader.dataset.count_classes - 1)
                ax[1].set_title('Ground Truth')
                ax[1].axis('off')

                ax[2].imshow(prediction, cmap='jet', vmin=0, vmax=loader.dataset.count_classes - 1)
                ax[2].set_title('Prediction')
                ax[2].axis('off')

                plt.tight_layout()
                plt.savefig(f"{output_dir}/prediction_sample_{samples}.png")
                plt.close()

                samples += 1

            if samples >= num_samples:
                break



def plot_loss_accuracy(train_losses, val_losses,
                       train_accs, val_accs,
                       train_dice_scores, val_dice_scores,
                       train_iou_scores, val_iou_scores,
                       n_batches, output_dir):
    
    batches = np.arange(1, n_batches + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(batches, train_losses, label='Training loss', color='blue')
    plt.plot(batches, val_losses, label='Validation loss', color='red')
    plt.title("Training and Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_validation_loss.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(batches, train_accs, label='Training accuracy', color='blue')
    plt.plot(batches, val_accs, label='Validation accuracy', color='red')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_validation_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(batches, train_dice_scores, label='Training Dice', color='blue')
    plt.plot(batches, val_dice_scores, label='Validation Dice', color='red')
    plt.title("Training and Validation Dice Coefficient")
    plt.xlabel("Batch")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_validation_dice.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(batches, train_iou_scores, label='Training IoU', color='blue')
    plt.plot(batches, val_iou_scores, label='Validation IoU', color='red')
    plt.title("Training and Validation Intersection over Union (IoU)")
    plt.xlabel("Batch")
    plt.ylabel("IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_validation_iou.png")
    plt.close()

def plot_loss(output_dir, metrics):
    steps = np.arange(len(metrics['train_losses']))

    plt.figure(figsize=(8, 6))
    
    plt.plot(steps, metrics['train_losses'], color='blue', label='Training loss', alpha=0.8)

    plt.plot(steps, metrics['valid_losses'], color='red', label='Test loss', alpha=0.8)

    plt.fill_between(steps, np.array(metrics['train_losses']) - 0.05, np.array(metrics['train_losses']) + 0.05, color='blue', alpha=0.2)
    plt.fill_between(steps, np.array(metrics['valid_losses']) - 0.05, np.array(metrics['valid_losses']) + 0.05, color='red', alpha=0.2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Steps')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_validation_iou.png")
    plt.close()