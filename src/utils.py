import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from models.model import UNet

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

def load_model_only(filename, num_classes, device):
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
