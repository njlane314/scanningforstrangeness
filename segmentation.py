import argparse
import os
import sys
import uproot
import resource
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

cudnn.benchmark = True

def print_memory_usage(msg=""):
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"{msg} Memory usage: {usage/1024:.2f} MB")

def get_parser():
    parser = argparse.ArgumentParser(description="Segmentation Training Script")
    parser.add_argument("--num-epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root",
                        help="Path to the single input ROOT file")
    parser.add_argument("--img-size", default=512, type=int, help="Image size (square) in pixels")
    parser.add_argument("--learning-rate", default=0.0001, type=float, help="Learning rate for the optimizer")
    parser.add_argument("--target-labels", type=str, default="0,1,2,5",
                        help="Comma-separated list of enum values to train with")
    parser.add_argument("--num-planes", default=3, type=int, help="Number of image planes")
    parser.add_argument("--checkpoint-directory", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation",
                        help="Directory to save model checkpoints")
    parser.add_argument("--loss-file", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/losses.npz",
                        help="Output file to save loss and metric results")
    parser.add_argument("--gamma", default=2.0, type=float, help="Gamma parameter for Focal Loss")
    return parser

def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def dropout(prob):
    return nn.Dropout(prob)

def reinit_layer(layer, leak=0.0, use_kaiming_normal=True):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        if use_kaiming_normal:
            nn.init.kaiming_normal_(layer.weight, a=leak)
        else:
            nn.init.kaiming_uniform_(layer.weight, a=leak)
            layer.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, k_pad=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.identity = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=1)
        reinit_layer(self.conv1)
        reinit_layer(self.conv2)
    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.relu(x + identity)

class TransposeConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, k_pad=1):
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=k_size, padding=k_pad, output_padding=1, stride=2),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True)
        )
        reinit_layer(self.block[0])
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_dim, num_classes, depth=4, n_filters=16, drop_prob=0.1, num_planes=3):
        super(UNet, self).__init__()
        self.num_planes = num_planes
        self.num_classes = num_classes
        self.ds_conv_1 = ConvBlock(in_dim, n_filters)
        self.ds_conv_2 = ConvBlock(n_filters, 2 * n_filters)
        self.ds_conv_3 = ConvBlock(2 * n_filters, 4 * n_filters)
        self.ds_conv_4 = ConvBlock(4 * n_filters, 8 * n_filters)
        self.ds_maxpool_1 = maxpool()
        self.ds_maxpool_2 = maxpool()
        self.ds_maxpool_3 = maxpool()
        self.ds_maxpool_4 = maxpool()
        self.ds_dropout_1 = dropout(drop_prob)
        self.ds_dropout_2 = dropout(drop_prob)
        self.ds_dropout_3 = dropout(drop_prob)
        self.ds_dropout_4 = dropout(drop_prob)
        self.bridge = ConvBlock(8 * n_filters, 16 * n_filters)
        self.us_tconv_4 = TransposeConvBlock(16 * n_filters, 8 * n_filters)
        self.us_tconv_3 = TransposeConvBlock(8 * n_filters, 4 * n_filters)
        self.us_tconv_2 = TransposeConvBlock(4 * n_filters, 2 * n_filters)
        self.us_tconv_1 = TransposeConvBlock(2 * n_filters, n_filters)
        self.us_conv_4 = ConvBlock(16 * n_filters, 8 * n_filters)
        self.us_conv_3 = ConvBlock(8 * n_filters, 4 * n_filters)
        self.us_conv_2 = ConvBlock(4 * n_filters, 2 * n_filters)
        self.us_conv_1 = ConvBlock(2 * n_filters, 1 * n_filters)
        self.us_dropout_4 = dropout(drop_prob)
        self.us_dropout_3 = dropout(drop_prob)
        self.us_dropout_2 = dropout(drop_prob)
        self.us_dropout_1 = dropout(drop_prob)
        self.output = nn.Conv2d(n_filters, num_planes * num_classes, 1)
    def forward(self, x):
        res = x
        res = self.ds_conv_1(res); conv_stack_1 = res.clone()
        res = self.ds_maxpool_1(res)
        res = self.ds_dropout_1(res)
        res = self.ds_conv_2(res); conv_stack_2 = res.clone()
        res = self.ds_maxpool_2(res)
        res = self.ds_dropout_2(res)
        res = self.ds_conv_3(res); conv_stack_3 = res.clone()
        res = self.ds_maxpool_3(res)
        res = self.ds_dropout_3(res)
        res = self.ds_conv_4(res); conv_stack_4 = res.clone()
        res = self.ds_maxpool_4(res)
        res = self.ds_dropout_4(res)
        res = self.bridge(res)
        res = self.us_tconv_4(res)
        res = torch.cat([res, conv_stack_4], dim=1)
        res = self.us_dropout_4(res)
        res = self.us_conv_4(res)
        res = self.us_tconv_3(res)
        res = torch.cat([res, conv_stack_3], dim=1)
        res = self.us_dropout_3(res)
        res = self.us_conv_3(res)
        res = self.us_tconv_2(res)
        res = torch.cat([res, conv_stack_2], dim=1)
        res = self.us_dropout_2(res)
        res = self.us_conv_2(res)
        res = self.us_tconv_1(res)
        res = torch.cat([res, conv_stack_1], dim=1)
        res = self.us_dropout_1(res)
        res = self.us_conv_1(res)
        res = self.output(res)
        return res.view(x.size(0), self.num_planes, self.num_classes, x.size(2), x.size(3))

class ImageDataset(Dataset):
    def __init__(self, args, file):
        self.args = args
        self.file_path = file
        self.tree_name = "imageanalyser/ImageTree"
        self.img_size = args.img_size
        self.num_classes = args.num_classes
        self.num_planes = args.num_planes
        self.enum_to_model = {val: idx for idx, val in enumerate(args.target_labels)}
        try:
            self.root_file = uproot.open(self.file_path, array_cache=None, num_workers=0)
        except Exception as e:
            raise RuntimeError(f"Failed to open ROOT file '{self.file_path}': {e}")
        self.tree = self.root_file[self.tree_name]
        event_types = self.tree["type"].array(library="np")
        self.indices = np.where(event_types == 0)[0]
        self.num_events = len(self.indices)
    def __len__(self):
        return self.num_events
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        try:
            data = self.tree.arrays(
                ["input", "truth", "run", "subrun", "event"],
                entry_start=actual_idx, entry_stop=actual_idx + 1,
                library="np"
            )
        except Exception as e:
            print(f"Error fetching event {actual_idx}: {e}", file=sys.stderr)
            raise
        run = data["run"][0]
        subrun = data["subrun"][0]
        event = data["event"][0]
        images = []
        for plane in range(self.num_planes):
            plane_iterable = data["input"][0][plane]
            plane_array = np.fromiter(plane_iterable, dtype=np.float32, count=self.img_size * self.img_size)
            images.append(plane_array.reshape(self.img_size, self.img_size))
        images_tensor = torch.tensor(np.stack(images), dtype=torch.float32)
        labels = []
        for plane_idx in range(self.num_planes):
            truth_iterable = data["truth"][0][plane_idx]
            truth_array = np.fromiter(truth_iterable, dtype=np.int64, count=self.img_size * self.img_size)
            labels.append(truth_array.reshape(self.img_size, self.img_size))
        label_tensor = torch.tensor(np.stack(labels), dtype=torch.long)
        remapped_labels = label_tensor.clone()
        for enum_val, model_val in self.enum_to_model.items():
            remapped_labels[label_tensor == enum_val] = model_val
        mask = ~torch.isin(label_tensor, torch.tensor(list(self.enum_to_model.keys()), dtype=torch.long))
        if mask.any():
            remapped_labels[mask] = 1
        return images_tensor, remapped_labels, run, subrun, event

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        self.weight = weight
        self.reduction = reduction
    def forward(self, inp: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class FocalLossFlat(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean', axis: int = 2):
        super(FocalLossFlat, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=True)
        self.weight = weight
        self.reduction = reduction
        self.axis = axis

    def forward(self, inp: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        if inp.dim() > 2:
            dims = list(range(inp.dim()))  # inp: [batch_size, num_planes, num_classes, H, W]
            dims.remove(self.axis)  # Remove 2: [0, 1, 2, 3, 4] -> [0, 1, 3, 4]
            dims.append(self.axis)  # Append 2: [0, 1, 3, 4, 2]
            inp = inp.permute(*dims)  # [8, 3, 512, 512, 5]
            inp = inp.contiguous().view(-1, inp.size(-1))  # [8*3*512*512, 5]
            targ = targ.view(-1)  # [8*3*512*512]
        return FocalLoss(gamma=self.gamma, weight=self.weight, reduction=self.reduction)(inp, targ)
    def decodes(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=self.axis)
    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.axis)

def computer_jaccard(pred, target, target_labels):
    jaccards = []
    for cls in target_labels:  
        intersection = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union == 0:
            jaccards.append(np.nan)
        else:
            jaccards.append(intersection / union)
    return np.nanmean(jaccards)

def compute_dice(pred, target, target_labels):
    dices = []
    for cls in target_labels:
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        denom = pred_cls.sum().item() + target_cls.sum().item()
        if denom == 0:
            dices.append(np.nan)
        else:
            dices.append((2 * intersection) / (denom + 1e-6))
    return np.nanmean(dices)

def compute_recall(pred, target, target_labels):
    recalls = []
    for cls in target_labels:
        tp = ((pred == cls) & (target == cls)).sum().item()
        fn = ((target == cls) & (pred != cls)).sum().item()
        if (tp + fn) == 0:
            recalls.append(np.nan)
        else:
            recalls.append(tp / (tp + fn))
    return np.nanmean(recalls)

def train_epoch(model, epoch, dataloader, optimiser, criterion, scheduler, device, args):
    model.train()
    train_loss, valid_loss, learning_rate = [], [], []
    gamma_history, jaccard_history, dice_history = [], [], []
    accuracy_history, recall_history = [], []
    model_to_enum = {idx: val for idx, val in enumerate(args.target_labels)}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for batch, (images, masks, _, _, _) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        half = images.size(0) // 2
        train_images, valid_images = images[:half], images[half:]
        train_masks, valid_masks = masks[:half], masks[half:]
        optimiser.zero_grad()
        train_outputs = model(train_images)
        loss_val = criterion(train_outputs, train_masks)
        loss_val.backward()
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_images)
            val_loss_val = criterion(valid_outputs, valid_masks)
            preds = criterion.decodes(valid_outputs)  
            preds_np = preds.cpu().numpy()
            preds_enum = np.vectorize(model_to_enum.get)(preds_np)
            masks_np = valid_masks.cpu().numpy()
            num_planes = preds_np.shape[1]
            jaccard_per_plane = []
            dice_per_plane = []
            acc_per_plane = []
            recall_per_plane = []
            for plane in range(num_planes):
                jaccard_plane = []
                dice_plane = []
                acc_plane = []
                recall_plane = []
                for sample in range(preds_np.shape[0]):
                    jaccard_plane.append(computer_jaccard(preds_enum[sample, plane],
                                                         masks_np[sample, plane],
                                                         args.target_labels))
                    dice_plane.append(compute_dice(preds_enum[sample, plane],
                                                  masks_np[sample, plane],
                                                  args.target_labels))
                    acc_plane.append(np.mean(preds_enum[sample, plane] == masks_np[sample, plane]))
                    recall_plane.append(compute_recall(preds_enum[sample, plane],
                                                      masks_np[sample, plane],
                                                      args.target_labels))
                jaccard_per_plane.append(np.nanmean(jaccard_plane))
                dice_per_plane.append(np.nanmean(dice_plane))
                acc_per_plane.append(np.nanmean(acc_plane))
                recall_per_plane.append(np.nanmean(recall_plane))
        model.train()
        optimiser.step()
        scheduler.step()
        train_loss.append(loss_val.item())
        valid_loss.append(val_loss_val.item())
        learning_rate.append(optimiser.param_groups[0]['lr'])
        gamma_history.append(criterion.gamma.item())
        jaccard_history.append(jaccard_per_plane)
        dice_history.append(dice_per_plane)
        accuracy_history.append(acc_per_plane)
        recall_history.append(recall_per_plane)
        
        if batch % 10 == 0:
            print(f"Batch {batch:03d} | Train Loss: {loss_val.item():.4f} | Val Loss: {val_loss_val.item():.4f}")
            print(f"   Per-plane metrics: Jaccard: {jaccard_per_plane}, Dice: {dice_per_plane}")
            print(f"                      Accuracy: {acc_per_plane}, Recall: {recall_per_plane}")
            print_memory_usage(f'[Batch {batch}] Memory usage:')

        if batch % 1000 == 0:
            model.eval()
            dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
            traced_model = torch.jit.trace(model, dummy_input)
            ts_model_path = os.path.join(args.checkpoint_directory, f"unet_epoch{epoch}_batch{batch}_{timestamp}_torchscript.pt")
            ts_labels_path = os.path.join(args.checkpoint_directory, f"target_labels_epoch{epoch}_batch{batch}_{timestamp}.pth")
            traced_model.save(ts_model_path)
            torch.save(args.target_labels, ts_labels_path)
            print(f"Saved checkpoint: {ts_model_path}")
            print(f"Saved target labels: {ts_labels_path}")
            model.train()
        torch.cuda.empty_cache()

    return (train_loss, valid_loss, learning_rate, gamma_history, 
            jaccard_history, dice_history, accuracy_history, recall_history)

def get_dataloader(args):
    file = args.root_file
    dataset = ImageDataset(args, file)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True)

def main():
    parser = get_parser()
    args = parser.parse_args()
    target_labels = [int(x) for x in args.target_labels.split(',')]
    args.num_classes = len(target_labels)
    args.target_labels = target_labels
    os.makedirs(args.checkpoint_directory, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    dataloader = get_dataloader(args)
    model = UNet(
        in_dim=3,
        num_classes=args.num_classes,
        depth=4,
        n_filters=16,
        drop_prob=0.1,
        num_planes=args.num_planes
    ).to(device)
    criterion = FocalLossFlat(gamma=args.gamma, reduction='mean', axis=2)
    optimiser = Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.num_epochs, eta_min=1e-6)
    
    training = {
        'train_loss': [],
        'valid_loss': [],
        'learning_rate': [],
        'gamma': [],
        'jaccard': [],
        'dice': [],
        'accuracy': [],
        'recall': []
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(args.num_epochs):
        (t_loss, v_loss, lrs, gma, jaccards, dices, accs, recs) = train_epoch(model, epoch, dataloader, optimiser, criterion, scheduler, device, args)
        training['train_loss'].extend(t_loss)
        training['valid_loss'].extend(v_loss)
        training['learning_rate'].extend(lrs)
        training['gamma'].extend(gma)
        training['jaccard'].extend(jaccards)
        training['dice'].extend(dices)
        training['accuracy'].extend(accs)
        training['recall'].extend(recs)
        torch.cuda.empty_cache()
        
    loss_file_with_timestamp = args.loss_file.replace(".npz", f"_{timestamp}.npz")
    np.savez(
        loss_file_with_timestamp,
        train_loss=np.array(training['train_loss']),
        valid_loss=np.array(training['valid_loss']),
        learning_rate=np.array(training['learning_rate']),
        gamma=np.array(training['gamma']),
        jaccard=np.array(training['jaccard']),
        dice=np.array(training['dice']),
        accuracy=np.array(training['accuracy']),
        recall=np.array(training['recall'])
    )

if __name__ == '__main__':
    main()