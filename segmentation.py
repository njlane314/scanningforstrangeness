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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
from datetime import datetime

cudnn.benchmark = True

def print_memory_usage(msg=""):
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"{msg} Memory usage: {usage/1024:.2f} MB")

def get_parser():
    parser = argparse.ArgumentParser(description="UResNet")
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--target-labels", type=str, default="0,1,2,4")
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output-dir", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation")
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
    def __init__(self, in_dim, n_classes, depth=4, n_filters=16, drop_prob=0.1, y_range=None):
        super(UNet, self).__init__()
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
        self.output = nn.Conv2d(n_filters, n_classes, 1)
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
        return res
    
class ImageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_file = uproot.open(args.root_file, array_cache=None, num_workers=0)
        self.tree = self.root_file["imageanalyser/ImageTree"]
        self.plane = args.plane
        self.img_size = args.img_size
        self.target_labels = [int(x) for x in args.target_labels.split(',')]
        self.foreground_labels = [lbl for lbl in self.target_labels if lbl >= 2]
        self.num_classes = len(self.foreground_labels)
        self.enum_to_model = {val: idx for idx, val in enumerate(self.foreground_labels)}
        event_types = self.tree["type"].array(library="np")
        self.indices = np.where(event_types == 0)[0]
        self.num_events = len(self.indices)
    def __len__(self):
        return self.num_events
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data = self.tree.arrays(
            ["input", "truth", "run", "subrun", "event"],
            entry_start=actual_idx, entry_stop=actual_idx + 1,
            library="np"
        )
        run, subrun, event = data["run"][0], data["subrun"][0], data["event"][0]
        img = np.fromiter(data["input"][0][self.plane], dtype=np.float32).reshape(1, self.img_size, self.img_size)  
        img = np.log1p(img)
        img = torch.from_numpy(img)
        truth = np.fromiter(data["truth"][0][self.plane], dtype=np.int64).reshape(self.img_size, self.img_size)
        masks = torch.zeros(self.num_classes, self.img_size, self.img_size, dtype=torch.float32)
        for i, label in enumerate(self.foreground_labels):
            masks[i] = torch.tensor(truth == label, dtype=torch.float32)
        return img, masks, run, subrun, event

def compute_metrics(preds, targets, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    precision_dict = {t: [] for t in thresholds}
    recall_dict = {t: [] for t in thresholds}
    for t in thresholds:
        preds_t = torch.sigmoid(preds) > t
        targets = targets.bool()
        tp = (preds_t & targets).float().sum(dim=(2, 3))
        fp = (preds_t & ~targets).float().sum(dim=(2, 3))
        fn = (~preds_t & targets).float().sum(dim=(2, 3))
        tp = tp.sum(dim=0)
        fp = fp.sum(dim=0)
        fn = fn.sum(dim=0)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        precision_dict[t] = precision
        recall_dict[t] = recall
    return precision_dict, recall_dict

def train_model(model, dataloader, optimiser, device, args):
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=len(dataloader), T_mult=2, eta_min=1e-6)
    train_losses = []
    valid_losses = []
    learning_rates = []
    precisions = {0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: []}
    recalls = {0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: []}
    cumulative_class_counts = torch.zeros(args.num_classes, device=device, dtype=torch.float32)
    cumulative_total_pixels = 0.0
    for epoch in range(args.num_epochs):
        for batch, (images, masks, _, _, _) in enumerate(dataloader):
            split_idx = int(0.5 * images.size(0))
            train_images = images[:split_idx]
            train_masks = masks[:split_idx]
            valid_images = images[split_idx:]
            valid_masks = masks[split_idx:]
            model.train()
            train_images, train_masks = train_images.to(device), train_masks.to(device)
            valid_images, valid_masks = valid_images.to(device), valid_masks.to(device)
            batch_size, C, H, W = train_masks.shape

            batch_class_counts = train_masks.sum(dim=[0, 2, 3])  
            cumulative_class_counts += batch_class_counts
            cumulative_total_pixels += batch_size * H * W

            pos_weight = (cumulative_total_pixels - cumulative_class_counts) / (cumulative_class_counts + 1e-6)
            pos_weight = torch.clamp(pos_weight, min=0.001, max=1000)

            optimiser.zero_grad()
            train_outputs = model(train_images)
            weight_mask = pos_weight.view(1, -1, 1, 1) * train_masks
            train_loss = F.binary_cross_entropy_with_logits(train_outputs, train_masks, pos_weight=weight_mask)
            train_losses.append(train_loss.item())
            model.eval()
            with torch.no_grad():
                valid_outputs = model(valid_images)
                valid_weight_mask = pos_weight.view(1, -1, 1, 1) * valid_masks
                valid_loss = F.binary_cross_entropy_with_logits(valid_outputs, valid_masks, pos_weight=valid_weight_mask).item()
                valid_losses.append(valid_loss)
            model.train()
            train_loss.backward()
            learning_rates.append(scheduler.get_last_lr()[0])
            optimiser.step()
            scheduler.step()
            precision_dict, recall_dict = compute_metrics(valid_outputs, valid_masks)
            for t in precision_dict:
                precisions[t].append(precision_dict[t].cpu().numpy())
                recalls[t].append(recall_dict[t].cpu().numpy())
            if batch % 1 == 0:
                print(f"Epoch {epoch}, Batch {batch}: Train Loss = {train_loss.item():.4f}, Valid Loss = {valid_loss:.4f}, "
                      f"Learning Rate = {learning_rates[-1]:.6f}, "
                      f"Precision (0.3) = {precisions[0.3][-1]}, Recall (0.3) = {recalls[0.3][-1]}, "
                      f"Precision (0.4) = {precisions[0.4][-1]}, Recall (0.4) = {recalls[0.4][-1]}, "
                      f"Precision (0.5) = {precisions[0.5][-1]}, Recall (0.5) = {recalls[0.5][-1]}, "
                      f"Precision (0.6) = {precisions[0.6][-1]}, Recall (0.6) = {recalls[0.6][-1]}, "
                      f"Precision (0.7) = {precisions[0.7][-1]}, Recall (0.7) = {recalls[0.7][-1]}, "
                      f"Precision (0.8) = {precisions[0.8][-1]}, Recall (0.8) = {recalls[0.8][-1]}, "
                      f"Precision (0.9) = {precisions[0.9][-1]}, Recall (0.9) = {recalls[0.9][-1]}")
            if batch % 100 == 0:
                model.eval()
                example_input = torch.randn(1, 1, args.img_size, args.img_size).to(device)
                traced_model = torch.jit.trace(model, example_input)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                traced_model.save(os.path.join(args.output_dir, f"uresnet_plane{args.plane}_{epoch}_{batch}_{timestamp}_ts.pt"))
        model.eval()
        example_input = torch.randn(1, 1, args.img_size, args.img_size).to(device)
        traced_model = torch.jit.trace(model, example_input)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traced_model.save(os.path.join(args.output_dir, f"uresnet_plane{args.plane}_{epoch}_{timestamp}_ts.pt"))
    return train_losses, valid_losses, learning_rates, precisions, recalls

def main():
    args = get_parser().parse_args()
    target_labels = [int(x) for x in args.target_labels.split(',')]
    args.num_classes = len([lbl for lbl in target_labels if lbl >= 2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    model = UNet(in_dim=1, n_classes=args.num_classes, depth=4, n_filters=16, drop_prob=0.1).to(device)
    optimiser = Adam(model.parameters(), lr=args.learning_rate)
    os.makedirs(args.output_dir, exist_ok=True)
    train_losses, valid_losses, learning_rates, precisions, recalls = train_model(
        model, dataloader, optimiser, device, args
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(os.path.join(args.output_dir, f"loss_{timestamp}.npz"),
             step_train_losses=train_losses,
             step_valid_losses=valid_losses,
             step_learning_rates=learning_rates,
             step_precisions_03=precisions[0.3],
             step_precisions_04=precisions[0.4],
             step_precisions_05=precisions[0.5],
             step_precisions_06=precisions[0.6],
             step_precisions_07=precisions[0.7],
             step_recalls_03=recalls[0.3],
             step_recalls_04=recalls[0.4],
             step_recalls_05=recalls[0.5],
             step_recalls_06=recalls[0.6],
             step_recalls_07=recalls[0.7])
    label_mapping = {str(k): v for k, v in dataset.enum_to_model.items()}
    np.savez(os.path.join(args.output_dir, f"label_mapping_plane{args.plane}.npz"), **label_mapping)

if __name__ == '__main__':
    main()