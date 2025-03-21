import torch
import torch.nn as nn
import MinkowskiEngine as ME
import uproot
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
from datetime import datetime
import argparse

class SparseImageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_file = uproot.open(args.root_file)
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
        data = self.tree.arrays(["input", "truth", "run", "subrun", "event"], entry_start=actual_idx, entry_stop=actual_idx + 1, library="np")
        img = torch.from_numpy(np.fromiter(data["input"][0][self.plane], dtype=np.float32).reshape(self.img_size, self.img_size))
        coords = torch.nonzero(img > 0)
        coords = torch.cat([torch.zeros(coords.size(0), 1, dtype=torch.int), coords], dim=1)  
        features = img[coords[:, 1], coords[:, 2]].unsqueeze(1)  
        input_tensor = ME.SparseTensor(features=features, coordinates=coords)

        truth = np.fromiter(data["truth"][0][self.plane], dtype=np.int64).reshape(self.img_size, self.img_size)
        masks = torch.stack([torch.tensor(truth == lbl, dtype=torch.float32) for lbl in self.foreground_labels])
        sparse_masks = masks[:, coords[:, 1], coords[:, 2]] 
        return input_tensor, sparse_masks, data["run"][0], data["subrun"][0], data["event"][0]

class SparseUNet(nn.Module):
    def __init__(self, in_dim, n_classes, depth=4, n_filters=16, drop_prob=0.1):
        super(SparseUNet, self).__init__()
        self.ds_conv_1 = ME.MinkowskiConvolution(in_dim, n_filters, kernel_size=3, dimension=2)
        self.ds_conv_2 = ME.MinkowskiConvolution(n_filters, 2 * n_filters, kernel_size=3, dimension=2)
        self.ds_conv_3 = ME.MinkowskiConvolution(2 * n_filters, 4 * n_filters, kernel_size=3, dimension=2)
        self.ds_conv_4 = ME.MinkowskiConvolution(4 * n_filters, 8 * n_filters, kernel_size=3, dimension=2)
        self.ds_maxpool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=2)
        self.ds_dropout = ME.MinkowskiDropout(drop_prob)
        self.bridge = ME.MinkowskiConvolution(8 * n_filters, 16 * n_filters, kernel_size=3, dimension=2)
        self.us_tconv_4 = ME.MinkowskiConvolutionTranspose(16 * n_filters, 8 * n_filters, kernel_size=2, stride=2, dimension=2)
        self.us_conv_4 = ME.MinkowskiConvolution(16 * n_filters, 8 * n_filters, kernel_size=3, dimension=2)
        self.us_tconv_3 = ME.MinkowskiConvolutionTranspose(8 * n_filters, 4 * n_filters, kernel_size=2, stride=2, dimension=2)
        self.us_conv_3 = ME.MinkowskiConvolution(8 * n_filters, 4 * n_filters, kernel_size=3, dimension=2)
        self.us_tconv_2 = ME.MinkowskiConvolutionTranspose(4 * n_filters, 2 * n_filters, kernel_size=2, stride=2, dimension=2)
        self.us_conv_2 = ME.MinkowskiConvolution(4 * n_filters, 2 * n_filters, kernel_size=3, dimension=2)
        self.us_tconv_1 = ME.MinkowskiConvolutionTranspose(2 * n_filters, n_filters, kernel_size=2, stride=2, dimension=2)
        self.us_conv_1 = ME.MinkowskiConvolution(2 * n_filters, n_filters, kernel_size=3, dimension=2)
        self.output = ME.MinkowskiConvolution(n_filters, n_classes, kernel_size=1, dimension=2)

    def forward(self, x):
        conv_stack_1 = self.ds_conv_1(x)
        res = self.ds_maxpool(conv_stack_1)
        res = self.ds_dropout(res)
        conv_stack_2 = self.ds_conv_2(res) 
        res = self.ds_maxpool(conv_stack_2)
        res = self.ds_dropout(res)
        conv_stack_3 = self.ds_conv_3(res)
        res = self.ds_maxpool(conv_stack_3)
        res = self.ds_dropout(res)
        conv_stack_4 = self.ds_conv_4(res)
        res = self.ds_maxpool(conv_stack_4)
        res = self.ds_dropout(res)
        res = self.bridge(res)
        res = self.us_tconv_4(res)
        res = ME.cat(res, conv_stack_4)
        res = self.us_conv_4(res)
        res = self.us_tconv_3(res)
        res = ME.cat(res, conv_stack_3)
        res = self.us_conv_3(res)
        res = self.us_tconv_2(res)
        res = ME.cat(res, conv_stack_2)
        res = self.us_conv_2(res)
        res = self.us_tconv_1(res)
        res = ME.cat(res, conv_stack_1)
        res = self.us_conv_1(res)
        output = self.output(res)
        return output, conv_stack_2 

class AttentionWeighting(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(AttentionWeighting, self).__init__()
        self.conv = ME.MinkowskiConvolution(in_channels, in_channels // reduction, kernel_size=3, dimension=2)
        self.bn = ME.MinkowskiBatchNorm(in_channels // reduction)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.out_conv = ME.MinkowskiConvolution(in_channels // reduction, 1, kernel_size=1, dimension=2)
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return self.sigmoid(x)  

class AttentionWeightedBCELoss(nn.Module):
    def __init__(self, feature_channels):
        super(AttentionWeightedBCELoss, self).__init__()
        self.attention = AttentionWeighting(feature_channels)

    def forward(self, inputs, targets, features):
        attn_weights = self.attention(features) 
        weights = attn_weights.features.expand(-1, inputs.size(1))  
        weights = weights.clamp(0.1, 10.0)  
        return F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction='mean')
    
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

def train_model(model, dataloader, optimiser, criterion, device, args):
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=len(dataloader), T_mult=2, eta_min=0)
    train_losses = []
    valid_losses = []
    learning_rates = []
    precisions = {0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: []}
    recalls = {0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: []}

    for epoch in range(args.num_epochs):
        for batch, (images, masks, _, _, _) in enumerate(dataloader):
            split_idx = int(0.5 * len(images))
            train_images = images[:split_idx]
            train_masks = masks[:split_idx].to(device)
            valid_images = images[split_idx:]
            valid_masks = masks[split_idx:].to(device)

            model.train()
            train_images = train_images.to(device)
            optimiser.zero_grad()
            train_outputs, train_features = model(train_images) 
            train_loss = criterion(train_outputs.features, train_masks, train_features)
            train_losses.append(train_loss.item())
            train_loss.backward()
            optimiser.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                valid_outputs, valid_features = model(valid_images)
                valid_loss = criterion(valid_outputs.features, valid_masks, valid_features).item()
                valid_losses.append(valid_loss)

            precision_dict, recall_dict = compute_metrics(valid_outputs.features, valid_masks)
            for t in precision_dict:
                precisions[t].append(precision_dict[t].cpu().numpy())
                recalls[t].append(recall_dict[t].cpu().numpy())

            if batch % 1 == 0:
                print(f"Epoch {epoch}, Batch {batch}: Train Loss = {train_loss.item():.4f}, "
                      f"Valid Loss = {valid_loss:.4f}, Learning Rate = {scheduler.get_last_lr()[0]:.6f}, "
                      f"Precision (0.3) = {precisions[0.3][-1]}, Recall (0.3) = {recalls[0.3][-1]}, "
                      f"Precision (0.5) = {precisions[0.5][-1]}, Recall (0.5) = {recalls[0.5][-1]}")

            if batch % 100 == 0:
                model.eval()
                example_input = ME.SparseTensor(
                    features=torch.randn(1, 1).to(device),
                    coordinates=torch.tensor([[0, 256, 256]]).to(device)
                )
                traced_model = torch.jit.trace(model, example_input)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                traced_model.save(os.path.join(args.output_dir, f"unet_plane{args.plane}_{epoch}_{batch}_{timestamp}.pt"))

        model.eval()
        example_input = ME.SparseTensor(
            features=torch.randn(1, 1).to(device),
            coordinates=torch.tensor([[0, 256, 256]]).to(device)
        )
        traced_model = torch.jit.trace(model, example_input)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traced_model.save(os.path.join(args.output_dir, f"unet_plane{args.plane}_{epoch}_{timestamp}.pt"))

    return train_losses, valid_losses, learning_rates, precisions, recalls

def get_parser():
    parser = argparse.ArgumentParser(description="Sparse UNet for LArTPC")
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--target-labels", type=str, default="0,1,2,4")
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True) 
    parser.add_argument("--output-dir", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation")
    return parser

def main():
    args = get_parser().parse_args()
    target_labels = [int(x) for x in args.target_labels.split(',')]
    args.num_classes = len([lbl for lbl in target_labels if lbl >= 2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SparseImageDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    model = SparseUNet(in_dim=1, n_classes=args.num_classes).to(device)

    criterion = AttentionWeightedBCELoss(feature_channels=32)
    optimiser = Adam(model.parameters(), lr=args.learning_rate)
    os.makedirs(args.output_dir, exist_ok=True)

    train_losses, valid_losses, learning_rates, precisions, recalls = train_model(
        model, dataloader, optimiser, criterion, device, args
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(os.path.join(args.output_dir, f"results_plane{args.plane}_{timestamp}.npz"),
             train_losses=train_losses,
             valid_losses=valid_losses,
             learning_rates=learning_rates,
             precisions_03=precisions[0.3],
             recalls_03=recalls[0.3],
             precisions_05=precisions[0.5],
             recalls_05=recalls[0.5])

    label_mapping = {str(k): v for k, v in dataset.enum_to_model.items()}
    np.savez(os.path.join(args.output_dir, f"label_mapping_plane{args.plane}.npz"), **label_mapping)

if __name__ == "__main__":
    main()