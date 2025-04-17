import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import uproot
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple, Dict, Any
import os
from datetime import datetime
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Train sparse UNet for LArTPC images with Minkowski Engine")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--root-file", type=str, 
                        default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--target-labels", type=str, default="0,1,2,4")
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output-dir", type=str, 
                        default="/gluster/data/dune/niclane/checkpoints/segmentation")
    return parser

class ResidualBlock(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(ResidualBlock, self).__init__(D)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dimension=D
        )
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dimension=D
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        self.skip = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dimension=D
        ) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = MF.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip is not None:
            residual = self.skip(residual)
        out += residual
        out = MF.relu(out)
        return out

class UResNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels: int, out_channels: int, D: int = 2):
        super(UResNet, self).__init__(D)
        self.initial_conv = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            dimension=D
        )
        self.level1 = ResidualBlock(32, 32, D)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=64,
            kernel_size=2,
            stride=2,
            dimension=D
        )
        self.level2 = ResidualBlock(64, 64, D)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=2,
            stride=2,
            dimension=D
        )
        self.level3 = ResidualBlock(128, 128, D)
        self.up2 = ME.MinkowskiConvolutionTranspose(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            dimension=D
        )
        self.level2_up = ResidualBlock(128, 64, D)  
        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2,
            dimension=D
        )
        self.level1_up = ResidualBlock(64, 32, D)
        self.final_bn = ME.MinkowskiBatchNorm(32)
        self.final_relu = ME.MinkowskiReLU()
        self.final_conv = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dimension=D
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        out = self.initial_conv(x)
        out_s1 = self.level1(out)
        out = self.down1(out_s1)
        out_s2 = self.level2(out)
        out = self.down2(out_s2)
        out = self.level3(out)
        out = self.up2(out)
        out = ME.cat(out, out_s2)
        out = self.level2_up(out)
        out = self.up1(out)
        out = ME.cat(out, out_s1)
        out = self.level1_up(out)
        out = self.final_bn(out)
        out = self.final_relu(out)
        out = self.final_conv(out, coordinate_map_key=x.coordinate_map_key)
        return out

class SparseDataset(Dataset):
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
        self.event_types = self.tree["type"].array(library="np")
        self.indices = np.where(self.event_types == 0)[0]
        self.num_events = len(self.indices)

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, int, int, int]:
        actual_idx = self.indices[idx]
        data = self._load_data(actual_idx)
        img = self._process_image(data["input"][0][self.plane])
        point_cloud = self._dense_to_sparse(img)
        truth = self._process_truth(data["truth"][0][self.plane])
        masks = self._generate_masks(truth)
        row_idx = point_cloud[0][:, 1].long()  
        col_idx = point_cloud[0][:, 0].long()  
        ground_truth = masks[:, row_idx, col_idx].T
        return point_cloud, ground_truth, data["run"][0], data["subrun"][0], data["event"][0]

    def _load_data(self, idx: int) -> Dict[str, Any]:
        return self.tree.arrays(
            ["input", "truth", "run", "subrun", "event"],
            entry_start=idx,
            entry_stop=idx + 1,
            library="np"
        )

    def _process_image(self, img_data: np.ndarray) -> np.ndarray:
        return np.fromiter(img_data, dtype=np.float32).reshape(self.img_size, self.img_size)

    def _dense_to_sparse(self, img: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = np.nonzero(img) 
        coords = np.stack([x, y], axis=1)  
        features = img[y, x] 
        coords = torch.from_numpy(coords).long()
        features = torch.from_numpy(features).float().unsqueeze(1)
        return coords, features

    def _process_truth(self, truth_data: np.ndarray) -> np.ndarray:
        return np.fromiter(truth_data, dtype=np.int64).reshape(self.img_size, self.img_size)

    def _generate_masks(self, truth: np.ndarray) -> torch.Tensor:
        return torch.stack([torch.from_numpy(truth == lbl).float() for lbl in self.foreground_labels])

def collate_fn(batch):
    locations = []
    features = []
    ground_truth_list = []
    run, subrun, event = [], [], []
    for batchIdx, (point_cloud, gt, r, s, e) in enumerate(batch):
        coords, feat = point_cloud
        batch_idx = torch.full((coords.shape[0], 1), batchIdx, dtype=torch.long)
        coords_with_batch = torch.cat([batch_idx, coords], dim=1)  
        locations.append(coords_with_batch)
        features.append(feat)
        ground_truth_list.append(gt)
        run.append(r)
        subrun.append(s)
        event.append(e)
    batched_locations = torch.cat(locations, dim=0)
    batched_features = torch.cat(features, dim=0)
    batched_ground_truth = torch.cat(ground_truth_list, dim=0)
    return (batched_locations, batched_features), batched_ground_truth, torch.tensor(run), torch.tensor(subrun), torch.tensor(event)

def compute_metrics(preds, targets, thresholds=[0.5]):
    metrics = {"sensitivity": {}, "specificity": {}}
    for t in thresholds:
        preds_t = torch.sigmoid(preds) > t
        targets = targets.bool()
        tp = (preds_t & targets).float().sum(dim=0)
        fp = (preds_t & ~targets).float().sum(dim=0)
        fn = (~preds_t & targets).float().sum(dim=0)
        tn = ((~preds_t) & (~targets)).float().sum(dim=0)
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        metrics["sensitivity"][t] = sensitivity
        metrics["specificity"][t] = specificity
    return metrics

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        probs = torch.sigmoid(output)
        intersection = (gt * probs).sum(dim=0)
        union = gt.sum(dim=0) + probs.sum(dim=0)
        dice_coeff = (2. * intersection) / (union + self.epsilon)
        dice_loss = 1 - dice_coeff
        loss = dice_loss.mean()
        return loss

class Trainer:
    def __init__(self, dataset, device, args):
        self.model = UResNet(in_channels=1, out_channels=args.classes, D=2).to(device)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, collate_fn=collate_fn)
        self.optimiser = Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = DiceLoss()
        self.device = device
        self.args = args
        self.scheduler = CosineAnnealingWarmRestarts(self.optimiser, T_0=len(self.dataloader), T_mult=2, eta_min=0)

    def iterate(self, epoch: int) -> Tuple[list, list, list, list, list, list, list]:
        train_losses, valid_losses, learning_rates = [], [], []
        train_sensitivity_list, train_specificity_list = [], []
        valid_sensitivity_list, valid_specificity_list = [], []

        for batch, (point_cloud, ground_truth, _, _, _) in enumerate(self.dataloader):
            coords, features = point_cloud
            coords = coords.to(self.device)
            features = features.to(self.device)
            ground_truth = ground_truth.to(self.device)
            batch_indices = coords[:, 0].unique()
            batch_size = len(batch_indices)
            split = batch_size // 2
            train_batch_indices = batch_indices[:split]
            valid_batch_indices = batch_indices[split:]
            train_mask = torch.isin(coords[:, 0], train_batch_indices)
            valid_mask = torch.isin(coords[:, 0], valid_batch_indices)

            input_tensor = ME.SparseTensor(features=features, coordinates=coords, device=self.device)

            self.model.train()
            self.optimiser.zero_grad()
            output = self.model(input_tensor)
            train_output = output.F[train_mask]
            train_ground_truth = ground_truth[train_mask]
            train_loss = self.criterion(train_output, train_ground_truth)
            train_losses.append(train_loss.item())
            train_loss.backward()
            learning_rates.append(self.scheduler.get_last_lr()[0])
            self.optimiser.step()
            self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                valid_output = output.F[valid_mask]
                valid_ground_truth = ground_truth[valid_mask]
                valid_loss = self.criterion(valid_output, valid_ground_truth)
                valid_losses.append(valid_loss.item())

            train_metrics = compute_metrics(train_output, train_ground_truth)
            valid_metrics = compute_metrics(valid_output, valid_ground_truth)
            threshold = 0.5
            train_sensitivity_list.append(train_metrics["sensitivity"][threshold].cpu().numpy())
            train_specificity_list.append(train_metrics["specificity"][threshold].cpu().numpy())
            valid_sensitivity_list.append(valid_metrics["sensitivity"][threshold].cpu().numpy())
            valid_specificity_list.append(valid_metrics["specificity"][threshold].cpu().numpy())

            if batch % 1 == 0:
                print(f"Epoch {epoch}, Batch {batch}: Train Loss = {train_loss.item():.4f}, "
                      f"Valid Loss = {valid_loss.item():.4f}, Learning Rate = {self.scheduler.get_last_lr()[0]:.6f}")
                for cls in range(self.args.classes):
                    train_sens = train_metrics["sensitivity"][threshold][cls]
                    train_spec = train_metrics["specificity"][threshold][cls]
                    valid_sens = valid_metrics["sensitivity"][threshold][cls]
                    valid_spec = valid_metrics["specificity"][threshold][cls]
                    print(f"Class {cls}: Train Sensitivity: {train_sens:.4f}, Train Specificity: {train_spec:.4f}, "
                          f"Valid Sensitivity: {valid_sens:.4f}, Valid Specificity: {valid_spec:.4f}")

        return (train_losses, valid_losses, learning_rates,
                train_sensitivity_list, train_specificity_list,
                valid_sensitivity_list, valid_specificity_list)

def main():
    args = get_parser().parse_args()
    target_labels = [int(x) for x in args.target_labels.split(',')]
    args.classes = len([lbl for lbl in target_labels if lbl >= 2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SparseDataset(args)
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = Trainer(dataset, device, args)
    train_losses, valid_losses, learning_rates = [], [], []
    for epoch in range(args.epochs):
        epoch_train_losses, epoch_valid_losses, epoch_learning_rates = trainer.iterate(epoch)
        train_losses.extend(epoch_train_losses)
        valid_losses.extend(epoch_valid_losses)
        learning_rates.extend(epoch_learning_rates)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(os.path.join(args.output_dir, f"results_plane{args.plane}_{timestamp}.npz"),
             train_losses=train_losses,
             valid_losses=valid_losses,
             learning_rates=learning_rates)
    label_mapping = {str(k): v for k, v in dataset.enum_to_model.items()}
    np.savez(os.path.join(args.output_dir, f"label_mapping_plane{args.plane}.npz"), **label_mapping)

if __name__ == "__main__":
    main()