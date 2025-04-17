import torch
import torch.nn as nn
import sparseconvnet as scn
import uproot
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple, Dict, Any
import os
from datetime import datetime
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Train sparse UNet for LArTPC images")
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

class UResNet(torch.nn.Module):
    def __init__(self, img_size: int, num_classes: int):
        super(UResNet, self).__init__()
        self.dimensions = 2
        self.spatial_size = (img_size, img_size)
        self.kernel_size = 2
        self.input_features = 1
        self.output_features = 32
        self.filter_size = 3
        self.repetitions = 2
        self.planes = [self.output_features] + [(2 ** i) * self.output_features for i in range(1, self.filter_size)]
        self.num_classes = num_classes
        self.img_size = img_size  
        self._build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _build_model(self) -> None:
        self.model = scn.Sequential()
        self.model.add(scn.InputLayer(self.dimensions, self.spatial_size, mode=3))
        self.model.add(scn.SubmanifoldConvolution(self.dimensions, self.input_features, self.output_features, self.filter_size, False))
        self.model.add(scn.UNet(self.dimensions, self.repetitions, self.planes, residual_blocks=True, downsample=[self.kernel_size, 2]))
        self.model.add(scn.BatchNormReLU(self.planes[0]))
        self.model.add(scn.OutputLayer(self.dimensions))
        self.linear = torch.nn.Linear(self.planes[0], self.num_classes)

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        print(f"Coordinates: type={coords.dtype}, device={coords.device}")
        print(f"Features: type={features.dtype}, device={features.device}")
        x = self.model((coords, features))
        x = self.linear(x)
        return x

class SparseDataset(Dataset):
    def __init__(self, args) -> None:
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
        row_idx = point_cloud[0][:, 0].long()  
        col_idx = point_cloud[0][:, 1].long()
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
        y, x = np.nonzero(img)  
        coords = np.stack([y, x], axis=1) 
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
        self.model = UResNet(args.img_size, args.classes).to(device)
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
            batch_indices = coords[:, 0].unique()
            batch_size = len(batch_indices)
            split = batch_size // 2
            train_batch_indices = batch_indices[:split]
            train_mask = torch.isin(coords[:, 0], train_batch_indices)
            train_coords = coords[train_mask].to(self.device) 
            train_features = features[train_mask].to(self.device) 
            train_ground_truths = ground_truth[train_mask].to(self.device)  
            valid_batch_indices = batch_indices[split:]
            valid_mask = torch.isin(coords[:, 0], valid_batch_indices)
            valid_coords = coords[valid_mask].to(self.device) 
            valid_features = features[valid_mask].to(self.device) 
            valid_ground_truths = ground_truth[valid_mask].to(self.device) 

            self.model.train()
            self.optimiser.zero_grad()
            train_outputs = self.model(train_coords, train_features)
            train_loss = self.criterion(train_outputs, train_ground_truths)
            train_losses.append(train_loss.item())
            train_loss.backward()
            learning_rates.append(self.scheduler.get_last_lr()[0])
            self.optimiser.step()
            self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                valid_outputs = self.model(valid_coords, valid_features)
                valid_loss = self.criterion(valid_outputs, valid_ground_truths)
                valid_losses.append(valid_loss.item())

            train_metrics = compute_metrics(train_outputs, train_ground_truths)
            valid_metrics = compute_metrics(valid_outputs, valid_ground_truths)
    
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
