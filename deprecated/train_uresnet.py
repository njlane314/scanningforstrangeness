import logging
import sys
import random
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uproot
import awkward as ak
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset, Subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    ROOT_FILE_PATH: Path = Path("/gluster/data/dune/niclane/scanning/numucc_snapshot.root")
    TTREE_NAME: str = "mc_strangeness_run1_fhc"
    TARGET_PLANE: str = "w"
    LABEL_BRANCH: str = "analysis_channel"

    SEG_CLASS_DEFINITIONS: Dict[int, str] = field(default_factory=lambda: {
        0: "Empty", 1: "Cosmic", 2: "Muon", 3: "Proton", 4: "Pion",
        5: "ChargedKaon", 6: "NeutralKaon", 7: "Lambda", 8: "ChargedSigma", 9: "Other"
    })

    IMAGE_HEIGHT: int = 512
    IMAGE_WIDTH: int = 512
    BATCH_SIZE: int = 16
    VAL_SPLIT: float = 0.2
    NUM_WORKERS: int = 0
    ADC_THRESHOLD: float = 4.0
    IGNORE_INDEX: int = 0

    URESNET_EPOCHS: int = 10
    URESNET_LR: float = 1e-4

    MLP_EPOCHS: int = 20
    MLP_LR: float = 1e-4
    MLP_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [1024, 512])

    MODEL_OUTPUT_DIR: Path = Path("trained_uresnet_models")
    METRICS_OUTPUT_DIR: Path = Path("training_uresnet_metrics")
    BEST_URESNET_PATH_FILE: Path = MODEL_OUTPUT_DIR / "best_uresnet_model.txt"

    MLP_CLASS_INDICES: Dict[str, int] = field(init=False)
    NUM_SEG_CLASSES: int = field(init=False)
    NUM_MLP_CLASSES: int = field(init=False)

    def __post_init__(self):
        self.NUM_SEG_CLASSES = len(self.SEG_CLASS_DEFINITIONS)
        
        mlp_indices = {
            name: i for i, name in enumerate(
                [cls_name for cls_val, cls_name in self.SEG_CLASS_DEFINITIONS.items() if cls_name not in ["Empty", "Cosmic", "Other"]]
            )
        }
        self.MLP_CLASS_INDICES = mlp_indices
        self.NUM_MLP_CLASSES = len(self.MLP_CLASS_INDICES)

class SegmentationDataset(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            self.file = uproot.open(self.cfg.ROOT_FILE_PATH)
        except FileNotFoundError:
            logging.error(f"ROOT file not found at: {self.cfg.ROOT_FILE_PATH}")
            sys.exit(1)
            
        if cfg.TTREE_NAME not in self.file:
            raise KeyError(f"TTree '{cfg.TTREE_NAME}' not found in '{cfg.ROOT_FILE_PATH}'. Available keys: {self.file.keys()}")
        self.tree = self.file[cfg.TTREE_NAME]

        all_labels = self.tree[self.cfg.LABEL_BRANCH].array(library="np")
        self.signal_indices = np.where((all_labels == 10) | (all_labels == 11))[0]
        
        if len(self.signal_indices) == 0:
            raise ValueError(f"Found 0 signal events (analysis_channel 10 or 11) in TTree '{cfg.TTREE_NAME}'. Please check your file and TTree name.")
        
        self.raw_branch = f"raw_image_{cfg.TARGET_PLANE}"
        self.true_branch = f"true_image_{cfg.TARGET_PLANE}"
        self.branches_to_load = [self.raw_branch, self.true_branch]

    def __len__(self) -> int:
        return len(self.signal_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.signal_indices[idx]
        entry = self.tree.arrays(
            self.branches_to_load, 
            entry_start=actual_idx, 
            entry_stop=actual_idx + 1, 
            library="np"
        )

        raw_image = entry[self.raw_branch][0].astype(np.float32).reshape(1, self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        raw_image[raw_image < self.cfg.ADC_THRESHOLD] = 0.0
        raw_image = np.log1p(raw_image)
        raw_tensor = torch.from_numpy(raw_image)

        true_image = entry[self.true_branch][0].astype(np.int64).reshape(self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        true_tensor = torch.from_numpy(true_image)
        logging.info(f"Event Index {idx}: Unique pixel values in true_image: {np.unique(true_image)}")

        mlp_label = torch.zeros(self.cfg.NUM_MLP_CLASSES, dtype=torch.float32)
        unique_pixels = torch.unique(true_tensor)
        for pixel_val, class_name in self.cfg.SEG_CLASS_DEFINITIONS.items():
            if class_name in self.cfg.MLP_CLASS_INDICES and pixel_val in unique_pixels:
                mlp_idx = self.cfg.MLP_CLASS_INDICES[class_name]
                mlp_label[mlp_idx] = 1.0

        return raw_tensor, true_tensor, mlp_label

class FixedNormalization(nn.Module):
    def __init__(self, mean: float, std: float):
        super(FixedNormalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SELayer(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return out

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UResNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.normalization = FixedNormalization(mean, std)
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024))
        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return nn.functional.adaptive_avg_pool2d(x5, (1, 1)).flatten(1)

class ParticleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def save_torchscript_model(model: nn.Module, path: Path, metadata: Dict[str, Any], cfg: Config): # Added cfg
    path.parent.mkdir(parents=True, exist_ok=True)
    
    model_copy = copy.deepcopy(model)
    model_copy.cpu()
    model_copy.eval()

    example_input = torch.randn(1, 1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, device='cpu')
    
    traced_model = torch.jit.trace(model_copy, example_input)

    try:
        torch.jit.save(traced_model, str(path))
        logging.info(f"TorchScript model saved successfully to {path}")
        logging.info(f"  - Method: Trace")
        logging.info(f"  - Target Device: CPU")
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"  - Metadata: Saved separately to {metadata_path}")

    except Exception as e:
        logging.error(f"An error occurred while saving the model: {e}")

def calculate_statistics(data_subset: Subset, sample_size: int = 1000):
    num_samples_to_use = min(sample_size, len(data_subset))
    
    random_indices = np.random.choice(len(data_subset), size=num_samples_to_use, replace=False)
    sampler = torch.utils.data.Subset(data_subset.dataset, [data_subset.indices[i] for i in random_indices])
    
    loader = DataLoader(sampler, batch_size=64, num_workers=0, shuffle=False)
    
    n_pixels = 0
    sum_ = torch.zeros(1)
    sum_sq = torch.zeros(1)

    logging.info(f"Calculating stats on subsample of images...")
    for images, _ in loader:
        sum_ += torch.sum(images)
        sum_sq += torch.sum(images ** 2)
        n_pixels += images.nelement()

    mean = sum_ / n_pixels
    std = torch.sqrt((sum_sq / n_pixels) - (mean ** 2))
    
    return mean.item(), std.item()

def train_uresnet_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    batch_losses = []
    num_batches = len(loader)

    for i, batch in enumerate(loader):
        images, seg_labels, _ = batch
        images, seg_labels = images.to(device), seg_labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, seg_labels)
        if torch.isnan(loss):
            logging.error("CRITICAL: Loss is NaN. Stopping training.")
            raise ValueError("Loss became NaN")
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if (i + 1) % 100 == 0: 
            print(f'Batch {i+1}, Gradient Norm Before Clipping: {total_norm:.4f}', flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        current_loss = loss.item()
        batch_losses.append(current_loss)

        logging.info(f"Train U-ResNet | Epoch {epoch}/{total_epochs} | Batch {i+1}/{num_batches} | Batch Loss: {current_loss:.4f}")
        sys.stdout.flush()
        
    return batch_losses

def validate_segmentation(model, loader, criterion, device, ignore_index):
    model.eval()
    total_loss, correct_pixels, total_pixels = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images, seg_labels, _ = batch
            images, seg_labels = images.to(device), seg_labels.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, seg_labels)
            total_loss += loss.item()

    
            mask = seg_labels != ignore_index
        
            _, predicted = torch.max(outputs, 1)
            total_pixels += mask.sum().item()
            correct_pixels += (predicted[mask] == seg_labels[mask]).sum().item()

    val_acc = (100 * correct_pixels / total_pixels) if total_pixels > 0 else 0
    return total_loss / len(loader), val_acc

def train_uresnet(cfg, device, smoke_test=False):
    logging.info("--- Training U-ResNet for Semantic Segmentation ---")
    dataset = SegmentationDataset(cfg)
    logging.info(f"Found {len(dataset.signal_indices)} signal events for target plane '{cfg.TARGET_PLANE}'.")
    val_size = int(cfg.VAL_SPLIT * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    
    logging.info("Calculating normalisation statistics from training set...")
    mean, std = calculate_statistics(train_dataset)
    logging.info(f"Calculated Mean: {mean:.4f}, Std: {std:.4f}. These will be fixed in the model.")

    if smoke_test:
        logging.info("--- Running in OVERFIT TEST mode: training on a single batch repeatedly. ---")
        logging.info(f"Will run for {cfg.NUM_EPOCHS} epochs to check for overfitting.")

        temp_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
        single_batch_images, single_batch_seg_labels, _ = next(iter(temp_loader))

        overfit_dataset = TensorDataset(single_batch_images, single_batch_seg_labels)
        train_loader = DataLoader(overfit_dataset, batch_size=cfg.BATCH_SIZE)
        val_loader = DataLoader(overfit_dataset, batch_size=cfg.BATCH_SIZE)

        logging.info(f"Created a static dataset of size {len(single_batch_images)} to check for overfitting.")
    else: 
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    logging.info("Calculating class weights for segmentation...")
    temp_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    class_pixel_counts = torch.zeros(cfg.NUM_SEG_CLASSES, dtype=torch.int64)

    for _, seg_labels, _ in temp_loader:
        flat_labels = seg_labels.flatten().to(torch.int64)
        batch_counts = torch.bincount(flat_labels, minlength=cfg.NUM_SEG_CLASSES)
        class_pixel_counts += batch_counts.cpu()

    logging.info(f"Total pixel counts per class: {class_pixel_counts.tolist()}")

    seg_class_weights = torch.zeros(cfg.NUM_SEG_CLASSES, dtype=torch.float32)
    
    active_classes_mask = torch.ones(cfg.NUM_SEG_CLASSES, dtype=torch.bool)
    active_classes_mask[cfg.IGNORE_INDEX] = False

    active_class_counts = class_pixel_counts[active_classes_mask]

    if torch.sum(active_class_counts) > 0:
        total_active_pixels = active_class_counts.sum()
        class_frequencies = active_class_counts.float() / total_active_pixels
        
        present_frequencies = class_frequencies[class_frequencies > 0]
        if len(present_frequencies) > 0:
            median_frequency = torch.median(present_frequencies)
            
            weights = median_frequency / (class_frequencies + 1e-6)
            seg_class_weights[active_classes_mask] = weights

    logging.info(f"Final calculated segmentation weights: {seg_class_weights.tolist()}")

    uresnet = UResNet(n_channels=1, n_classes=cfg.NUM_SEG_CLASSES, mean=mean, std=std).to(device)
    
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for U-ResNet training.")
        uresnet = nn.DataParallel(uresnet)

    seg_criterion = nn.CrossEntropyLoss(weight=seg_class_weights.to(device), ignore_index=cfg.IGNORE_INDEX)
    optimizer = optim.Adam(uresnet.parameters(), lr=cfg.URESNET_LR, weight_decay=1e-4)
    
    all_batch_losses, epoch_val_losses, epoch_val_accs, epoch_lrs = [], [], [], []
    best_val_acc = -1.0
    
    for epoch in range(cfg.URESNET_EPOCHS):
        epoch_num = epoch + 1
        current_lr = optimizer.param_groups[0]['lr']
        epoch_lrs.append(current_lr)
        logging.info(f"-- Starting Epoch {epoch_num} with Learning Rate: {current_lr}")

        batch_losses = train_uresnet_epoch(uresnet, train_loader, optimizer, seg_criterion, device, epoch_num, cfg.URESNET_EPOCHS)
        all_batch_losses.extend(batch_losses)
        
        logging.info("Validating U-ResNet...")
        val_loss, val_acc = validate_segmentation(uresnet, val_loader, seg_criterion, device, cfg.IGNORE_INDEX)
        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_acc)

        if smoke_test:
            logging.info(f"--- End of Overfit Epoch {epoch_num} | Loss on Batch: {val_loss:.4f} | Accuracy on Batch: {val_acc:.2f}% ---")
        else:
            logging.info(f"--- End of Epoch {epoch_num} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}% ---")
        sys.stdout.flush()

        model_path = cfg.MODEL_OUTPUT_DIR / f"uresnet_epoch_{epoch_num}.pt"
        save_torchscript_model(uresnet, model_path, {'class_definitions': cfg.SEG_CLASS_DEFINITIONS})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = model_path
            with open(cfg.BEST_URESNET_PATH_FILE, 'w') as f:
                f.write(str(best_model_path))
            logging.info(f"New best model path saved: {best_model_path}")

    cfg.METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.METRICS_OUTPUT_DIR / "uresnet_training_metrics.npz"
    np.savez(metrics_path,
             batch_losses=np.array(all_batch_losses),
             epoch_validation_losses=np.array(epoch_val_losses),
             epoch_validation_accuracies=np.array(epoch_val_accs),
             epoch_learning_rates=np.array(epoch_lrs))
    logging.info(f"U-ResNet training metrics saved to {metrics_path}")

def train_mlp(cfg, device):
    logging.info("--- Training MLP on Extracted Features ---")
    if not cfg.BEST_URESNET_PATH_FILE.exists():
        logging.error("Best U-ResNet model path file not found. Please run U-ResNet training first.")
        sys.exit(1)

    with open(cfg.BEST_URESNET_PATH_FILE, 'r') as f:
        best_uresnet_path = f.read().strip()
    
    logging.info(f"Loading U-ResNet for feature extraction from: {best_uresnet_path}")
    uresnet_extractor = torch.jit.load(best_uresnet_path, map_location=device)
    uresnet_extractor.eval()

    dataset = SegmentationDataset(cfg)
    full_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    all_features, all_mlp_labels = [], []
    with torch.no_grad():
        for batch in full_loader:
            images, _, mlp_labels = batch
            features = uresnet_extractor.encode(images.to(device))
            all_features.append(features.cpu())
            all_mlp_labels.append(mlp_labels)

    features_tensor = torch.cat(all_features, dim=0)
    mlp_labels_tensor = torch.cat(all_mlp_labels, dim=0)
    logging.info(f"Extracted features shape: {features_tensor.shape}, MLP labels shape: {mlp_labels_tensor.shape}")

    feature_dataset = TensorDataset(features_tensor, mlp_labels_tensor)
    val_size_mlp = int(cfg.VAL_SPLIT * len(feature_dataset))
    train_feat_dataset, _ = random_split(feature_dataset, [len(feature_dataset) - val_size_mlp, val_size_mlp])
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    logging.info("Calculating positive class weights for MLP...")
    train_labels = train_feat_dataset.dataset.tensors[1][train_feat_dataset.indices]
    
    positive_counts = train_labels.sum(dim=0)
    negative_counts = len(train_labels) - positive_counts
    mlp_pos_weights = negative_counts / (positive_counts + 1e-6)
    mlp_pos_weights = mlp_pos_weights.to(device)
    logging.info(f"Calculated MLP Positive Weights: {mlp_pos_weights.cpu().numpy()}")
    
    mlp = ParticleMLP(input_dim=features_tensor.shape[1], hidden_dims=cfg.MLP_HIDDEN_DIMS, output_dim=cfg.NUM_MLP_CLASSES).to(device)
    
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for MLP training.")
        mlp = nn.DataParallel(mlp)

    mlp_criterion = nn.BCEWithLogitsLoss(pos_weight=mlp_pos_weights.float())
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=cfg.MLP_LR)
    
    all_mlp_batch_losses = []
    for epoch in range(cfg.MLP_EPOCHS):
        epoch_num = epoch + 1
        mlp.train()
        num_batches = len(train_feat_loader)
        for i, (features, labels) in enumerate(train_feat_loader):
            features, labels = features.to(device), labels.to(device)
            mlp_optimizer.zero_grad()
            outputs = mlp(features)
            loss = mlp_criterion(outputs, labels)
            loss.backward()
            mlp_optimizer.step()
            
            current_loss = loss.item()
            all_mlp_batch_losses.append(current_loss)
            logging.info(f"Train MLP | Epoch {epoch_num}/{cfg.MLP_EPOCHS} | Batch {i+1}/{num_batches} | Batch Loss: {current_loss:.4f}")
            sys.stdout.flush()
        
        model_path = cfg.MODEL_OUTPUT_DIR / f"mlp_epoch_{epoch_num}.pt"
        save_torchscript_model(mlp, model_path, {'class_definitions': cfg.MLP_CLASS_INDICES})
    
    cfg.METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.METRICS_OUTPUT_DIR / "mlp_training_metrics.npz"
    np.savez(metrics_path, batch_losses=np.array(all_mlp_batch_losses))
    logging.info(f"MLP training metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="U-ResNet and MLP Training Pipeline")
    parser.add_argument("--run", type=str, choices=['train-uresnet', 'train-mlp'], default=None, help="Specify which part to run: 'train-uresnet', 'train-mlp', or run both if not specified.")
    parser.add_argument("--smoke-test", action="store_true", help="Run in single-batch overfitting mode to check model sanity.")
    args = parser.parse_args()

    cfg = Config()
    cfg.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.run is None or args.run == 'train-uresnet':
        train_uresnet(cfg, device, smoke_test=args.smoke_test)
    
    if args.run is None or args.run == 'train-mlp':
        train_mlp(cfg, device)

    logging.info("Pipeline finished successfully!")

if __name__ == "__main__":
    main()