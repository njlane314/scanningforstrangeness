import logging
import sys
import random
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uproot
import awkward as ak
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

### scp mpid_epoch_10.pt nlane@uboonegpvm03.fnal.gov:/exp/uboone/app/users/nlane/production/strangeness_mcc9/srcs/ubana/ubana/searchingforstrangeness/.
### scp nlane@uboonegpvm05.fnal.gov:/exp/uboone/app/users/nlane/analysis/rarexsec-analysis/filtered_numu_cc_events.root .

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

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
    ROOT_FILE_PATH: Path = Path("/gluster/home/niclane/scanningforstrangeness/filtered_numu_cc_events.root")
    TTREE_NAME: str = "mc_inclusive_run1_fhc"
    IMAGE_BRANCH: str = "detector_image_w"
    LABEL_BRANCH: str = "inclusive_strange_channels"
    
    CLASS_A_LABEL: int = 21
    CLASS_B_LABELS: List[int] = field(default_factory=lambda: [20, 22, 23, 30, 31])

    IMAGE_HEIGHT: int = 512
    IMAGE_WIDTH: int = 512
    ADC_THRESHOLD: float = 4.0
    BATCH_SIZE: int = 32
    VAL_SPLIT: float = 0.2
    NUM_WORKERS: int = 0
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 0.0001 
    DROPOUT_RATE: float = 0.5
    
    MODEL_OUTPUT_DIR: Path = Path("trained_binary_classifier")
    METRICS_OUTPUT_DIR: Path = Path("training_binary_classifier_metrics")
    NUM_CLASSES: int = 1

class MPID(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(MPID, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.GroupNorm(64, 64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(), nn.GroupNorm(64, 64), nn.AvgPool2d(2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.GroupNorm(96, 96),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1), nn.ReLU(), nn.GroupNorm(96, 96), nn.AvgPool2d(2, padding=1),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.GroupNorm(128, 128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1), nn.ReLU(), nn.GroupNorm(128, 128), nn.AvgPool2d(2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.GroupNorm(160, 160),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1), nn.ReLU(), nn.GroupNorm(160, 160), nn.AvgPool2d(2, padding=1),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.GroupNorm(192, 192),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1), nn.GroupNorm(192, 192), nn.AvgPool2d(2, padding=1)
        )
        self.dropout_features_output = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(192 * 8 * 8, 192 * 8),
            nn.Dropout(dropout), nn.Linear(192 * 8, 192), nn.Linear(192, num_classes)
        )
    def forward(self, x):
        x_features_output = self.features(x); x_flattened = torch.flatten(x_features_output, 1)
        x = self.dropout_features_output(x_flattened); x = self.classifier(x)
        return x

class ImageClassifierDataset(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        all_labels_raw = []
        image_shapes = []
        logging.info(f"Performing initial scan of {cfg.ROOT_FILE_PATH}...")
        try:
            with uproot.open(f"{self.cfg.ROOT_FILE_PATH}:{self.cfg.TTREE_NAME}") as tree:
                branches_to_scan = [self.cfg.LABEL_BRANCH, self.cfg.IMAGE_BRANCH]
                for chunk in tree.iterate(branches_to_scan, library="np", step_size="100 MB"):
                    all_labels_raw.extend(chunk[cfg.LABEL_BRANCH])
                    shapes_in_chunk = [len(img) for img in chunk[cfg.IMAGE_BRANCH]]
                    image_shapes.extend(shapes_in_chunk)
        except Exception as e:
            logging.error(f"Failed during initial file scan: {e}"); sys.exit(1)
        logging.info("Initial scan complete.")

        all_allowed_labels = [self.cfg.CLASS_A_LABEL] + self.cfg.CLASS_B_LABELS
        expected_image_size = self.cfg.IMAGE_HEIGHT * self.cfg.IMAGE_WIDTH
        
        self.valid_indices = []
        self.labels = []
        for i, (label, shape) in enumerate(zip(all_labels_raw, image_shapes)):
            if label in all_allowed_labels and shape == expected_image_size:
                self.valid_indices.append(i)
                binary_label = 0 if label == self.cfg.CLASS_A_LABEL else 1
                self.labels.append(binary_label)

        if not self.valid_indices:
            logging.error("No valid events found for any specified class. Cannot continue."); sys.exit(1)
        self.tree = None

    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tree is None:
            self.tree = uproot.open(f"{self.cfg.ROOT_FILE_PATH}:{self.cfg.TTREE_NAME}")
        global_root_index = self.valid_indices[idx]
        label_tensor = torch.tensor([float(self.labels[idx])], dtype=torch.float32)
        image_data = self.tree[self.cfg.IMAGE_BRANCH].array(entry_start=global_root_index, entry_stop=global_root_index + 1)
        image_np = ak.to_numpy(image_data[0])
        image_np = image_np.astype(np.float32).reshape(1, self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        image_np[image_np < self.cfg.ADC_THRESHOLD] = 0.0
        image_np = np.log1p(image_np)    
        image_tensor = torch.from_numpy(image_np)
        return image_tensor, label_tensor

def save_torchscript_model(model: nn.Module, path: Path, metadata: Dict[str, Any], cfg: Config):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_copy = copy.deepcopy(model)
    model_copy.cpu()
    model_copy.eval()
    example_input = torch.randn(1, 1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, device='cpu')
    traced_model = torch.jit.trace(model_copy, example_input)
    try:
        torch.jit.save(traced_model, str(path))
        logging.info(f"TorchScript model saved successfully to {path}")
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata saved separately to {metadata_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the model: {e}")

def train_classifier_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct = (predicted == labels).sum().item()
        batch_accuracy = (100 * correct / labels.size(0))
        optimizer.zero_grad()
        if torch.isnan(loss):
            logging.error("CRITICAL: Loss is NaN. Stopping training."); raise ValueError("Loss became NaN")
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}/{total_epochs} | Train Batch {i+1}/{len(train_loader)} | Train Loss: {loss.item():.4f}, Train Acc: {batch_accuracy:.2f}%", end='\r', flush=True)
    print()

def validate_classifier(model, loader, criterion, device):
    model.eval()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (100 * correct_predictions / total_samples) if total_samples > 0 else 0.0
    return avg_loss, accuracy

def train_classifier(cfg: Config, device: torch.device):
    logging.info(f"--- Training Binary Classifier: Class {cfg.CLASS_A_LABEL} vs Others ---")
    
    full_dataset = ImageClassifierDataset(cfg)

    train_size = int((1.0 - cfg.VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logging.info(f"Total valid events: {len(full_dataset)}")
    logging.info(f" -> Training on {len(train_dataset)} events.")
    logging.info(f" -> Validating on {len(val_dataset)} events.")

    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    num_samples = len(train_labels)
    
    if 0 in class_counts:
        logging.error("Training set has 0 samples for one of the classes. Cannot train."); return
    
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)
    logging.info(f"Training set composition: {class_counts[0]} events of Class A (label 0), {class_counts[1]} of Class B (label 1).")
    logging.info("Using WeightedRandomSampler for 50:50 batch composition.")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    
    model = MPID(num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT_RATE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)
    best_val_loss = float('inf')
    
    for epoch in range(cfg.NUM_EPOCHS):
        epoch_num = epoch + 1
        logging.info(f"--- Starting Epoch {epoch_num}/{cfg.NUM_EPOCHS} ---")
        train_classifier_epoch(model, train_loader, optimizer, criterion, device, epoch_num, cfg.NUM_EPOCHS)
        
        val_loss, val_acc = validate_classifier(model, val_loader, criterion, device)
        logging.info(f"End of Epoch {epoch_num} | Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = cfg.MODEL_OUTPUT_DIR / f"best_model_epoch_{epoch_num}.pt"
            metadata = {
                'description': 'Binary classifier separating one class from others',
                'class_a': cfg.CLASS_A_LABEL,
                'class_b': cfg.CLASS_B_LABELS,
                'epoch': epoch_num,
                'validation_loss': val_loss,
                'validation_accuracy': val_acc
            }
            save_torchscript_model(model, model_path, metadata, cfg)
            logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
    logging.info("Classifier training finished.")

def main():
    parser = argparse.ArgumentParser(description="Train a binary classifier.")
    args = parser.parse_args()
    cfg = Config()
    cfg.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_classifier(cfg, device)
    logging.info("Pipeline finished successfully!")

if __name__ == "__main__":
    main()