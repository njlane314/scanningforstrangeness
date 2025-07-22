import logging
import sys
import random
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
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

import MinkowskiEngine as ME

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
    
    CLASS_A_LABEL: int = 20
    CLASS_B_LABELS: List[int] = field(default_factory=lambda: [21])

    IMAGE_HEIGHT: int = 512
    IMAGE_WIDTH: int = 512
    ADC_THRESHOLD: float = 4.0
    
    DIMENSION: int = 2
    MINK_CHANNELS: List[int] = field(default_factory=lambda: [32, 48, 64, 96, 128])
    
    BATCH_SIZE: int = 64
    VAL_SPLIT: float = 0.2
    NUM_WORKERS: int = 0
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 0.01  # Adjusted for Adam optimizer
    
    MODEL_OUTPUT_DIR: Path = Path("trained_minkowski_classifier")
    METRICS_OUTPUT_DIR: Path = Path("training_minkowski_classifier_metrics")
    NUM_CLASSES: int = 1

class MinkowskiFCNN(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        D,
        channels,
        embedding_channel=1024,
    ):
        super(MinkowskiFCNN, self).__init__(D)
        self.D = D
        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
    ):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(
            channels[0], channels[1], kernel_size=kernel_size, stride=1
        )
        self.conv2 = self.get_conv_block(
            channels[1], channels[2], kernel_size=kernel_size, stride=2
        )
        self.conv3 = self.get_conv_block(
            channels[2], channels[3], kernel_size=kernel_size, stride=2
        )
        self.conv4 = self.get_conv_block(
            channels[3], channels[4], kernel_size=kernel_size, stride=2
        )
        
        # This block processes the concatenated feature maps from the encoder path
        self.conv5 = nn.Sequential(
            self.get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
            ),
             self.get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
            ),
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=kernel_size, stride=2, dimension=self.D)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x0 = x.sparse() 
        x_processed = self.mlp1(x0)
        
        y = self.conv1(x_processed)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)
        
        y = self.conv4(y3)
        y4 = self.pool(y)
        
        # Use .slice() to bring feature maps to the same resolution for concatenation
        x1 = y1.slice(x0)
        x2 = y2.slice(x0)
        x3 = y3.slice(x0)
        x4 = y4.slice(x0)
        
        # Concatenate along the feature dimension
        x_cat = ME.cat(x1, x2, x3, x4)

        # Process the concatenated feature map
        y = self.conv5(x_cat.sparse())
        
        # Global pooling and final classification
        x_max = self.global_max_pool(y)
        x_avg = self.global_avg_pool(y)
        
        return self.final(ME.cat(x_max, x_avg)).F

def minkowski_collate_fn(list_data):
    coords, feats, labels = list(zip(*list_data))
    labels = torch.cat([torch.tensor([l], dtype=torch.float32) for l in labels])
    
    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.concatenate(feats, axis=0)).float()
    
    return {
        "coordinates": coords_batch,
        "features": feats_batch,
        "labels": labels,
    }

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
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        if self.tree is None:
            self.tree = uproot.open(f"{self.cfg.ROOT_FILE_PATH}:{self.cfg.TTREE_NAME}")
        
        global_root_index = self.valid_indices[idx]
        label = self.labels[idx]
        
        image_data = self.tree[self.cfg.IMAGE_BRANCH].array(entry_start=global_root_index, entry_stop=global_root_index + 1)
        image_np = ak.to_numpy(image_data[0])
        image_np = image_np.astype(np.float32).reshape(self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        
        coords = np.argwhere(image_np > self.cfg.ADC_THRESHOLD).astype(np.int32)
        
        if len(coords) == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.array([[np.log1p(self.cfg.ADC_THRESHOLD)]], dtype=np.float32)
        else:
            feats = image_np[coords[:, 0], coords[:, 1]]
            feats = np.log1p(feats).reshape(-1, 1).astype(np.float32)
            
        return coords, feats, label

def save_torchscript_model(model: nn.Module, path: Path, metadata: Dict[str, Any], cfg: Config):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    logging.info("Attempting to trace model for TorchScript...")
    
    try:
        # Create a new instance of the model for tracing to avoid deepcopy issues
        trace_model = MinkowskiFCNN(
            in_channel=1,
            out_channel=cfg.NUM_CLASSES,
            D=cfg.DIMENSION,
            channels=cfg.MINK_CHANNELS
        )
        trace_model.load_state_dict(model.state_dict())
        trace_model.cpu()
        trace_model.eval()

        num_example_points = 100
        coords_list = [torch.randint(0, 100, (num_example_points, cfg.DIMENSION), dtype=torch.int32)]
        example_coords_batched = ME.utils.batched_coordinates(coords_list).cpu()
        example_feats = torch.randn(num_example_points, 1).cpu()

        example_input = ME.TensorField(
            features=example_feats,
            coordinates=example_coords_batched,
            device='cpu'
        )

        traced_model = torch.jit.trace(trace_model, example_input)
        
        torch.jit.save(traced_model, str(path))
        logging.info(f"TorchScript model saved successfully to {path}")
        
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata saved separately to {metadata_path}")

    except Exception as e:
        logging.error(f"Failed to trace or save TorchScript model. Error: {e}", exc_info=True)
        logging.warning("FALLBACK: Saving model state_dict instead.")
        fallback_path = path.with_suffix('.pt.state_dict')
        # Save the original model's state_dict
        torch.save(model.cpu().state_dict(), str(fallback_path))
        logging.info(f"Model state_dict saved successfully to {fallback_path}")

def train_classifier_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    
    ema_loss = 0.0
    ema_acc = 0.0
    alpha = 0.1 

    for i, data_dict in enumerate(train_loader):
        # CORRECTED SECTION: Move all tensors to the target device FIRST
        coords = data_dict['coordinates'].to(device)
        feats = data_dict['features'].to(device)
        labels = data_dict['labels'].to(device).view(-1, 1)

        # Now, create the sparse tensor from the tensors that are on the correct device
        s_input = ME.TensorField(features=feats, coordinates=coords, device=device)
        
        # This will now work, as model and s_input are on the same device
        outputs = model(s_input)
        
        # The loss calculation will also work, as outputs and labels are on the same device
        loss = criterion(outputs, labels)

        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct = (predicted == labels).sum().item()
        batch_accuracy = (100 * correct / labels.size(0))
        
        if i == 0:
            ema_loss = loss.item()
            ema_acc = batch_accuracy
        else:
            ema_loss = alpha * loss.item() + (1 - alpha) * ema_loss
            ema_acc = alpha * batch_accuracy + (1 - alpha) * ema_acc
        
        optimizer.zero_grad()
        if torch.isnan(loss):
            logging.error("CRITICAL: Loss is NaN. Stopping training."); raise ValueError("Loss became NaN")
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{total_epochs} | Batch {i+1}/{len(train_loader)} | Moving Avg Loss: {ema_loss:.4f}, Moving Avg Acc: {ema_acc:.2f}%", end='\r', flush=True)
    
    print()

def validate_classifier(model, loader, criterion, device):
    model.eval()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for data_dict in loader:
            # CORRECTED SECTION: Move all tensors to the target device FIRST
            coords = data_dict['coordinates'].to(device)
            feats = data_dict['features'].to(device)
            labels = data_dict['labels'].to(device).view(-1, 1)

            # Create the sparse tensor from the tensors on the correct device
            s_input = ME.TensorField(features=feats, coordinates=coords, device=device)

            # This will now work
            outputs = model(s_input)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            
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
    
    if 0 in class_counts:
        logging.error("Training set has 0 samples for one of the classes. Cannot train."); return
    
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(train_labels))
    logging.info(f"Training set composition: {class_counts[0]} events of Class A (label 0), {class_counts[1]} of Class B (label 1).")
    logging.info("Using WeightedRandomSampler for balanced batch composition.")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=cfg.NUM_WORKERS, collate_fn=minkowski_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, collate_fn=minkowski_collate_fn)
    
    model = MinkowskiFCNN(
        in_channel=1, 
        out_channel=cfg.NUM_CLASSES, 
        D=cfg.DIMENSION, 
        channels=cfg.MINK_CHANNELS
    ).to(device)
    logging.info(f"Model:\n{model}")
    
    criterion = nn.BCEWithLogitsLoss()
    # Using Adam optimizer with a more standard learning rate
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
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
                'description': 'MinkowskiFCNN binary classifier (TorchScript)',
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
    parser = argparse.ArgumentParser(description="Train a sparse CNN classifier using Minkowski Engine.")
    args = parser.parse_args()
    cfg = Config()
    cfg.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    train_classifier(cfg, device)
    logging.info("Pipeline finished successfully!")

if __name__ == "__main__":
    main()