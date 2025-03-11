import argparse
import os
import sys
import time
from typing import Tuple, List, Optional, Dict
import glob
import uproot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.optim import Adam

cudnn.benchmark = True

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Contrastive Learning for Neutrino Events')
    parser.add_argument('--contrastive-epochs', default=5, type=int, help='Number of epochs for contrastive training')
    parser.add_argument('--batch', default=32, type=int, help='Batch size')
    parser.add_argument('--train-frac', default=0.7, type=float, help='Fraction of data for training')
    parser.add_argument('--workers', default=2, type=int, help='Data loading workers') 
    parser.add_argument('--data-dir', type=str, default='/gluster/data/dune/niclane/background', help='Path to ROOT files')
    parser.add_argument('--root-file', type=str, default=None, help='Specific ROOT file (overrides data-dir glob)')
    parser.add_argument('--img-size', type=int, default=512, help='Square image dimension in pixels')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for Adam')
    parser.add_argument('--resnet', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet variant')
    parser.add_argument('--ckpt-dir', type=str, default='/gluster/data/dune/niclane/checkpoints', help='Checkpoint save directory')
    parser.add_argument('--ckpt-freq', type=int, default=1, help='Checkpoint save frequency in epochs')
    parser.add_argument('--loss-file', type=str, default='/gluster/data/dune/niclane/checkpoints/losses.npz', help='File to save loss logs')
    return parser

def ifnone(val: Optional[any], default: any) -> any:
    return default if val is None else val

class ImageDataset(Dataset):
    def __init__(self, args: argparse.Namespace, file: str, indices: Optional[np.ndarray] = None):
        self.args = args
        self.file = file
        self.tree_name = "imageanalyser/ImageTree"
        self.img_size = args.img_size
        
        print(f"Opening ROOT file: {self.file}")
        start_time = time.time()
        self.root_file = uproot.open(self.file)
        self.tree = self.root_file[self.tree_name]
        self.num_events = self.tree.num_entries
        print(f"Found {self.num_events} events in {time.time() - start_time:.2f}s")
        
        self.chain = self.tree.arrays(["run", "subrun", "event", "type", "input"], library="np")
        print(f"Loaded arrays in {time.time() - start_time:.2f}s")
        
        self.pot = self._load_weights()
        self._load_metadata()
        
        self.indices = indices
        if self.indices is not None:
            self._filter_by_indices()
        
        self.weights = np.full(len(self.run_array), 1.0 / self.pot if self.pot > 0 else 1.0, dtype=np.float32)
        print(f"Dataset ready with {len(self.run_array)} events in {time.time() - start_time:.2f}s")

    def _load_weights(self) -> float:
        default_pot = 1.0
        print(f"Loading weights from {self.file}")
        try:
            sample_tree = self.root_file.get("imageanalyser/JobTree")
            if sample_tree is None:
                print(f"  Warning: 'JobTree' not found, using default pot={default_pot}")
                return default_pot
            pot = sample_tree.get("accumulated_pot", None)
            pot_value = float(pot.array(library="np")[0]) if pot and len(pot.array(library="np")) > 0 else default_pot
            print(f"  Loaded pot={pot_value}")
            return pot_value if pot_value > 0 else default_pot
        except Exception as e:
            print(f"  Error loading weights: {e}, using default pot={default_pot}")
            return default_pot

    def _load_metadata(self) -> None:
        self.run_array = self.chain["run"]
        self.subrun_array = self.chain["subrun"]
        self.event_array = self.chain["event"]
        self.type_array = self.chain["type"]
        self.event_to_file_idx = np.array([(0, i) for i in range(self.num_events)], 
                                         dtype=[('file_idx', 'i4'), ('event_idx', 'i4')])

    def _filter_by_indices(self) -> None:
        if self.indices is not None:
            self.run_array = self.run_array[self.indices]
            self.subrun_array = self.subrun_array[self.indices]
            self.event_array = self.event_array[self.indices]
            self.type_array = self.type_array[self.indices]
            self.chain = {k: v[self.indices] for k, v in self.chain.items()}
            self.event_to_file_idx = self.event_to_file_idx[self.indices]

    def __len__(self) -> int:
        return len(self.run_array)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, int, int, int]:
        run, subrun, event, evt_type = [self.__getattribute__(f"{k}_array")[idx] for k in ["run", "subrun", "event", "type"]]
        weight = self.weights[idx]
        input_data = self.chain["input"][idx]
        images = [np.fromiter(input_data[i], dtype=np.float32, count=self.img_size * self.img_size)
                 .reshape(self.img_size, self.img_size) for i in range(len(input_data))]
        return torch.tensor(np.stack(images), dtype=torch.float32), evt_type, weight, run, subrun, event

def split_dataset(full_ds: ImageDataset, train_frac: float) -> Tuple[Dataset, Dataset]:
    n_total = len(full_ds)
    n_train = int(n_total * train_frac)
    indices = np.random.permutation(n_total)
    train_idx, valid_idx = indices[:n_train], indices[n_train:]
    
    # Create lightweight views instead of reloading the file
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    valid_ds = torch.utils.data.Subset(full_ds, valid_idx)
    return train_ds, valid_ds

def create_dataloaders(train_ds: Dataset, valid_ds: Dataset, batch_size: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    full_ds = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
    train_sampler = WeightedRandomSampler(full_ds.weights[train_ds.indices], len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=workers, sampler=train_sampler, 
                          drop_last=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=workers, shuffle=False, 
                          drop_last=True, pin_memory=True)
    return train_dl, valid_dl

def get_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    workers = ifnone(args.workers, min(2, torch.get_num_threads()))
    file = args.root_file if args.root_file else glob.glob(os.path.join(args.data_dir, "*.root"))[0]
    print(f"Loading file: {file}")
    full_ds = ImageDataset(args, file)
    train_ds, valid_ds = split_dataset(full_ds, args.train_frac)
    return create_dataloaders(train_ds, valid_ds, args.batch, workers)

class ContrastiveResNet(nn.Module):
    def __init__(self, resnet_type: str = 'resnet18', num_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        resnet_map = {k: getattr(models, k) for k in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']}
        self.resnet = resnet_map[resnet_type](pretrained=True)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Linear(512, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.resnet(x))

def contrastive_loss(embeddings: List[torch.Tensor], temperature: float = 0.5) -> torch.Tensor:
    embeddings = [F.normalize(emb, dim=1) for emb in embeddings]
    batch_size = embeddings[0].size(0)
    total_loss = sum(F.cross_entropy(torch.matmul(embeddings[i], embeddings[j].T) / temperature, 
                                     torch.arange(batch_size).to(embeddings[0].device))
                     for i in range(3) for j in range(i + 1, 3)) / 3
    return total_loss

def train_epoch(model: nn.Module, train_dl: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss, total_samples = 0.0, 0
    epoch_start = time.time()
    print(f"Training epoch started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for batch_idx, batch in enumerate(train_dl):
        batch_start = time.time()
        images, labels, weights, _, _, _ = [x.to(device) for x in batch]
        bg_mask = (labels == 1)
        if bg_mask.sum() < 1:
            print(f"Batch {batch_idx}: Skipped (no background events)")
            continue
        
        images, weights = images[bg_mask], weights[bg_mask]
        optimizer.zero_grad()
        
        embeddings = [model(images[:, i:i+1, :, :]) for i in range(3)]
        loss = contrastive_loss(embeddings) * weights.mean()
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item() * images.size(0)
        total_loss += batch_loss
        total_samples += images.size(0)
        
        batch_time = time.time() - batch_start
        print(f"Batch {batch_idx}: Processed {images.size(0)} samples, Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Training epoch completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, Total Samples: {total_samples}")
    return avg_loss

def validate_epoch(model: nn.Module, valid_dl: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0
    epoch_start = time.time()
    print(f"Validation epoch started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            batch_start = time.time()
            images, labels, weights, _, _, _ = [x.to(device) for x in batch]
            bg_mask = (labels == 1)
            if bg_mask.sum() < 1:
                print(f"Batch {batch_idx}: Skipped (no background events)")
                continue
            
            images, weights = images[bg_mask], weights[bg_mask]
            embeddings = [model(images[:, i:i+1, :, :]) for i in range(3)]
            loss = contrastive_loss(embeddings) * weights.mean()
            
            batch_loss = loss.item() * images.size(0)
            total_loss += batch_loss
            total_samples += images.size(0)
            
            batch_time = time.time() - batch_start
            print(f"Batch {batch_idx}: Processed {images.size(0)} samples, Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Validation epoch completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, Total Samples: {total_samples}")
    return avg_loss

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace, filename: str) -> None:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args)}, 
               os.path.join(args.ckpt_dir, filename))
    print(f"Saved checkpoint: {filename}")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets...")
    data_start = time.time()
    train_dl, valid_ds = get_dataloaders(args)
    print(f"Dataset loading took {time.time() - data_start:.2f}s")
    print(f"Train: {len(train_dl.dataset)} events ({len(train_dl)} batches), Valid: {len(valid_ds.dataset)} events ({len(valid_ds)} batches)")

    print("Initializing model...")
    model = ContrastiveResNet(resnet_type=args.resnet, num_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(f"Model: {args.resnet} with {count_parameters(model):,} parameters")

    train_losses, valid_losses = [], []
    for epoch in range(args.contrastive_epochs):
        print(f"Epoch {epoch}/{args.contrastive_epochs-1}")
        train_loss = train_epoch(model, train_dl, optimizer, device)
        valid_loss = validate_epoch(model, valid_ds, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"  Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        if (epoch + 1) % args.ckpt_freq == 0:
            save_checkpoint(model, optimizer, epoch, args, f"ckpt_epoch_{epoch+1}.pth")

    save_checkpoint(model, optimizer, args.contrastive_epochs - 1, args, "final.pth")
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    np.savez(args.loss_file, train_losses=train_losses, valid_losses=valid_losses)
    print(f"Losses saved to {args.loss_file}")

if __name__ == '__main__':
    main()