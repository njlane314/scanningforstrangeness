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
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cudnn.benchmark = True

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Classification Training for Neutrino Events')
    parser.add_argument('--classifier-epochs', default=5, type=int, help='Number of epochs for classifier training')
    parser.add_argument('--batch', default=32, type=int, help='Batch size')
    parser.add_argument('--train-frac', default=0.7, type=float, help='Fraction of data for training')
    parser.add_argument('--val-frac', default=0.15, type=float, help='Fraction of data for validation')
    parser.add_argument('--workers', default=4, type=int, help='Data loading workers')
    parser.add_argument('--data-dir', type=str, default='/gluster/data/dune/niclane/background/', help='Path to ROOT files')
    parser.add_argument('--img-size', type=int, default=512, help='Square image dimension in pixels')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for Adam')
    parser.add_argument('--resnet', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet variant')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints', help='Checkpoint save directory')
    parser.add_argument('--ckpt-freq', type=int, default=1, help='Checkpoint save frequency in epochs')
    parser.add_argument('--loss-file', type=str, default='losses.npz', help='File to save loss logs')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pre-trained contrastive model checkpoint')
    return parser

def ifnone(val: Optional[any], default: any) -> any:
    return default if val is None else val

class ImageDataset(Dataset):
    def __init__(self, args: argparse.Namespace, files: List[str], indices: Optional[np.ndarray] = None):
        self.args = args
        self.files = sorted(files)
        if not self.files:
            sys.exit()
        
        self.tree_name = "imageanalyser/ImageTree"
        self.img_size = args.img_size
        
        self.chain = uproot.concatenate([f"{file}:{self.tree_name}" for file in self.files], 
                                      library="np")
        self.pot_dict = self._load_weights()
        self._load_metadata()
        if indices is not None:
            self._filter_by_indices(indices)
        
        self.num_events = len(self.run_array)
        self.weights = self._compute_weights()
    
    def _load_weights(self) -> Dict[str, float]:
        pot_dict = {}
        for file in self.files:
            with uproot.open(file) as f:
                sample_tree = f["imageanalyser/SampleTree"]
                pot = sample_tree["accumulated_pot"].array(library="np")
                pot_dict[os.path.basename(file)] = float(pot[0])
        return pot_dict
    
    def _load_metadata(self) -> None:
        self.run_array = self.chain["run"]
        self.subrun_array = self.chain["subrun"]
        self.event_array = self.chain["event"]
        self.type_array = self.chain["type"]
        
        self.event_to_file_idx = []
        offset = 0
        for i, file in enumerate(self.files):
            with uproot.open(file) as f:
                n_events = f[self.tree_name].num_entries
            self.event_to_file_idx.extend([(i, j) for j in range(n_events)])
            offset += n_events
        self.event_to_file_idx = np.array(self.event_to_file_idx, 
                                        dtype=[('file_idx', 'i4'), ('event_idx', 'i4')])
    
    def _filter_by_indices(self, indices: np.ndarray) -> None:
        for key in ["run", "subrun", "event", "type"]:
            setattr(self, f"{key}_array", getattr(self, f"{key}_array")[indices])
        self.event_to_file_idx = self.event_to_file_idx[indices]
        self.chain = {k: v[indices] for k, v in self.chain.items()}
    
    def _compute_weights(self) -> np.ndarray:
        weights = np.zeros(self.num_events, dtype=np.float32)
        for i, (file_idx, _) in enumerate(self.event_to_file_idx):
            fname = os.path.basename(self.files[file_idx])
            pot = self.pot_dict[fname]
            weights[i] = 1.0 / pot
        return weights
    
    def __len__(self) -> int:
        return self.num_events
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, int, int, int]:
        run, subrun, event, evt_type = [self.__getattribute__(f"{k}_array")[idx] 
                                        for k in ["run", "subrun", "event", "type"]]
        file_idx, event_idx = self.event_to_file_idx[idx]
        weight = self.weights[idx]
        
        input_data = self.chain["input"][idx]
        images = self._process_planes(input_data)
        images_tensor = torch.tensor(np.stack(images, axis=0), dtype=torch.float32)
        
        return images_tensor, evt_type, weight, run, subrun, event
    
    def _process_planes(self, input_data: np.ndarray) -> List[np.ndarray]:
        images = []
        for plane_idx in range(len(input_data)):
            plane = np.fromiter(input_data[plane_idx], dtype=np.float32, 
                              count=self.img_size * self.img_size).reshape(self.img_size, self.img_size)
            images.append(plane)
        return images

def split_dataset(full_ds: ImageDataset, train_frac: float, val_frac: float) -> Tuple[ImageDataset, ImageDataset, ImageDataset]:
    n_total = len(full_ds)
    n_train = int(n_total * train_frac)
    n_valid = int(n_total * val_frac)
    n_test = n_total - n_train - n_valid
    
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    
    return (ImageDataset(full_ds.args, full_ds.files, train_idx),
            ImageDataset(full_ds.args, full_ds.files, valid_idx),
            ImageDataset(full_ds.args, full_ds.files, test_idx))

def create_dataloaders(train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset, batch_size: int, workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    return train_dl, valid_dl, test_dl

def get_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    workers = ifnone(args.workers, min(4, torch.get_num_threads()))
    files = glob.glob(os.path.join(args.data_dir, "*.root"))
    full_ds = ImageDataset(args, files)
    
    train_ds, valid_ds, test_ds = split_dataset(full_ds, args.train_frac, args.val_frac)
    return create_dataloaders(train_ds, valid_ds, test_ds, args.batch, workers)

class ContrastiveResNet(nn.Module):
    def __init__(self, resnet_type: str = 'resnet18', num_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        resnet_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        resnet_fn = resnet_map[resnet_type]
        self.resnet = resnet_fn(pretrained=True)
        
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet(x)
        return self.projection(features)

class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(embedding_dim * 3, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = []
            for plane_idx in range(3):
                plane = x[:, plane_idx:plane_idx+1, :, :]
                emb = self.backbone(plane)
                embeddings.append(emb)
            features = torch.cat(embeddings, dim=1)
        return self.fc(features)

def train_classifier_epoch(model: nn.Module, train_dl: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in train_dl:
        images, labels, weights, _, _, _ = batch
        images, labels, weights = images.to(device), labels.to(device), weights.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item() * images.size(0)
        total_samples += images.size(0)
    
    return total_loss / total_samples

def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dl:
            images, labels, weights, _, _, _ = batch
            images, labels, weights = images.to(device), labels.to(device), weights.to(device)
            
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            weighted_loss = (loss * weights).mean()
            
            total_loss += weighted_loss.item() * images.size(0)
            total += images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / total, correct / total, all_preds, all_labels

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace, filename: str) -> None:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }
    torch.save(checkpoint, os.path.join(args.ckpt_dir, filename))
    print(f"Saved checkpoint: {filename} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, device: torch.device = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint: {checkpoint_path}")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_results(model: nn.Module, dl: DataLoader, device: torch.device, num_samples: int = 6):
    model.eval()
    images, labels, _, runs, subruns, events = next(iter(dl))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)
    
    images = images.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    runs = runs[:num_samples].numpy()
    subruns = subruns[:num_samples].numpy()
    events = events[:num_samples].numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(num_samples):
        img = images[i].transpose(1, 2, 0)
        axes[i].imshow(img[:, :, 0], cmap='gray')
        axes[i].set_title(f"Pred: {preds[i]} | True: {labels[i]}\nRun: {runs[i]} Subrun: {subruns[i]} Event: {events[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels: List[int], pred_labels: List[int]):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Signal', 'Background'], yticklabels=['Signal', 'Background'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    print(f"Starting classification training program...")
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading datasets...")
    data_start = time.time()
    train_dl, valid_dl, test_dl = get_dataloaders(args)
    data_time = time.time() - data_start
    
    print(f"Dataset loading took {data_time:.2f}s")
    print(f"Dataset sizes:")
    print(f"  Training:   {len(train_dl.dataset)} events ({len(train_dl)} batches)")
    print(f"  Validation: {len(valid_dl.dataset)} events ({len(valid_dl)} batches)")
    print(f"  Test:       {len(test_dl.dataset)} events ({len(test_dl)} batches)")
    
    print("Loading pre-trained contrastive model...")
    contrastive_model = ContrastiveResNet(resnet_type=args.resnet, num_channels=1).to(device)
    load_checkpoint(args.checkpoint, contrastive_model, device=device)
    
    print("Initializing classifier with frozen backbone...")
    classifier = Classifier(contrastive_model, embedding_dim=128, num_classes=2).to(device)
    optimizer = Adam(classifier.fc.parameters(), lr=args.lr)
    param_count = count_parameters(classifier)
    print(f"Classifier Model: {args.resnet} with {param_count:,} trainable parameters")
    
    train_losses = []
    valid_losses = []
    for epoch in range(args.classifier_epochs):
        print(f"Classifier Training - Epoch {epoch}/{args.classifier_epochs-1}")
        train_loss = train_classifier_epoch(classifier, train_dl, optimizer, device)
        valid_loss, valid_acc, _, _ = evaluate(classifier, valid_dl, device)
        test_loss, test_acc, test_preds, test_labels = evaluate(classifier, test_dl, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f} | Acc: {valid_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f} | Acc: {test_acc:.4f}")
        
        if (epoch + 1) % args.ckpt_freq == 0:
            checkpoint_name = f"ckpt_classifier_epoch_{epoch+1}.pth"
            save_checkpoint(classifier, optimizer, epoch, args, checkpoint_name)
    
    print("Saving final classifier model...")
    save_checkpoint(classifier, optimizer, args.classifier_epochs - 1, args, "final_classifier.pth")
    
    print("Displaying sample predictions...")
    show_results(classifier, test_dl, device, num_samples=6)
    
    print("Plotting confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    np.savez(args.loss_file, train_losses=train_losses, valid_losses=valid_losses)
    print(f"Losses saved to {args.loss_file}")

if __name__ == '__main__':
    main()