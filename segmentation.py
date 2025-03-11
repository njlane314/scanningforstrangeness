import argparse
import os
import sys
import time
import glob
import uproot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import Adam

cudnn.benchmark = True

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

class Sigmoid(nn.Module):
    def __init__(self, out_range=None):
        super(Sigmoid, self).__init__()
        if out_range is not None:
            self.low, self.high = out_range
            self.range = self.high - self.low
        else:
            self.low = None
            self.high = None
            self.range = None

    def forward(self, x):
        if self.low is not None:
            return torch.sigmoid(x) * (self.range) + self.low
        else:
            return torch.sigmoid(x)

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

        self.output = nn.Sequential(nn.Conv2d(n_filters, n_classes, 1), Sigmoid(y_range))

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

        return self.output(res)

class ImageDataset(Dataset):
    def __init__(self, args: argparse.Namespace, file: str, indices: Optional[np.ndarray] = None):
        self.args = args
        self.file = file
        self.tree_name = "imageanalyser/ImageTree"
        self.img_size = args.img_size
        
        print(f"Opening ROOT file: {self.file}")
        start_time = time.time()
        with uproot.open(self.file) as f:
            tree = f[self.tree_name]
            self.num_events = tree.num_entries
            print(f"Found {self.num_events} events in {time.time() - start_time:.2f}s")
            self.data = tree.arrays(["input", "mask"], library="np")
        print(f"Data loaded in {time.time() - start_time:.2f}s")
        
        self.indices = indices
        if self.indices is not None:
            self.data = {k: v[self.indices] for k, v in self.data.items()}
        
        self.num_events = len(self.data["input"])
        print(f"Dataset ready with {self.num_events} events")

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_data = self.data["input"][idx]
        mask_data = self.data["mask"][idx]
        
        images = [np.fromiter(input_data[i], dtype=np.float32, count=self.img_size * self.img_size)
                 .reshape(self.img_size, self.img_size) for i in range(len(input_data))]
        images_tensor = torch.tensor(np.stack(images), dtype=torch.float32)
        
        mask = np.fromiter(mask_data, dtype=np.float32, count=self.img_size * self.img_size)
        mask_tensor = torch.tensor(mask.reshape(self.img_size, self.img_size), dtype=torch.float32)
        
        return images_tensor, mask_tensor

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='UNet Segmentation for Neutrino Events')
    parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch', default=8, type=int, help='Batch size')
    parser.add_argument('--train-frac', default=0.8, type=float, help='Fraction of data for training')
    parser.add_argument('--workers', default=2, type=int, help='Data loading workers')
    parser.add_argument('--data-dir', type=str, default='/gluster/data/dune/niclane/background/', help='Path to ROOT files')
    parser.add_argument('--root-file', type=str, default=None, help='Specific ROOT file')
    parser.add_argument('--img-size', type=int, default=512, help='Image dimension in pixels')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for Adam')
    parser.add_argument('--n-classes', default=1, type=int, help='Number of segmentation classes')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints_seg', help='Checkpoint save directory')
    parser.add_argument('--ckpt-freq', type=int, default=2, help='Checkpoint save frequency in epochs')
    parser.add_argument('--loss-file', type=str, default='seg_losses.npz', help='File to save loss logs')
    return parser

def ifnone(val: Optional[any], default: any) -> any:
    return default if val is None else val

def split_dataset(full_ds: ImageDataset, train_frac: float) -> Tuple[Dataset, Dataset]:
    n_total = len(full_ds)
    n_train = int(n_total * train_frac)
    indices = np.random.permutation(n_total)
    train_ds = torch.utils.data.Subset(full_ds, indices[:n_train])
    valid_ds = torch.utils.data.Subset(full_ds, indices[n_train:])
    return train_ds, valid_ds

def get_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    workers = ifnone(args.workers, min(2, torch.get_num_threads()))
    file = args.root_file if args.root_file else glob.glob(os.path.join(args.data_dir, "*.root"))[0]
    print(f"Loading file: {file}")
    full_ds = ImageDataset(args, file)
    train_ds, valid_ds = split_dataset(full_ds, args.train_frac)
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=workers, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, num_workers=workers, shuffle=False, pin_memory=True)
    return train_dl, valid_dl

def train_epoch(model: nn.Module, train_dl: DataLoader, optimizer: torch.optim.Optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss, total_samples = 0.0, 0
    epoch_start = time.time()
    print(f"Training epoch started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for batch_idx, (images, masks) in enumerate(train_dl):
        batch_start = time.time()
        images, masks = images.to(device), masks.to(device).unsqueeze(1)  
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
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

def validate_epoch(model: nn.Module, valid_dl: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0
    epoch_start = time.time()
    print(f"Validation epoch started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(valid_dl):
            batch_start = time.time()
            images, masks = images.to(device), masks.to(device).unsqueeze(1) 
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            batch_loss = loss.item() * images.size(0)
            total_loss += batch_loss
            total_samples += images.size(0)
            
            batch_time = time.time() - batch_start
            print(f"Batch {batch_idx}: Processed {images.size(0)} samples, Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Validation epoch completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, Total Samples: {total_samples}")
    return avg_loss

def main():
    print(f"Starting segmentation training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets...")
    data_start = time.time()
    train_dl, valid_dl = get_dataloaders(args)
    print(f"Dataset loading took {time.time() - data_start:.2f}s")
    print(f"Train: {len(train_dl.dataset)} events ({len(train_dl)} batches), Valid: {len(valid_dl.dataset)} events ({len(valid_dl)} batches)")

    print("Initializing UNet model...")
    model = UNet(in_dim=3, n_classes=args.n_classes, depth=4, n_filters=16, drop_prob=0.1).to(device)  
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss() if args.n_classes == 1 else nn.CrossEntropyLoss()  
    
    train_losses, valid_losses = [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}/{args.epochs-1}")
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        valid_loss = validate_epoch(model, valid_dl, criterion, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        if (epoch + 1) % args.ckpt_freq == 0:
            checkpoint_name = f"ckpt_seg_epoch_{epoch+1}.pth"
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args)}, 
                       os.path.join(args.ckpt_dir, checkpoint_name))
            print(f"Saved checkpoint: {checkpoint_name}")

    torch.save({'epoch': args.epochs - 1, 'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args)}, 
               os.path.join(args.ckpt_dir, "final_seg.pth"))
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    np.savez(args.loss_file, train_losses=train_losses, valid_losses=valid_losses)
    print(f"Losses saved to {args.loss_file}")

if __name__ == '__main__':
    main()