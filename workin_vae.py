# main_script_topology.py

import os
import sys
import argparse
import logging
import numpy as np
from time import time
import uproot

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    import MinkowskiEngine as ME
except ImportError:
    raise ImportError(
        "MinkowskiEngine not found. Please install it, e.g., "
        "by following the instructions at https://github.com/NVIDIA/MinkowskiEngine"
    )

# ==============================================================================
# ## --- DATA LOADING (For Topology) --- ##
# ==============================================================================

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, phase, config):
        self.phase = phase
        self.root_file_path = config.root_file_path
        self.tree_names = config.tree_names
        self.file_handle = None
        self.events = []
        self._scan_root_file()

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self):
        logging.info(f"Scanning ROOT file: {self.root_file_path}")
        with uproot.open(self.root_file_path) as file:
            for tree_name in self.tree_names:
                if tree_name not in file:
                    logging.warning(f"TTree '{tree_name}' not found.")
                    continue
                num_entries = file[tree_name].num_entries
                logging.info(f"Found {num_entries} events in TTree '{tree_name}'.")
                for i in range(num_entries):
                    self.events.append((tree_name, i))
        logging.info(f"Loaded {len(self.events)} event indices for phase '{self.phase}'.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        flattened_image = tree['detector_image_w'].array(entry_start=entry_index, entry_stop=entry_index + 1, library="np")[0]
        rows, cols = np.where(flattened_image.reshape(512, 512) > 0)
        if rows.size < 10: return None # Skip nearly empty events
        coords = np.stack([rows, cols], axis=1).astype(np.int32)
        feats = np.ones((len(coords), 1), dtype=np.float32)
        return (coords, feats)

def collate_fn(list_data):
    list_data = [data for data in list_data if data is not None]
    if not list_data: return None
    coords, feats = list(zip(*list_data))
    return {
        "coords": ME.utils.batched_coordinates(coords),
        "feats": torch.from_numpy(np.vstack(feats)),
    }

def make_data_loader(phase, config):
    is_train = phase == 'train'
    dset = ImageDataset(phase, config=config)
    return torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        sampler=InfSampler(dset, shuffle=True) if is_train and config.train_mode else None,
        shuffle=not (is_train and config.train_mode),
        pin_memory=True,
        drop_last=is_train
    )

class InfSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        self._perm = torch.randperm(perm).tolist() if self.shuffle else list(range(perm))

    def __iter__(self): return self
    def __next__(self):
        if not self._perm: self.reset_permutation()
        return self._perm.pop(0)
    def __len__(self): return len(self.data_source)

# ==============================================================================
# ## --- MODEL ARCHITECTURE (Standard Autoencoder) --- ##
# ==============================================================================

class Encoder(nn.Module):
    CHANNELS = [1, 16, 32, 64, 128, 256]
    def __init__(self, D=2):
        super().__init__()
        ch = self.CHANNELS
        self.net = nn.Sequential(
            self.make_block(ch[0], ch[1], D),
            self.make_block(ch[1], ch[2], D),
            self.make_block(ch[2], ch[3], D),
            self.make_block(ch[3], ch[4], D),
            self.make_block(ch[4], ch[5], D),
        )
        self.final_conv = ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=1, dimension=D)

    def make_block(self, in_channels, out_channels, D):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
        )

    def forward(self, x):
        x = self.net(x)
        return self.final_conv(x)


class Decoder(nn.Module):
    CHANNELS = [256, 128, 64, 32, 16, 8]
    def __init__(self, D=2):
        super().__init__()
        ch = self.CHANNELS
        self.blocks = nn.ModuleList([self.make_block(ch[i], ch[i+1], D) for i in range(len(ch) - 1)])
        self.cls_layers = nn.ModuleList([ME.MinkowskiConvolution(c, 1, kernel_size=1, bias=True, dimension=D) for c in ch])
        self.pruning = ME.MinkowskiPruning()

    def make_block(self, in_channels, out_channels, D):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
        )

    @torch.no_grad()
    def get_target(self, out, target_key):
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(
            out.coordinate_map_key, strided_target_key, kernel_size=1, region_type=ME.RegionType.HYPER_CUBE
        )
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def forward(self, z, target_key):
        out_cls, targets = [], []
        out = z
        
        # Initial classification and pruning
        out_cls.append(self.cls_layers[0](out))
        if self.training:
            target = self.get_target(out, target_key)
            targets.append(target)
            keep = target  # Use ground truth target for pruning
        else:
            keep = (out_cls[-1].F.squeeze() > 0)  # Use predictions during inference
        if self.pruning is not None and keep.sum() > 0:
            out = self.pruning(out, keep)

        # Upsampling blocks
        for i, block in enumerate(self.blocks):
            out = block(out)
            out_cls.append(self.cls_layers[i+1](out))
            if self.training:
                target = self.get_target(out, target_key)
                targets.append(target)
                keep = target  # Use ground truth target for pruning
            else:
                keep = (out_cls[-1].F.squeeze() > 0)  # Use predictions during inference
            if self.pruning is not None and keep.sum() > 0:
                out = self.pruning(out, keep)

        return out_cls, targets

def weights_init(m):
    if isinstance(m, ME.MinkowskiConvolution):
        ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
    if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

class Autoencoder(nn.Module):
    def __init__(self, D=2):
        super().__init__()
        self.encoder = Encoder(D=D)
        self.decoder = Decoder(D=D)
        self.apply(weights_init)

    def forward(self, sinput):
        target_key = sinput.coordinate_map_key
        z = self.encoder(sinput)
        out_cls, targets = self.decoder(z, target_key)
        return out_cls, targets

# ==============================================================================
# ## --- MAIN SCRIPT LOGIC --- ##
# ==============================================================================

def train(net, dataloader, device, config):
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    crit = nn.BCEWithLogitsLoss()
    net.train()

    train_iter = iter(dataloader)
    logging.info("Starting Autoencoder training...")
    strides = [32, 16, 8, 4, 2, 1]
    for i in range(config.max_iter):
        try:
            data_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            data_dict = next(train_iter)
        if data_dict is None: continue

        optimizer.zero_grad()
        sin = ME.SparseTensor(
            features=data_dict["feats"].to(device), coordinates=data_dict["coords"].to(device)
        )

        out_cls, targets = net(sin)
        
        bce_loss = 0
        for out, target in zip(out_cls, targets):
            bce_loss += crit(out.F.squeeze(), target.float())
        bce_loss /= len(out_cls)
        
        loss = bce_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        if i % config.stat_freq == 0:
            logging.info(f"Iter: {i}, Loss: {loss.item():.4f}")

        if i > 0 and i % config.val_freq == 0:
            logging.info(f"Saving weights to {config.weights}")
            torch.save({'state_dict': net.state_dict()}, config.weights)
            scheduler.step()


def visualize(net, dataloader, device, config):
    net.eval()
    output_dir = "visualizations_diagnostic"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving diagnostic visualizations to '{output_dir}/'")
    
    with torch.no_grad():
        for i, data_dict in enumerate(dataloader):
            if i >= config.max_visualization: break
            if data_dict is None: continue

            sin = ME.SparseTensor(
                features=data_dict["feats"].to(device), coordinates=data_dict["coords"].to(device)
            )
            out_cls, _ = net(sin)
            final_out = out_cls[-1]

            # Prepare ground truth input for visualization
            input_coords = sin.C.cpu().numpy()
            input_batch_indices = input_coords[:, 0]
            
            # Prepare model output for visualization
            if final_out.F.numel() > 0:
                out_coords = final_out.C.cpu().numpy()
                out_batch_indices = out_coords[:, 0]
                out_probs = torch.sigmoid(final_out.F).squeeze().cpu().numpy()
            else:
                logging.warning(f"Visualization {i+1}: Decoder produced no output.")
                out_coords, out_batch_indices, out_probs = [], [], []

            # Visualize the first item in the batch
            batch_idx = input_batch_indices[0]
            
            # 1. Ground Truth
            gt_map = np.zeros((512, 512))
            gt_mask = input_batch_indices == batch_idx
            gt_coords_b = input_coords[gt_mask, 1:]
            gt_map[gt_coords_b[:, 0], gt_coords_b[:, 1]] = 1

            # 2. Heatmap
            heatmap = np.zeros((512, 512))
            if len(out_probs) > 0:
                out_mask = out_batch_indices == batch_idx
                out_coords_b = out_coords[out_mask, 1:]
                out_probs_b = out_probs[out_mask]
                if out_coords_b.shape[0] > 0:
                    heatmap[out_coords_b[:, 0], out_coords_b[:, 1]] = out_probs_b

            # 3. Create diagnostic plot
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            axes[0].imshow(gt_map, cmap='gray')
            axes[0].set_title(f"Input #{i+1}")

            im = axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title("Generated Heatmap")
            
            # Overlay
            overlay_map = np.stack([heatmap, gt_map, np.zeros_like(gt_map)], axis=-1)
            axes[2].imshow(overlay_map)
            axes[2].set_title("Overlay (R: Gen, G: GT)")

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
            save_path = os.path.join(output_dir, f"diagnostic_{i+1}.png")
            plt.savefig(save_path)
            plt.close(fig)
            logging.info(f"Saved diagnostic plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data and Model Config
    parser.add_argument("--root_file_path", type=str, required=True)
    parser.add_argument("--tree_names", nargs='+', type=str, required=True)
    parser.add_argument("--weights", type=str, default="neutrino_ae_final.pth")
    
    # Training Config
    parser.add_argument("--max_iter", type=int, default=20000)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    
    # Logging and Validation
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--stat_freq", type=int, default=100)
    
    # Execution Config
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train", dest='train_mode', action='store_true')
    parser.add_argument("--test", dest='train_mode', action='store_false')
    parser.set_defaults(train_mode=True)
    parser.add_argument("--max_visualization", type=int, default=8)
    config = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = Autoencoder(D=2)
    net.to(device)

    if config.train_mode:
        dataloader = make_data_loader("train", config)
        train(net, dataloader, device, config)
    else: # Test mode
        if not os.path.exists(config.weights):
            raise FileNotFoundError(f"Weights file not found: {config.weights}")
        logging.info(f"Loading weights from {config.weights}")
        checkpoint = torch.load(config.weights)
        net.load_state_dict(checkpoint["state_dict"])
        
        dataloader = make_data_loader("test", config)
        visualize(net, dataloader, device, config)