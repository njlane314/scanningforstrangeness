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
    raise ImportError("MinkowskiEngine not found.")

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
        if rows.size == 0: return None
        coords = np.stack([rows, cols], axis=1).astype(np.int32)
        # Features are just 1s to indicate occupancy for shape reconstruction
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
    dset = ImageDataset(phase, config=config)
    return torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        sampler=InfSampler(dset, phase == 'train') if config.train else None,
        shuffle=not config.train,
        pin_memory=False,
        drop_last=True
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
# ## --- MODEL ARCHITECTURE (Topology Generation) --- ##
# ==============================================================================

class Encoder(nn.Module):
    CHANNELS = [16, 32, 64, 128, 256]
    def __init__(self):
        super().__init__()
        ch = self.CHANNELS
        D = 2
        self.net = nn.Sequential(
            self.make_block(1, ch[0], D),
            self.make_block(ch[0], ch[1], D),
            self.make_block(ch[1], ch[2], D),
            self.make_block(ch[2], ch[3], D),
            self.make_block(ch[3], ch[4], D),
        )
        self.global_pool = ME.MinkowskiGlobalPooling()
        self.linear_mean = ME.MinkowskiLinear(ch[4], ch[4], bias=True)
        self.linear_log_var = ME.MinkowskiLinear(ch[4], ch[4], bias=True)

    def make_block(self, in_channels, out_channels, D):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
        )

    def forward(self, sinput):
        out = self.net(sinput)
        out = self.global_pool(out)
        return self.linear_mean(out), self.linear_log_var(out)

class Decoder(nn.Module):
    CHANNELS = [256, 128, 64, 32, 16]
    resolution = 32
    def __init__(self):
        super().__init__()
        ch = self.CHANNELS
        D = 2
        
        self.blocks = nn.ModuleList([
            self.make_block(ch[i], ch[i+1], D) for i in range(len(ch) - 1)
        ])
        
        self.final_cls = ME.MinkowskiConvolution(ch[-1], 1, kernel_size=1, bias=True, dimension=D)
        
    def make_block(self, in_channels, out_channels, D):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
        )

    def forward(self, z_glob, target_key):
        out = ME.SparseTensor(
            features=z_glob.F, coordinates=z_glob.C,
            tensor_stride=self.resolution, coordinate_manager=z_glob.coordinate_manager,
        )
        
        for block in self.blocks:
            out = block(out)
        
        out_cls = self.final_cls(out)
        
        if self.training:
            target = self.get_target(out_cls, target_key)
            return out_cls, target
        else:
            return out_cls, None

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

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, sinput, target_key):
        means, log_vars = self.encoder(sinput)
        if self.training:
            zs = ME.SparseTensor(
                features=means.F + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F),
                coordinate_manager=means.coordinate_manager, coordinate_map_key=means.coordinate_map_key
            )
        else:
            zs = means
            
        out_cls, target = self.decoder(zs, target_key)
        return out_cls, target, means, log_vars

# ==============================================================================
# ## --- MAIN SCRIPT LOGIC (BCE Loss) --- ##
# ==============================================================================

def train(net, dataloader, device, config):
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    crit = nn.BCEWithLogitsLoss()
    net.train()

    train_iter = iter(dataloader)
    for i in range(config.max_iter):
        try:
            data_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            data_dict = next(train_iter)
        if data_dict is None: continue

        optimizer.zero_grad()
        sin = ME.SparseTensor(
            features=data_dict["feats"].to(device), coordinates=data_dict["coords"].to(device),
        )

        out_cls, target, means, log_vars = net(sin, sin.coordinate_map_key)
        
        bce_loss = crit(out_cls.F.squeeze(), target.float())
        kld_loss = -0.5 * torch.mean(torch.sum(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1))
        loss = bce_loss + kld_loss * config.kld_weight

        loss.backward()
        optimizer.step()

        if i % config.stat_freq == 0:
            logging.info(
                f"Iter: {i}, Loss: {loss.item():.4f}, BCE: {bce_loss.item():.4f}, "
                f"KLD: {kld_loss.item():.4f}"
            )

        if i > 0 and i % config.val_freq == 0:
            logging.info(f"Saving weights to {config.weights}")
            torch.save({'state_dict': net.state_dict()}, config.weights)
            scheduler.step()

def visualize(net, dataloader, device, config):
    net.eval()
    for b_idx, data_dict in enumerate(dataloader):
        if data_dict is None: continue
        if b_idx * config.batch_size >= config.max_visualization: break

        sin = ME.SparseTensor(
            features=data_dict["feats"].to(device), coordinates=data_dict["coords"].to(device),
        )
        
        with torch.no_grad():
            out_cls, _, _, _ = net(sin, sin.coordinate_map_key)
        
        s_probs = torch.sigmoid(out_cls.F)
        pruning_mask = s_probs.squeeze() > 0.5
        
        s_final_shape = ME.MinkowskiPruning()(out_cls, pruning_mask) if pruning_mask.sum() > 0 else out_cls

        input_coords, _ = sin.decomposed_coordinates_and_features
        output_coords, _ = s_final_shape.decomposed_coordinates_and_features
        
        for i in range(len(input_coords)):
            if b_idx * config.batch_size + i >= config.max_visualization: break
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            in_c = input_coords[i].cpu().numpy()
            in_img = np.zeros((512, 512)); in_img[in_c[:, 0], in_c[:, 1]] = 1
            axes[0].imshow(in_img, cmap='gray_r', origin='lower')
            axes[0].set_title(f'Input Event (Index {b_idx * config.batch_size + i})')

            out_img = np.zeros((512, 512))
            if i < len(output_coords):
                out_c = output_coords[i].cpu().numpy()
                if out_c.size > 0: out_img[out_c[:, 0], out_c[:, 1]] = 1
            axes[1].imshow(out_img, cmap='gray_r', origin='lower')
            axes[1].set_title('Generated Output (Prob > 0.5)')
            
            plt.tight_layout()
            plt.savefig(f"reconstruction_event_{b_idx * config.batch_size + i}.png")
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_file_path", type=str)
    parser.add_argument("--tree_names", nargs='+', type=str)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--val_freq", type=int, default=500)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--kld_weight", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stat_freq", type=int, default=50)
    parser.add_argument("--weights", type=str, default="neutrino_vae_topo.pth")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--max_visualization", type=int, default=4)
    config = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = VAE()
    net.to(device)

    if config.train:
        dataloader = make_data_loader("train", config)
        train(net, dataloader, device, config)

    if config.test:
        if not os.path.exists(config.weights):
            raise FileNotFoundError(f"Weights file not found: {config.weights}")
        logging.info(f"Loading weights from {config.weights}")
        checkpoint = torch.load(config.weights)
        net.load_state_dict(checkpoint["state_dict"])
        dataloader = make_data_loader("test", config)
        visualize(net, dataloader, device, config)