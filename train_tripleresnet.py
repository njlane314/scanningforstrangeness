import os
import sys
import logging
import argparse
import random
import numpy as np
import uproot
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import sklearn.metrics as metrics
import MinkowskiEngine as ME

def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

class CustomBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, dimension=-1):
        super(CustomBasicBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes)
        self.relu = ME.MinkowskiELU()
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        super(ResNetBase, self).__init__()
        self.D = D
        assert self.BLOCK is not None
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(self.inplanes),
            ME.MinkowskiELU(),
        )
        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)
        self.pooling = ME.MinkowskiGlobalAvgPooling()
        self.final = nn.Linear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, dimension=self.D),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, dimension=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))
        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling(x)
        return self.final(x.F)

class MinkowskiResNet18(ResNetBase):
    BLOCK = CustomBasicBlock
    LAYERS = (2, 2, 2, 2)

class NeutrinoDataset(Dataset):
    def __init__(self, root_file_path, tree_names, image_branches, label_branch, min_hits=10, fiducial_cut=None):
        self.root_file_path = root_file_path
        self.image_branches = image_branches
        self.label_branch = label_branch
        self.min_hits = min_hits
        self.fiducial_cut = fiducial_cut
        self.file_handle = None
        self.num_classes = 2
        self._scan_root_file(tree_names, label_branch)

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _is_in_fiducial_volume(self, coords):
        if self.fiducial_cut is None:
            return True
        # Assuming coords are (plane, u, v) or similar, and fiducial cut is on u,v
        # This is a simplified example; a real FV cut would be more complex
        u_coords = coords[:, 1]
        v_coords = coords[:, 2]
        in_fv = (u_coords > self.fiducial_cut) & (u_coords < (512 - self.fiducial_cut)) & \
                (v_coords > self.fiducial_cut) & (v_coords < (512 - self.fiducial_cut))
        return in_fv.any()

    def _scan_root_file(self, tree_names, label_branch):
        self.events = []
        all_labels = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file:
                    continue
                tree = file[tree_name]
                num_entries = tree.num_entries
                
                # Load all necessary branches for pre-filtering
                labels_array = tree[label_branch].array(library="np")
                
                for i in range(num_entries):
                    total_hits = 0
                    for branch_name in self.image_branches:
                        total_hits += len(tree[branch_name].array(entry_start=i, entry_stop=i + 1)[0])
                    
                    if total_hits < self.min_hits:
                        continue

                    self.events.append((tree_name, i))
                    all_labels.append(labels_array[i])

        if not self.events:
            raise RuntimeError("No events loaded after applying cuts.")
        
        raw_labels = np.array(all_labels)
        self.labels = (raw_labels.astype(int) == 20).astype(int).flatten()

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        clean_label = self.labels[idx]
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        plane_data = []
        for branch_name in self.image_branches:
            flattened_image = tree[branch_name].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
            image_grid = np.array(flattened_image, dtype=np.float32).reshape(512, 512)
            rows, cols = np.where(image_grid > 1.0)
            if rows.size == 0:
                coords = torch.empty((0, 2), dtype=torch.int32)
                feats = torch.empty((0, 1), dtype=torch.float32)
            else:
                coords = torch.from_numpy(np.stack([rows, cols], axis=1)).int()
                # Simple feature scaling
                feats = torch.from_numpy(np.log10(image_grid[rows, cols] + 1e-6)[:, np.newaxis]).float()
                feats = (feats - feats.min()) / (feats.max() - feats.min() + 1e-8)
            plane_data.append((coords, feats))
        return plane_data, torch.tensor([clean_label], dtype=torch.long)

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.labels = dataset.labels
        self.signal_indices = np.where(self.labels == 1)[0]
        self.background_indices = np.where(self.labels == 0)[0]
        
        if len(self.signal_indices) == 0 or len(self.background_indices) == 0:
            raise ValueError("Not enough data for both signal and background classes after filtering.")

        self.num_batches = min(len(self.signal_indices), len(self.background_indices)) // self.half_batch

    def __iter__(self):
        np.random.shuffle(self.signal_indices)
        np.random.shuffle(self.background_indices)
        for i in range(self.num_batches):
            start_idx = i * self.half_batch
            end_idx = start_idx + self.half_batch
            signal_batch = self.signal_indices[start_idx:end_idx]
            background_batch = self.background_indices[start_idx:end_idx]
            batch = np.concatenate([signal_batch, background_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches

def collate_fn_fused(batch):
    filtered_batch = [item for item in batch if item is not None and all(c.shape[0] > 0 for c, f in item[0])]
    if not filtered_batch:
        return None
    
    all_coords, all_feats, all_labels = [], [], []
    for (plane_data, label) in filtered_batch:
        event_coords, event_feats = [], []
        for plane_id, (coords, feats) in enumerate(plane_data):
            plane_coords = torch.cat([
                torch.full((len(coords), 1), plane_id, dtype=torch.int32),
                coords
            ], dim=1)
            event_coords.append(plane_coords)
            event_feats.append(feats)
        
        all_coords.append(torch.cat(event_coords, dim=0))
        all_feats.append(torch.cat(event_feats, dim=0))
        all_labels.append(label)

    if not all_coords:
        return None

    coords_batch, feats_batch = ME.utils.sparse_collate(all_coords, all_feats)
    labels_batch = torch.cat(all_labels)

    return coords_batch, feats_batch, labels_batch

def make_data_loader(dataset, config):
    batch_sampler = BalancedBatchSampler(dataset, config.batch_size)
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn_fused,
        pin_memory=True,
    )

def criterion(pred, labels):
    return F.cross_entropy(pred, labels.contiguous().view(-1), reduction="mean")

def train(net, train_loader, device, config):
    optimiser = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    net.train()
    running_loss = 0.0
    running_accuracy = 0.0
    logging.info(f"--- Starting Training for {config.max_steps} steps ---")
    train_iter = iter(train_loader)
    for i in range(config.max_steps):
        optimiser.zero_grad()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if batch is None:
            continue
        
        coords, feats, labels = batch
        
        if labels.numel() == 0:
            continue

        input_tensor = ME.SparseTensor(features=feats, coordinates=coords, device=device)
        
        logit = net(input_tensor)
        
        loss = criterion(logit, labels.to(device))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        
        optimiser.step()

        with torch.no_grad():
            pred = torch.argmax(logit, 1)
            accuracy = metrics.accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
        
        running_loss += loss.item()
        running_accuracy += accuracy
        
        if (i + 1) % config.stat_freq == 0:
            avg_loss = running_loss / (i + 1)
            avg_accuracy = running_accuracy / (i + 1)
            logging.info(f"Iter: {i + 1}, Loss: {loss.item():.3e}, Batch Acc: {accuracy:.4f}, Avg Loss: {avg_loss:.3e}, Avg Acc: {avg_accuracy:.4f}")

    logging.info("--- Training Complete ---")
    torch.save(net.state_dict(), config.weights)
    logging.info(f"Final model saved to {config.weights}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description="Train a Fused-Input Minkowski ResNet for Neutrino Event Classification.")
    parser.add_argument("--root_file_path", type=str, required=True)
    parser.add_argument("--image_branches", nargs=3, default=["detector_image_u", "detector_image_v", "detector_image_w"])
    parser.add_argument("--label_branch", type=str, default="inclusive_strange_channels")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stat_freq", type=int, default=1)
    parser.add_argument("--weights", type=str, default="neutrino_fused_resnet18_model.pth")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min_hits", type=int, default=50, help="Minimum number of hits for an event to be included.")
    parser.add_argument("--fiducial_cut", type=int, default=10, help="Pixel border to exclude for fiducial volume.")
    config = parser.parse_args()
    
    tree_names = ['mc_inclusive_run1_fhc']
    
    logging.info(f"Running with configuration: {config}")
    logging.info(f"Using tree names: {tree_names}")
    
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        full_dataset = NeutrinoDataset(
            root_file_path=config.root_file_path,
            tree_names=tree_names,
            image_branches=config.image_branches,
            label_branch=config.label_branch,
            min_hits=config.min_hits,
            fiducial_cut=config.fiducial_cut
        )
        
        net = MinkowskiResNet18(in_channels=1, out_channels=full_dataset.num_classes, D=3).to(device)
        
        logging.info(f"Model created successfully.")
        train_loader = make_data_loader(full_dataset, config)
        train(net, train_loader, device, config)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)