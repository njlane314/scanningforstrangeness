import argparse
import sklearn.metrics as metrics
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, Sampler
import torch.optim as optim

import MinkowskiEngine as ME
import uproot

class SparseInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dimension=2):
        super().__init__()
        out1 = out_channels // 2
        reduce3 = out_channels // 4
        out3 = out_channels // 4
        reduce5 = out_channels // 8
        out5 = out_channels // 8
        out_pool = out_channels // 8
        self.branch1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out1, kernel_size=1, stride=stride, dimension=dimension
            ),
            ME.MinkowskiReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, reduce3, kernel_size=1, dimension=dimension),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(reduce3, out3, kernel_size=3, stride=stride, dimension=dimension),
            ME.MinkowskiReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, reduce5, kernel_size=1, dimension=dimension),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(reduce5, out5, kernel_size=5, stride=stride, dimension=dimension),
            ME.MinkowskiReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            ME.MinkowskiMaxPooling(kernel_size=3, stride=stride, dimension=dimension),
            ME.MinkowskiConvolution(in_channels, out_pool, kernel_size=1, dimension=dimension),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x: ME.SparseTensor):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        concatenated = self._sparse_cat([out1, out2, out3, out4])
        return concatenated

    def _sparse_cat(self, tensors: List[ME.SparseTensor]) -> ME.SparseTensor:
        if not tensors:
            raise ValueError("No tensors to concatenate")
        
        channel_sizes = [t.F.shape[1] for t in tensors]
        total_channels = sum(channel_sizes)
        
        all_coordinates = torch.cat([t.C for t in tensors], dim=0)
        unique_coordinates, inverse_indices = torch.unique(all_coordinates, dim=0, return_inverse=True)
        
        new_features = torch.zeros(
            (len(unique_coordinates), total_channels),
            dtype=tensors[0].F.dtype,
            device=tensors[0].F.device
        )
        
        start_idx = 0
        ch_start = 0
        for i, t in enumerate(tensors):
            num_points = len(t.F)
            inv_idx = inverse_indices[start_idx : start_idx + num_points]
            new_features[inv_idx, ch_start : ch_start + channel_sizes[i]] = t.F
            start_idx += num_points
            ch_start += channel_sizes[i]
        
        return ME.SparseTensor(
            features=new_features,
            coordinates=unique_coordinates,
            device=tensors[0].device
        )

class MinkowskiDUNECVN(nn.Module):
    def __init__(self,
                 initial_conv_filters=64,
                 depth=[2, 2, 3, 2],
                 filters=[64, 128, 256, 512],
                 output_names=['is_strange_signal'],
                 output_neurons=[1],
                 dimension=2):
        super().__init__()
        self.output_names = output_names
        self.branches = nn.ModuleList([
            nn.Sequential(
                ME.MinkowskiConvolution(1, initial_conv_filters, kernel_size=7, stride=2, dimension=dimension),
                ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=dimension)
            ) for _ in range(3)
        ])
        self.stage1_blocks = nn.ModuleList([
             self._make_inception_layer(initial_conv_filters, filters[0], depth[0], stride=1, dimension=dimension)
             for _ in range(3)
        ])
        self.main_trunk = nn.Sequential(
            self._make_inception_layer(filters[0], filters[1], depth[1], stride=2, dimension=dimension),
            self._make_inception_layer(filters[1], filters[2], depth[2], stride=2, dimension=dimension),
            self._make_inception_layer(filters[2], filters[3], depth[3], stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(filters[3]),
            ME.MinkowskiReLU(inplace=True)
        )
        self.glob_pool = ME.MinkowskiGlobalPooling()
        self.output_heads = nn.ModuleDict()
        for name, num_neurons in zip(output_names, output_neurons):
            self.output_heads[name] = ME.MinkowskiLinear(filters[3] * 3, num_neurons)

    def _make_inception_layer(self, in_channels, out_channels, num_blocks, stride, dimension):
        layers = []
        current_in = in_channels
        s = stride
        for i in range(num_blocks):
            layers.append(SparseInceptionBlock(current_in, out_channels, stride=s, dimension=dimension))
            current_in = out_channels
            s = 1  # Only apply stride to the first block
        return nn.Sequential(*layers)

    def forward(self, inputs: List[ME.SparseTensor]):
        branch_outputs = [self.stage1_blocks[i](self.branches[i](x)) for i, x in enumerate(inputs)]
        
        pooled_features_list = []
        for bo in branch_outputs:
            x = self.main_trunk(bo)
            pooled = self.glob_pool(x)
            pooled_features_list.append(pooled.F)
        
        cat_features = torch.cat(pooled_features_list, dim=1)
        pooled_st = ME.SparseTensor(
            features=cat_features,
            coordinate_manager=pooled_features_list[0].coordinate_manager,
            coordinate_map_key=pooled_features_list[0].coordinate_map_key,
        )
        outputs = {
            name: head(pooled_st).F for name, head in self.output_heads.items()
        }
        return outputs

class NeutrinoDataset(torch.utils.data.Dataset):
    def __init__(self, root_file_path, tree_names, is_train=False):
        self.root_file_path = root_file_path
        self.image_branches = [f'detector_image_{p}' for p in ['u', 'v', 'w']]
        self.weight_branch = 'base_event_weight'
        self.is_train = is_train
        self.file_handle = None
        self._scan_root_file(tree_names.split(','))
        # Compute scale factor for signal boosting
        self.scale_factor = 1.0
        if np.sum(self.labels == 1) > 0 and np.sum(self.labels == 0) > 0:
            mean_sig = np.mean(self.weights[self.labels == 1])
            mean_bg = np.mean(self.weights[self.labels == 0])
            self.scale_factor = mean_bg / mean_sig
            print(f"Computed signal scale factor: {self.scale_factor:.4f} (mean_sig: {mean_sig:.4f}, mean_bg: {mean_bg:.4f})")
            print(f"Num events: {len(self.events)}, signal: {np.sum(self.labels==1)}, background: {np.sum(self.labels==0)}")

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self, tree_names):
        self.events = []
        all_labels = []
        all_weights = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file: continue
                tree = file[tree_name]
                inclusive_branch = 'inclusive_strange_channels'
                exclusive_branch = 'exclusive_strange_channels'
                inclusive_array = tree[inclusive_branch].array(library="np")
                exclusive_array = tree[exclusive_branch].array(library="np")
                weights_array = tree[self.weight_branch].array(library="np")
                for i in range(tree.num_entries):
                    inc = int(inclusive_array[i])
                    exc = int(exclusive_array[i])
                    if exc == 52:
                        label = 1
                    elif inc == 20:
                        label = 0
                    else:
                        continue
                    self.events.append((tree_name, i))
                    all_labels.append(label)
                    all_weights.append(weights_array[i])
        self.labels = np.array(all_labels)
        self.weights = np.array(all_weights)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        label = self.labels[idx]
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]

        views_data = []
        for branch_name in self.image_branches:
            flattened_image = tree[branch_name].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
            image_grid = np.array(flattened_image, dtype=np.float32).reshape(512, 512)
            image_grid = np.maximum(image_grid, 0)
            rows, cols = np.nonzero(image_grid > 1e-6)
            if rows.size == 0:
                coords = torch.empty((0, 2), dtype=torch.int32)
                feats = torch.empty((0, 1), dtype=torch.float32)
            else:
                coords = torch.from_numpy(np.stack([rows, cols], axis=1)).int()
                feats = torch.from_numpy(np.log10(image_grid[rows, cols] + 1e-6)[:, np.newaxis]).float()
                feats = (feats - feats.mean()) / (feats.std() + 1e-8)
            views_data.append((coords, feats))
        
        weight = self.weights[idx]
        if label == 1 and self.is_train:
            weight *= self.scale_factor
        
        return views_data, label, weight

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.signal_indices = np.where(self.dataset.labels == 1)[0]
        self.background_indices = np.where(self.dataset.labels == 0)[0]
        if len(self.signal_indices) < self.half_batch or len(self.background_indices) < self.half_batch:
            raise ValueError("Not enough signal or background events for the batch size.")
        self.num_batches = min(len(self.signal_indices), len(self.background_indices)) // self.half_batch

    def __iter__(self):
        np.random.shuffle(self.signal_indices)
        np.random.shuffle(self.background_indices)
        for i in range(self.num_batches):
            start = i * self.half_batch
            end = (i + 1) * self.half_batch
            signal_batch = self.signal_indices[start:end].tolist()
            background_batch = self.background_indices[start:end].tolist()
            batch = signal_batch + background_batch
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def collate_fn(batch):
    batch = [b for b in batch if all(view[0].shape[0] > 0 for view in b[0])]
    if not batch: return None, None, None
    
    views_b, labels_b, weights_b = zip(*batch)
    collated_views_data = []
    for i in range(3):
        coords_list = [view[i][0] for view in views_b]
        feats_list = [view[i][1] for view in views_b]
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_list, feats_list)
        collated_views_data.append((coords_batch, feats_batch))
        
    return collated_views_data, torch.tensor(labels_b, dtype=torch.float32), torch.tensor(weights_b, dtype=torch.float32)

def make_data_loader(config, is_train=True):
    dataset = NeutrinoDataset(
        config.root_file_path, config.tree_names, is_train=is_train
    )
    
    if is_train:
        batch_sampler = BalancedBatchSampler(dataset, config.batch_size)
        loader = DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=config.num_workers,
            collate_fn=collate_fn, pin_memory=True)
    else:
        loader = DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            collate_fn=collate_fn, pin_memory=True, shuffle=False)
    return loader, dataset

def test(net, device, config):
    data_loader, _ = make_data_loader(config, is_train=False)
    net.eval()
    all_labels, all_preds, all_weights = [], [], []
    with torch.no_grad():
        for views_data, labels, weights in data_loader:
            if views_data is None: continue
            
            input_views = [
                ME.SparseTensor(
                    features=feats.to(device), 
                    coordinates=coords.to(device)
                ) for coords, feats in views_data
            ]
            
            labels = labels.to(device)
            weights = weights.to(device)
            
            outputs = net(input_views)
            logit = outputs['is_strange_signal'].squeeze()
            preds = (torch.sigmoid(logit) > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_weights.extend(weights.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_weights = np.array(all_weights)
    
    signal_mask = (all_labels == 1)
    if np.sum(signal_mask) == 0:
        return 0.0
    tp_weights = all_weights[(all_labels == 1) & (all_preds == 1)].sum()
    fn_weights = all_weights[(all_labels == 1) & (all_preds == 0)].sum()
    sensitivity = tp_weights / (tp_weights + fn_weights + 1e-6)
    return sensitivity

def focal_bce_with_logits(input, target, weights=None, gamma=2.0, alpha=0.75, reduction='mean'):
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(-bce_loss)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
    if weights is not None:
        focal_loss = focal_loss * weights
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

def train(net, device, config):
    optimizer = optim.Adam(
        net.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1)
    best_metric = 0.0
    
    train_loader, train_dataset = make_data_loader(config, is_train=True)

    for epoch in range(config.epochs):
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (views_data, labels, weights) in enumerate(train_loader):
            if views_data is None: continue

            optimizer.zero_grad(set_to_none=True)
            
            input_views = [
                ME.SparseTensor(
                    features=feats.to(device), 
                    coordinates=coords.to(device)
                ) for coords, feats in views_data
            ]
            
            labels = labels.to(device)
            weights = weights.to(device)
        
            outputs = net(input_views)
            logit = outputs['is_strange_signal'].squeeze()
            loss = focal_bce_with_logits(logit, labels, weights=weights, gamma=2.0, alpha=0.75, reduction='mean')
            
            if torch.isnan(loss):
                continue

            loss.backward()
            clip_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logit) > 0.5).long()
                batch_acc = (preds == labels.long()).float().mean().item()
                tp_weights = torch.sum(weights * (labels == 1) * (preds == 1))
                fn_weights = torch.sum(weights * (labels == 1) * (preds == 0))
                batch_sens = tp_weights / (tp_weights + fn_weights + 1e-6)

            running_loss = 0.95 * running_loss + 0.05 * loss.item()
            running_acc = 0.95 * running_acc + 0.05 * batch_acc
            
            print(
                f"Epoch: {epoch+1}, Iter: {i}, Batch Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.3f}, "
                f"Batch Sensitivity: {batch_sens.item():.3f}, Avg Loss: {running_loss:.4f}, Avg Acc: {running_acc:.3f}, "
                f"Grad Norm: {clip_norm.item():.4f}, Mean Logit: {logit.mean().item():.4f}"
            )
        
        val_sensitivity = test(net, device, config)
        scheduler.step(val_sensitivity)

        if val_sensitivity > best_metric:
            best_metric = val_sensitivity
            torch.save(net.state_dict(), config.weights)
        
        print(
            f"End of Epoch {epoch+1}, Validation sensitivity: {val_sensitivity:.4f}. Best: {best_metric:.4f}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights", type=str, default="dunecvn.pth")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--root_file_path", type=str, required=True)
    parser.add_argument("--tree_names", type=str, required=True)
    
    config = parser.parse_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = MinkowskiDUNECVN().to(device)
    
    train(net, device, config)
    sensitivity = test(net, device, config)
    print(f"Final Test sensitivity: {sensitivity}")