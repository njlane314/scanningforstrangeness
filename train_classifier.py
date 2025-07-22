# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 if you use any part
# of the code.
import argparse
import sklearn.metrics as metrics
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Sampler
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import MinkowskiEngine as ME
from MinkowskiEngine.modules.senet_block import SELayer
import uproot

# ==============================================================================
#                      DUNE CVN MODEL IMPLEMENTATION
# ==============================================================================

class PreActivationSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dimension=2):
        super().__init__()
        self.bn1 = ME.MinkowskiBatchNorm(in_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=stride, dimension=dimension
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, stride=1, dimension=dimension
        )
        self.se = SELayer(channel=out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=stride, dimension=dimension
            )

    def forward(self, x: ME.SparseTensor):
        preactivated_x = self.relu(self.bn1(x))
        if self.shortcut is not None:
            residual = self.shortcut(preactivated_x)
        else:
            residual = x
        out = self.conv1(preactivated_x)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.se(out)
        out += residual
        return out

class MinkowskiDUNECVN(nn.Module):
    def __init__(self,
                 initial_conv_filters=64,
                 depth=[3, 4, 6, 3],
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
             self._make_layer(initial_conv_filters, filters[0], depth[0], stride=1, dimension=dimension)
             for _ in range(3)
        ])
        concat_filters = filters[0] * 3
        self.main_trunk = nn.Sequential(
            self._make_layer(concat_filters, filters[1], depth[1], stride=2, dimension=dimension),
            self._make_layer(filters[1], filters[2], depth[2], stride=2, dimension=dimension),
            self._make_layer(filters[2], filters[3], depth[3], stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(filters[3]),
            ME.MinkowskiReLU(inplace=True)
        )
        self.glob_pool = ME.MinkowskiGlobalPooling()
        self.output_heads = nn.ModuleDict()
        for name, num_neurons in zip(output_names, output_neurons):
            self.output_heads[name] = ME.MinkowskiLinear(filters[3], num_neurons)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dimension):
        layers = [PreActivationSEBlock(in_channels, out_channels, stride=stride, dimension=dimension)]
        for _ in range(1, num_blocks):
            layers.append(PreActivationSEBlock(out_channels, out_channels, stride=1, dimension=dimension))
        return nn.Sequential(*layers)

    def forward(self, inputs: list[ME.SparseTensor]):
        branch_outputs = [self.stage1_blocks[i](self.branches[i](x)) for i, x in enumerate(inputs)]
        x = ME.cat(branch_outputs)
        x = self.main_trunk(x)
        pooled_features = self.glob_pool(x)
        outputs = {
            name: head(pooled_features).F for name, head in self.output_heads.items()
        }
        return outputs

# ==============================================================================
#                      DATA LOADING AND TRAINING SCRIPT
# ==============================================================================

class NeutrinoDataset(torch.utils.data.Dataset):
    def __init__(self, root_file_path, tree_names, label_branch):
        self.root_file_path = root_file_path
        self.image_branches = [f'detector_image_{p}' for p in ['u', 'v', 'w']]
        self.label_branch = label_branch
        self.file_handle = None
        self._scan_root_file(tree_names.split(','))

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self, tree_names):
        self.events = []
        all_labels = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file: continue
                tree = file[tree_name]
                labels_array = tree[self.label_branch].array(library="np")
                for i in range(tree.num_entries):
                    self.events.append((tree_name, i))
                    all_labels.append(labels_array[i])
        self.labels = np.array(all_labels, dtype=int).flatten()

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
            rows, cols = np.nonzero(image_grid > 1e-6)
            if rows.size == 0:
                coords = torch.empty((0, 2), dtype=torch.int32)
                feats = torch.empty((0, 1), dtype=torch.float32)
            else:
                coords = torch.from_numpy(np.stack([rows, cols], axis=1)).int()
                feats = torch.from_numpy(np.log10(image_grid[rows, cols] + 1e-6)[:, np.newaxis]).float()
                feats = (feats - feats.mean()) / (feats.std() + 1e-8)
            views_data.append((coords, feats))
        
        return views_data, label

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
            start, end = i * self.half_batch, (i + 1) * self.half_batch
            signal_batch = self.signal_indices[start:end].tolist()
            background_batch = self.background_indices[start:end].tolist()
            batch = signal_batch + background_batch
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def collate_fn(batch):
    batch = [b for b in batch if all(view[0].shape[0] > 0 for view in b[0])]
    if not batch: return None, None
    
    views_b, labels_b = zip(*batch)
    collated_views = []
    for i in range(3): # For u, v, w views
        coords_list = [view[i][0] for view in views_b]
        feats_list = [view[i][1] for view in views_b]
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_list, feats_list)
        collated_views.append(ME.SparseTensor(feats_batch, coords_batch))
        
    return collated_views, torch.tensor(labels_b, dtype=torch.float32)

def make_data_loader(config, is_train=True):
    dataset = NeutrinoDataset(
        config.root_file_path, config.tree_names, 'inclusive_strange_channels')
    
    if is_train:
        batch_sampler = BalancedBatchSampler(dataset, config.batch_size)
        return DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=config.num_workers,
            collate_fn=collate_fn, pin_memory=True)
    else: 
        return DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            collate_fn=collate_fn, pin_memory=True, shuffle=False)

def test(net, device, config):
    data_loader = make_data_loader(config, is_train=False)
    net.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for views, labels in data_loader:
            if views is None: continue
            input_views = [view.to(device) for view in views]
            labels = labels.to(device)
            
            outputs = net(input_views)
            logit = outputs['is_strange_signal'].squeeze()
            preds = (torch.sigmoid(logit) > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    return metrics.accuracy_score(all_labels, all_preds) if all_labels else 0.0

def train(net, device, config):
    optimizer = optim.SGD(
        net.parameters(), 
        lr=config.lr, 
        momentum=config.momentum, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    best_metric = 0.0
    
    train_loader = make_data_loader(config, is_train=True)

    for epoch in range(config.epochs):
        net.train()
        train_iter = iter(train_loader)
        running_loss, running_acc = 0.0, 0.0
        
        for i, (views, labels) in enumerate(train_iter):
            if views is None: continue

            optimizer.zero_grad(set_to_none=True)
            
            input_views = [view.to(device) for view in views]
            labels = labels.to(device)
            
            with autocast():
                outputs = net(input_views)
                logit = outputs['is_strange_signal'].squeeze()
                loss = criterion(logit, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds = (torch.sigmoid(logit) > 0.5).long()
                batch_acc = (preds == labels.long()).float().mean().item()

            if i == 0 and epoch == 0:
                running_loss, running_acc = loss.item(), batch_acc
            else:
                running_loss = 0.95 * running_loss + 0.05 * loss.item()
                running_acc = 0.95 * running_acc + 0.05 * batch_acc
            
            if i % config.stat_freq == 0:
                print(
                    f"Epoch: {epoch+1}, Iter: {i}, Batch Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.3f}, "
                    f"Avg Loss: {running_loss:.4f}, Avg Acc: {running_acc:.3f}"
                )
        
        # Validation step at the end of each epoch
        val_loss = 0
        val_accuracy = test(net, device, config)
        scheduler.step(val_accuracy) # Or use a validation loss if preferred

        if val_accuracy > best_metric:
            best_metric = val_accuracy
            torch.save(net.state_dict(), config.weights)
        
        print(
            f"End of Epoch {epoch+1}, Validation accuracy: {val_accuracy:.4f}. Best: {best_metric:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stat_freq", type=int, default=10)
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
    accuracy = test(net, device, config)
    print(f"Final Test accuracy: {accuracy}")