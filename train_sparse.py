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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import sklearn.metrics as metrics
import MinkowskiEngine as ME

def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

class NeutrinoDataset(Dataset):
    def __init__(self, root_file_path, tree_names, image_branch, label_branch, weight_branch, quantization_size=1.0, image_shape=(512, 512)):
        self.root_file_path = root_file_path
        self.image_branch = image_branch
        self.label_branch = label_branch
        self.quantization_size = quantization_size
        self.image_shape = image_shape
        self.file_handle = None
        self._scan_root_file(tree_names, weight_branch, label_branch)

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self, tree_names, weight_branch, label_branch):
        logging.info(f"Scanning ROOT file: {self.root_file_path}...")
        self.events = []
        all_weights = []
        unique_raw_labels = set()
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file:
                    logging.warning(f"TTree '{tree_name}' not found. Skipping.")
                    continue
                tree = file[tree_name]
                num_entries = tree.num_entries
                logging.info(f"Scanning TTree '{tree_name}' with {num_entries} entries.")
                weights = tree[weight_branch].array(library="np")
                labels = tree[label_branch].array(library="np")
                unique_raw_labels.update(labels)
                for i in range(num_entries):
                    self.events.append((tree_name, i))
                all_weights.append(weights)
        if not self.events:
            raise RuntimeError("No events loaded. Check TTree names and file content.")
        
        self.weights = np.concatenate(all_weights)
        
        sorted_labels = sorted(list(unique_raw_labels))
        self.label_map = {raw_label: i for i, raw_label in enumerate(sorted_labels)}
        self.num_classes = len(self.label_map)
        
        logging.info(f"Scan complete. Found {len(self.events)} total events.")
        logging.info(f"Automatically detected {self.num_classes} unique classes.")
        logging.info(f"Label mapping: {self.label_map}")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        
        flattened_image = tree[self.image_branch].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
        raw_label = tree[self.label_branch].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
        
        clean_label = self.label_map[raw_label]
            
        image_grid = np.array(flattened_image, dtype=np.float32).reshape(self.image_shape)
        
        rows, cols = np.where(image_grid > 1.0)
        
        if rows.size == 0:
            return (torch.empty((0, 2), dtype=torch.int32),
                    torch.empty((0, 1), dtype=torch.float32),
                    torch.tensor([clean_label], dtype=torch.long))

        coords = np.stack([rows, cols], axis=1)
        feats = np.log(image_grid[rows, cols])[:, np.newaxis]

        return (torch.from_numpy(coords).int(),
                torch.from_numpy(feats).float(),
                torch.tensor([clean_label], dtype=torch.long))

def make_data_loader(dataset, config, phase="train"):
    assert phase in ["train", "test"]
    is_train = phase == "train"
    
    sampler = WeightedRandomSampler(
        weights=dataset.weights, 
        num_samples=len(dataset), 
        replacement=True
    ) if is_train else None

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=ME.utils.SparseCollation(),
        pin_memory=True,
    )

class MinkowskiFCNN(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, D=2):
        super().__init__(D)
        channels = (32, 64, 128, 256)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, channels[0], kernel_size=3, stride=1, dimension=D), ME.MinkowskiBatchNorm(channels[0]), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=3, stride=2, dimension=D), ME.MinkowskiBatchNorm(channels[1]), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=3, stride=2, dimension=D), ME.MinkowskiBatchNorm(channels[2]), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(channels[2], channels[3], kernel_size=3, stride=2, dimension=D), ME.MinkowskiBatchNorm(channels[3]), ME.MinkowskiReLU(),
            ME.MinkowskiGlobalMaxPooling(),
            ME.MinkowskiLinear(channels[3], out_channel, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.SparseTensor):
        return self.net(x)

def criterion(pred, labels, smoothing=0.2):
    labels = labels.contiguous().view(-1)
    if smoothing > 0.0:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        loss = -(F.log_softmax(pred, dim=1) * one_hot).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, labels, reduction="mean")
    return loss

def test(net, data_loader, device):
    net.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            coords, feats, target_labels = batch
            if coords.shape[0] == 0: continue
            input = ME.SparseTensor(features=feats, coordinates=coords, device=device)
            logit = net(input)
            pred = torch.argmax(logit.F, 1)
            labels.append(target_labels.cpu().numpy())
            preds.append(pred.cpu().numpy())
    if not labels: return 0.0
    return metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds))

def train(net, train_loader, test_loader, device, config):
    optimiser = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.max_steps)
    logging.info(f"Optimiser: {optimiser}")
    
    train_iter = iter(train_loader)
    best_metric = 0
    net.train()

    logging.info(f"--- Starting Training for {config.max_steps} steps ---")
    for i in range(config.max_steps):
        optimiser.zero_grad()
        try:
            coords, feats, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            coords, feats, labels = next(train_iter)
        if coords.shape[0] == 0: continue

        input = ME.SparseTensor(features=feats, coordinates=coords, device=device)
        logit = net(input)
        
        loss = criterion(logit.F, labels.to(device))
        loss.backward()
        optimiser.step()
        scheduler.step()

        if (i + 1) % config.stat_freq == 0:
            logging.info(f"Iter: {i + 1}, Loss: {loss.item():.3e}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if (i + 1) % config.val_freq == 0 and i > 0:
            accuracy = test(net, test_loader, device)
            if best_metric < accuracy:
                best_metric = accuracy
                logging.info(f"New best validation accuracy: {accuracy:.4f}, saving model to {config.weights}")
                torch.save({"state_dict": net.state_dict(), "curr_iter": i + 1}, config.weights)
            else:
                logging.info(f"Validation accuracy: {accuracy:.4f}. Best accuracy: {best_metric:.4f}")
            net.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinkowskiEngine Neutrino Classification Training")
    parser.add_argument("--root_file_path", type=str, required=True, help="Path to the input ROOT file.")
    parser.add_argument("--tree_names", nargs='+', required=True, help="List of TTree names to process.")
    parser.add_argument("--image_branch", type=str, default="detector_image_u", help="Branch name for the detector image.")
    parser.add_argument("--label_branch", type=str, default="exclusive_strange_channels", help="Branch name for the event labels.")
    parser.add_argument("--weight_branch", type=str, default="base_event_weight", help="Branch name for the event weights.")
    parser.add_argument("--voxel_size", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--val_freq", type=int, default=500)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stat_freq", type=int, default=100)
    parser.add_argument("--weights", type=str, default="neutrino_model.pth")
    parser.add_argument("--seed", type=int, default=777)
    config = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = NeutrinoDataset(
        root_file_path=config.root_file_path,
        tree_names=config.tree_names,
        image_branch=config.image_branch,
        label_branch=config.label_branch,
        weight_branch=config.weight_branch,
        quantization_size=config.voxel_size
    )
    
    net = MinkowskiFCNN(in_channel=1, out_channel=dataset.num_classes, D=2).to(device)
    logging.info(f"Model: {net}")
    
    train_loader = make_data_loader(dataset, config, phase="train")
    test_loader = make_data_loader(dataset, config, phase="test")

    train(net, train_loader, test_loader, device, config)

    logging.info("--- Final Test Evaluation ---")
    accuracy = test(net, test_loader, device)
    logging.info(f"Final test accuracy: {accuracy:.4f}")