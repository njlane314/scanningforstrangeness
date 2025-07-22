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
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

class NeutrinoDataset(Dataset):
    def __init__(self, root_file_path, tree_names, image_branches, label_branch, image_shape=(512, 512)):
        self.root_file_path = root_file_path
        self.image_branches = image_branches
        self.label_branch = label_branch
        self.image_shape = image_shape
        self.file_handle = None
        self._scan_root_file(tree_names)

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self, tree_names):
        logging.info(f"Scanning ROOT file: {self.root_file_path}...")
        self.events = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file:
                    logging.warning(f"TTree '{tree_name}' not found. Skipping.")
                    continue
                tree = file[tree_name]
                num_entries = tree.num_entries
                logging.info(f"Scanning TTree '{tree_name}' with {num_entries} entries.")
                self.events.extend([(tree_name, i) for i in range(num_entries)])

        if not self.events:
            raise RuntimeError("No events loaded. Check TTree names and file content.")
        logging.info(f"Scan complete. Found {len(self.events)} total events.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]

        views_coords = []
        views_feats = []

        for branch_name in self.image_branches:
            flattened_image = tree[branch_name].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
            image_grid = np.array(flattened_image, dtype=np.float32).reshape(self.image_shape)
            rows, cols = np.where(image_grid > 1.0)

            if rows.size == 0:
                coords = torch.empty((0, 2), dtype=torch.int32)
                feats = torch.empty((0, 1), dtype=torch.float32)
            else:
                coords_np = np.stack([rows, cols], axis=1)
                feats_np = np.log(image_grid[rows, cols])[:, np.newaxis]
                coords = torch.from_numpy(coords_np).int()
                feats = torch.from_numpy(feats_np).float()

            views_coords.append(coords)
            views_feats.append(feats)

        raw_label = tree[self.label_branch].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
        return views_coords, views_feats, torch.tensor(raw_label, dtype=torch.long)

def contrastive_collate_fn(batch):
    # Filter events to include only those with exactly 3 non-empty views
    filtered_batch = [
        item for item in batch
        if len(item[0]) == 3 and all(coords.size(0) > 0 for coords in item[0])
    ]
    
    # Handle case where no valid events remain
    if not filtered_batch:
        return (
            torch.empty(0, 3, dtype=torch.int32),
            torch.empty(0, 1, dtype=torch.float32),
            torch.empty(0, dtype=torch.long)
        )
    
    all_coords, all_feats, labels = [], [], []
    for views_coords, views_feats, label in filtered_batch:
        all_coords.extend(views_coords)
        all_feats.extend(views_feats)
        labels.append(label)
    
    # Assign batch indices to each view
    batch_indices = []
    for i, coords in enumerate(all_coords):
        batch_indices.append(torch.full((coords.size(0), 1), i, dtype=coords.dtype))
    
    coords_batch = [torch.cat([b, c], dim=1) for c, b in zip(all_coords, batch_indices)]
    
    if not coords_batch:
        return (
            torch.empty(0, 3, dtype=torch.int32),
            torch.empty(0, 1, dtype=torch.float32),
            torch.empty(0, dtype=torch.long)
        )
    
    coords = torch.cat(coords_batch)
    feats = torch.cat(all_feats)
    labels = torch.stack(labels)

    return coords, feats, labels

class MinkowskiFCNN(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_dim, embedding_channel=512, channels=(16, 24, 32, 48, 64), D=2):
        super().__init__(D)
        self.network_initialization(in_channel, out_dim, channels, embedding_channel, kernel_size=3, D=D)
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(ME.MinkowskiLinear(in_channel, out_channel, bias=False), ME.MinkowskiBatchNorm(out_channel), ME.MinkowskiLeakyReLU())

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(ME.MinkowskiConvolution(in_channel, out_channel, kernel_size=kernel_size, stride=stride, dimension=self.D), ME.MinkowskiBatchNorm(out_channel), ME.MinkowskiLeakyReLU())

    def network_initialization(self, in_channel, out_dim, channels, embedding_channel, kernel_size, D):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(channels[0], channels[1], kernel_size=kernel_size, stride=1)
        self.conv2 = self.get_conv_block(channels[1], channels[2], kernel_size=kernel_size, stride=2)
        self.conv3 = self.get_conv_block(channels[2], channels[3], kernel_size=kernel_size, stride=2)
        self.conv4 = self.get_conv_block(channels[3], channels[4], kernel_size=kernel_size, stride=2)
        self.conv5 = nn.Sequential(
            self.get_conv_block(channels[1] + channels[2] + channels[3] + channels[4], embedding_channel // 4, kernel_size=3, stride=2),
            self.get_conv_block(embedding_channel // 4, embedding_channel // 2, kernel_size=3, stride=2),
            self.get_conv_block(embedding_channel // 2, embedding_channel, kernel_size=3, stride=2),
        )
        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.projection_head = nn.Sequential(
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_dim, bias=True),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, ME.MinkowskiLinear):
                nn.init.kaiming_normal_(m.linear.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (ME.MinkowskiBatchNorm, nn.BatchNorm1d)):
                target_weight = m.bn.weight if hasattr(m, 'bn') else m.weight
                target_bias = m.bn.bias if hasattr(m, 'bn') else m.bias
                nn.init.constant_(target_weight, 1)
                nn.init.constant_(target_bias, 0)

    def forward(self, x: ME.TensorField):
        x0 = self.mlp1(x)
        y = x0.sparse()
        logging.info(f"Initial points: {y.C.shape[0]}")

        y1 = self.conv1(y)
        y1 = self.pool(y1)
        logging.info(f"After conv1+pool: {y1.C.shape[0]} points")
        
        y2 = self.conv2(y1)
        y2 = self.pool(y2)
        logging.info(f"After conv2+pool: {y2.C.shape[0]} points")
        
        y3 = self.conv3(y2)
        y3 = self.pool(y3)
        logging.info(f"After conv3+pool: {y3.C.shape[0]} points")
        
        y4 = self.conv4(y3)
        y4 = self.pool(y4)
        logging.info(f"After conv4+pool: {y4.C.shape[0]} points")

        z1 = y1.slice(x0)
        z2 = y2.slice(x0)
        z3 = y3.slice(x0)
        z4 = y4.slice(x0)
        z = ME.cat(z1, z2, z3, z4)
        y = z.sparse()
        logging.info(f"After slicing+cat: {y.C.shape[0]} points")

        y = self.conv5(y)
        logging.info(f"After conv5: {y.C.shape[0]} points")

        y1 = self.global_max_pool(y)
        y2 = self.global_avg_pool(y)
        return self.projection_head(ME.cat(y1, y2)).F

def info_nce_loss(features, temperature, n_views, device):
    if features.shape[0] % n_views != 0:
        raise ValueError(f"Feature batch size {features.shape[0]} must be divisible by n_views {n_views}")
    
    batch_size = features.shape[0] // n_views 
    
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0).to(device)
    positives_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    positives_mask.fill_diagonal_(0) 
    
    total_loss = 0.0
    num_anchors = features.shape[0]
    
    for i in range(num_anchors):
        pos_sim = similarity_matrix[i][positives_mask[i].bool()]  
        neg_sim = similarity_matrix[i][~positives_mask[i].bool()] 
        
        anchor_loss = 0.0
        for pos in pos_sim:
            logits = torch.cat([pos.unsqueeze(0), neg_sim], dim=0)
            target = torch.tensor([0], dtype=torch.long).to(device)
            anchor_loss += F.cross_entropy(logits.unsqueeze(0), target)
        anchor_loss /= len(pos_sim)  
        total_loss += anchor_loss
    
    return total_loss / num_anchors

def train(net, train_loader, device, config):
    optimiser = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.max_steps)
    train_loader_iter = iter(train_loader)
    net.train()
    best_loss = float('inf')
    logging.info(f"--- Starting Contrastive Training for {config.max_steps} steps ---")

    for i in range(config.max_steps):
        optimiser.zero_grad()
        try:
            coords, feats, _ = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            coords, feats, _ = next(train_loader_iter)
        if coords.shape[0] == 0:
            logging.info(f"Iter: {i + 1}, Skipping batch: No valid coordinates after collation.")
            continue

        input_field = ME.TensorField(features=feats.to(device), coordinates=coords.to(device))
        
        features = net(input_field)
        loss = info_nce_loss(features, config.temperature, len(config.image_branches), device)
        
        if loss is None:
            logging.info(f"Iter: {i + 1}, Skipping batch: Loss calculation failed (e.g., mismatched views, zero features).")
            continue

        loss.backward()
        optimiser.step()
        scheduler.step()

        if (i + 1) % config.stat_freq == 0:
            logging.info(f"Iter: {i + 1}, Loss: {loss.item():.3e}, LR: {scheduler.get_last_lr()[0]:.2e}")
        if (i + 1) % config.val_freq == 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                logging.info(f"New best training loss: {best_loss:.4f}, saving model to {config.weights}")
                torch.save({"state_dict": net.state_dict(), "curr_iter": i + 1, "best_loss": best_loss}, config.weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinkowskiEngine Contrastive Feature Learning")
    parser.add_argument("--root_file_path", type=str, required=True, help="Path to the input ROOT file.")
    parser.add_argument("--tree_names", nargs='+', required=True, help="List of TTree names to process.")
    parser.add_argument("--weights", type=str, default="contrastive_feature_extractor.pth", help="Path to save the trained model.")
    parser.add_argument("--image_branches", nargs='+', required=True, help="List of branch names for different plane views.")
    parser.add_argument("--label_branch", type=str, default="exclusive_strange_channels", help="Branch name for event labels (not used in loss).")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stat_freq", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=500)
    parser.add_argument('--out_dim', default=128, type=int)
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument("--seed", type=int, default=777)
    config = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = NeutrinoDataset(
        root_file_path=config.root_file_path,
        tree_names=config.tree_names,
        image_branches=config.image_branches,
        label_branch=config.label_branch
    )
    train_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=contrastive_collate_fn, shuffle=True, drop_last=True)
    net = MinkowskiFCNN(in_channel=1, D=2, out_dim=config.out_dim).to(device)
    logging.info(f"Model: {net}")
    train(net, train_loader, device, config)
    logging.info("--- Training Finished ---")