import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import MinkowskiEngine as ME
import uproot
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import random
from scipy.spatial import cKDTree

# ==============================================================================
# ## --- Configuration --- ##
# ==============================================================================
BIAS = False
DIMENSION = 2
DETECTOR_IMAGE_BRANCH = 'detector_image_w'
BACKGROUND_TREES = ["mc_inclusive_run1_fhc"]
SIGNAL_TREES = ["mc_strangeness_run1_fhc"]


# ==============================================================================
# ## --- Data Handling with Rich Feature Engineering --- ##
# ==============================================================================
class AnomalyDataset(Dataset):
    def __init__(self, root_file_path, tree_names, label):
        self.root_file_path = root_file_path
        self.tree_names = tree_names
        self.label = torch.tensor(label, dtype=torch.float32)
        self.events = []
        self.file_handle = uproot.open(self.root_file_path)

        for tree_name in self.tree_names:
            if tree_name not in self.file_handle:
                logging.warning(f"TTree '{tree_name}' not found.")
                continue
            num_entries = self.file_handle[tree_name].num_entries
            for i in range(num_entries):
                self.events.append((tree_name, i))
        logging.info(f"Loaded {len(self.events)} event indices for label {int(self.label.item())} from {self.tree_names}.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        tree = self.file_handle[tree_name]
        flattened_image = tree[DETECTOR_IMAGE_BRANCH].array(entry_start=entry_index, entry_stop=entry_index + 1, library="np")[0]

        image_2d = flattened_image.reshape(512, 512)
        rows, cols = np.where(image_2d > 0)

        if len(rows) < 10: return None

        coords = np.stack([rows, cols], axis=1)
        adc_values = image_2d[rows, cols]
        
        # --- Rich Feature Engineering ---
        # 1. Normalized Coordinates (scaled to [-1, 1] for better network performance)
        norm_rows = (rows - 256.0) / 256.0
        norm_cols = (cols - 256.0) / 256.0
        
        # 2. Log-transformed ADC
        log_adc = np.log10(adc_values + 1e-6)
        
        # 3. Local Hit Density (a powerful topological feature)
        tree = cKDTree(coords)
        # Count hits within a 5-pixel radius for each point
        local_density = tree.query_ball_point(coords, r=5, return_length=True).astype(np.float32)
        # Normalize the density for stability
        norm_density = np.log1p(local_density) # Use log1p for better scaling of density

        # Stack features into a 4-channel tensor
        feats = np.stack([norm_rows, norm_cols, log_adc, norm_density], axis=1).astype(np.float32)

        return (coords.astype(np.int32), feats, self.label)


def collate_fn_classifier(list_data):
    list_data = [data for data in list_data if data is not None]
    if not list_data: return None
    coords_list, feats_list, labels_list = zip(*list_data)
    batched_coords = ME.utils.batched_coordinates(coords_list)
    batched_feats = torch.from_numpy(np.vstack(feats_list))
    labels = torch.stack(labels_list)
    return {"coords": batched_coords, "feats": batched_feats, "labels": labels}


# ==============================================================================
# ## --- Model Architecture (ResNet34) --- ##
# ==============================================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, dimension=-1):
        super(BasicBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.norm1(out); out = self.relu(out)
        out = self.conv2(out); out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        super(ResNetBase, self).__init__()
        self.D = D
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D, bias=BIAS),
            ME.MinkowskiBatchNorm(self.inplanes), ME.MinkowskiReLU(inplace=True))
        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2, dilation=4)
        self.glob_pool = ME.MinkowskiGlobalPooling()
        self.dropout = ME.MinkowskiDropout()
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, dimension=self.D),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, dimension=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.glob_pool(x)
        x = self.dropout(x)
        return self.final(x)

class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


# ==============================================================================
# ## --- Core Logic --- ##
# ==============================================================================
def test(model, test_loader, device, iteration):
    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            input = ME.SparseTensor(features=batch['feats'], coordinates=batch['coords'], device=device)
            logits = model(input).F
            all_labels.append(batch['labels'].cpu().numpy())
            all_scores.append(logits.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores).flatten()
    auc = roc_auc_score(all_labels, all_scores)

    plt.figure(figsize=(10, 6))
    plt.hist(all_scores[all_labels == 0], bins=100, alpha=0.7, label='Background', density=True)
    plt.hist(all_scores[all_labels == 1], bins=100, alpha=0.7, label='Signal', density=True)
    plt.title(f'Classifier Score Distribution - Iteration {iteration} (AUC: {auc:.4f})')
    plt.xlabel('Model Output (Logit)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('separation_histogram_classifier.png')
    plt.close()
    print(f"Validation AUC at iteration {iteration}: {auc:.4f}")
    return auc

def main(config):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_background_dset = AnomalyDataset(config.root_file_path, BACKGROUND_TREES, label=0)
    train_signal_dset = AnomalyDataset(config.root_file_path, SIGNAL_TREES, label=1)
    train_dset = ConcatDataset([train_background_dset, train_signal_dset])
    
    train_loader = DataLoader(
        train_dset, batch_size=config.batch_size, collate_fn=collate_fn_classifier,
        num_workers=0, shuffle=True
    )
    
    test_background_dset = AnomalyDataset(config.root_file_path, BACKGROUND_TREES, label=0)
    test_signal_dset = AnomalyDataset(config.root_file_path, SIGNAL_TREES, label=1)
    test_dset = ConcatDataset([test_background_dset, test_signal_dset])
    
    sample_size = min(config.test_sample_size, len(test_dset))
    print(f"Using a random validation subset of {sample_size} events.")
    test_indices = np.random.choice(len(test_dset), sample_size, replace=False)
    val_subset = Subset(test_dset, test_indices)
    
    test_loader = DataLoader(
        val_subset, batch_size=config.batch_size, collate_fn=collate_fn_classifier,
        num_workers=0, shuffle=False
    )

    # The model now takes a 4-channel input
    net = ResNet34(in_channels=4, out_channels=1, D=DIMENSION).to(device)
    optimizer = Adam(net.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_iter)
    criterion = nn.BCEWithLogitsLoss()

    print("\n--- Model Architecture (ResNet34 with Rich Features) ---")
    print(net)
    print("---------------------------------------------------------")

    best_auc = 0
    net.train()
    for i in range(config.max_iter):
        if i > 0 and i % config.val_freq == 0:
            auc = test(net, test_loader, device, i)
            if auc > best_auc:
                best_auc = auc
                print(f"New best AUC: {best_auc:.4f}. Saving model weights to {config.weights}")
                torch.save(net.state_dict(), config.weights)
            net.train()
        
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            continue
        if batch is None: continue
        
        input = ME.SparseTensor(features=batch['feats'], coordinates=batch['coords'], device=device)
        labels = batch['labels'].to(device)
        
        logits = net(input).F
        
        loss = criterion(logits.squeeze(), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 1 == 0:
            print(f"Iteration: {i}, Classification Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# ==============================================================================
# ## --- Entry Point --- ##
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_file_path", type=str, required=True)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--val_freq", type=int, default=250)
    parser.add_argument("--test_sample_size", type=int, default=4000)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weights", type=str, default="binary_classifier_rich_features.pth")
    config = parser.parse_args()
    main(config)