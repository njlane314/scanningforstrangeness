import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset, WeightedRandomSampler
import MinkowskiEngine as ME
import uproot
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
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
    def __init__(self, root_file_path, tree_names, label, branch_filter=None, target_value=None):
        self.root_file_path = root_file_path
        self.tree_names = tree_names
        self.label = torch.tensor(label, dtype=torch.float32)
        self.branch_filter = branch_filter
        self.target_value = target_value
        self.events = []
        self.file_handle = uproot.open(self.root_file_path)

        for tree_name in self.tree_names:
            if tree_name not in self.file_handle:
                logging.warning(f"TTree '{tree_name}' not found.")
                continue
            tree = self.file_handle[tree_name]
            if self.branch_filter:
                if self.branch_filter not in tree:
                    logging.warning(f"Branch '{self.branch_filter}' not found in tree '{tree_name}'.")
                    continue
                flags = tree[self.branch_filter].array(library="np")
                indices = np.where(flags == self.target_value)[0]
            else:
                indices = np.arange(tree.num_entries)
            for idx in indices:
                self.events.append((tree_name, int(idx)))
        logging.info(f"Loaded {len(self.events)} event indices for label {int(self.label.item())} from {self.tree_names}.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        tree = self.file_handle[tree_name]
        flattened_image = tree[DETECTOR_IMAGE_BRANCH].array(entry_start=entry_index, entry_stop=entry_index + 1, library="np")[0]

        image_2d = flattened_image.reshape(512, 512)
        rows, cols = np.where(image_2d > 0)

        if len(rows) == 0:
            return None

        coords = np.stack([rows, cols], axis=1)
        adc_values = image_2d[rows, cols]
        
        norm_rows = (rows - 256.0) / 256.0
        norm_cols = (cols - 256.0) / 256.0
        log_adc = np.log10(adc_values + 1e-6)
        tree_kd = cKDTree(coords)
        local_density = tree_kd.query_ball_point(coords, r=5, return_length=True).astype(np.float32)
        norm_density = np.log1p(local_density)

        feats = np.stack([norm_rows, norm_cols, log_adc, norm_density], axis=1).astype(np.float32)

        return (coords.astype(np.int32), feats, self.label, image_2d)


def collate_fn_classifier(list_data):
    # Filter out None values correctly
    list_data = [data for data in list_data if data is not None]
    # If list_data is empty after filtering, return None
    if not list_data:
        return None
    # Unpack the valid data into coordinates, features, and labels
    coords_list, feats_list, labels_list, images_list = zip(*list_data)
    # Batch the data using MinkowskiEngine utilities
    batched_coords = ME.utils.batched_coordinates(coords_list)
    batched_feats = torch.from_numpy(np.vstack(feats_list))
    labels = torch.stack(labels_list)
    # Return the batched data as a dictionary
    return {"coords": batched_coords, "feats": batched_feats, "labels": labels, "images": list(images_list)}


# ==============================================================================
# ## --- Model Architecture (MinkUNet Classifier) --- ##
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension):
        super(ResidualBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, dimension=dimension
        )
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, dimension=dimension
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, dimension=dimension
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out += residual
        return self.relu(out)

class MinkUNetClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, dimension, base_filters=16, num_strides=3):
        super(MinkUNetClassifier, self).__init__()
        self.dimension = dimension
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=3,
            dimension=dimension
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        current_channels = base_filters
        for i in range(num_strides):
            self.encoder.append(
                ResidualBlock(current_channels, current_channels * 2, dimension=dimension)
            )
            self.encoder.append(
                ME.MinkowskiConvolution(
                    current_channels * 2,
                    current_channels * 2,
                    kernel_size=2,
                    stride=2,
                    dimension=dimension
                )
            )
            current_channels *= 2
        
        # Bottleneck
        self.bottleneck = ResidualBlock(current_channels, current_channels, dimension=dimension)
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_strides):
            up_channels = current_channels // 2
            self.decoder.append(
                ME.MinkowskiConvolutionTranspose(
                    current_channels,
                    up_channels,
                    kernel_size=2,
                    stride=2,
                    dimension=dimension
                )
            )
            # Calculate skip_channels for this level
            skip_channels = base_filters * (2 ** (num_strides - i))
            in_channels = up_channels + skip_channels
            self.decoder.append(
                ResidualBlock(in_channels, up_channels, dimension=dimension)
            )
            current_channels = up_channels
        
        # Global pooling and final linear layer
        self.bn_relu = nn.Sequential(
            ME.MinkowskiBatchNorm(base_filters),
            ME.MinkowskiReLU(inplace=True)
        )
        self.global_pool = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(base_filters, out_channels)
    
    def forward(self, x):
        x = self.conv0(x)
        
        # Encoder with skip connections
        skips = []
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)  # Residual block
            skips.append(x)
            x = self.encoder[i + 1](x)  # Downsampling
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Upsampling
            skip = skips.pop()
            x = ME.cat(x, skip)  # Concatenate skip connection
            x = self.decoder[i + 1](x)  # Residual block
        
        # BatchNorm, ReLU, pooling, and classification
        x = self.bn_relu(x)
        x = self.global_pool(x)
        x = self.linear(x)
        return x  # Shape: [batch_size, 1]

# ==============================================================================
# ## --- Core Logic --- ##
# ==============================================================================
def test(model, test_loader, device, iteration, visualize=False):
    model.eval()
    all_labels, all_scores, all_images = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            input = ME.SparseTensor(features=batch['feats'], coordinates=batch['coords'], device=device)
            logits = model(input).F
            all_labels.append(batch['labels'].cpu().numpy())
            all_scores.append(logits.cpu().numpy())
            all_images.extend(batch['images'])

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores).flatten()
    auc = roc_auc_score(all_labels, all_scores)

    # Compute probabilities from logits
    probs = 1 / (1 + np.exp(-all_scores))

    # Plot probability distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(probs[all_labels == 0], bins=100, alpha=0.7, label='Background', density=True)
    plt.hist(probs[all_labels == 1], bins=100, alpha=0.7, label='Signal', density=True)
    plt.title(f'Classifier Probability Distribution - Iteration {iteration} (AUC: {auc:.4f})')
    plt.xlabel('Model Output (Probability)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('separation_histogram_unet.png')
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Iteration {iteration}')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_unet.png')
    plt.close()

    if visualize:
        # Visualize 5 high prob events (>0.9)
        high_indices = np.where(probs > 0.9)[0][:10]
        for i, idx in enumerate(high_indices):
            img = all_images[idx]
            label_val = all_labels[idx]
            prob_val = probs[idx]
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            plt.title(f'High Prob Event {i+1}: Label={label_val}, Prob={prob_val:.4f}')
            plt.axis('off')
            plt.savefig(f'high_prob_event_{i+1}.png')
            plt.close()

        # Visualize 5 low prob events (<0.1)
        low_indices = np.where(probs < 0.1)[0][:5]
        for i, idx in enumerate(low_indices):
            img = all_images[idx]
            label_val = all_labels[idx]
            prob_val = probs[idx]
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            plt.title(f'Low Prob Event {i+1}: Label={label_val}, Prob={prob_val:.4f}')
            plt.axis('off')
            plt.savefig(f'low_prob_event_{i+1}.png')
            plt.close()

    print(f"Validation AUC at iteration {iteration}: {auc:.4f}")
    return auc

def main(config):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_background_dset = AnomalyDataset(config.root_file_path, BACKGROUND_TREES, label=0, branch_filter="inclusive_strange_channels", target_value=20)
    train_signal_dset = AnomalyDataset(config.root_file_path, BACKGROUND_TREES, label=1, branch_filter="inclusive_strange_channels", target_value=22)
    train_dset = ConcatDataset([train_background_dset, train_signal_dset])
    
    num_bg = len(train_background_dset)
    num_sig = len(train_signal_dset)
    print(f"Background events: {num_bg}, Signal events: {num_sig}")
    
    # Create weights for balanced sampling
    sample_weights = [1.0 / num_bg if i < num_bg else 1.0 / num_sig for i in range(len(train_dset))]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dset), replacement=True)
    
    train_loader = DataLoader(
        train_dset, batch_size=config.batch_size, collate_fn=collate_fn_classifier,
        sampler=sampler, num_workers=0, shuffle=False
    )

    
    test_background_dset = AnomalyDataset(config.root_file_path, BACKGROUND_TREES, label=0, branch_filter="inclusive_strange_channels", target_value=20)
    test_signal_dset = AnomalyDataset(config.root_file_path, BACKGROUND_TREES, label=1, branch_filter="inclusive_strange_channels", target_value=22)
    test_dset = ConcatDataset([test_background_dset, test_signal_dset])
    
    sample_size = min(config.test_sample_size, len(test_dset))
    print(f"Using a random validation subset of {sample_size} events.")
    test_indices = np.random.choice(len(test_dset), sample_size, replace=False)
    val_subset = Subset(test_dset, test_indices)
    
    test_loader = DataLoader(
        val_subset, batch_size=config.batch_size, collate_fn=collate_fn_classifier,
        num_workers=0, shuffle=False
    )

    # Define model parameters
    in_channels = 4  # From dataset: norm_rows, norm_cols, log_adc, norm_density
    out_channels = 1  # Binary classification
    net = MinkUNetClassifier(
        in_channels=in_channels,
        out_channels=out_channels,
        dimension=DIMENSION
    ).to(device)

    print("\n--- Model Architecture (MinkUNet Classifier) ---")
    print(net)
    print("--------------------------------------------------")

    if config.mode == "train":
        optimizer = Adam(net.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.max_iter)
        criterion = nn.BCEWithLogitsLoss()

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
            
            logits = net(input).F.view(-1)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % 1 == 0:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                acc = (preds == labels).float().mean().item() * 100
                print(f"Iteration: {i}, Classification Loss: {loss.item():.4f}, Batch Accuracy: {acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")
    elif config.mode == "evaluate":
        net.load_state_dict(torch.load(config.weights))
        net.eval()
        auc = test(net, test_loader, device, 0, visualize=True)
        print(f"Evaluation AUC: {auc:.4f}")

# ==============================================================================
# ## --- Entry Point --- ##
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_file_path", type=str, required=True)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--val_freq", type=int, default=250)
    parser.add_argument("--test_sample_size", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=16, help="Smaller batch size for larger model")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weights", type=str, default="minkunet_classifier.pth")
    parser.add_argument("--mode", type=str, default="evaluate", choices=["train", "evaluate"])
    config = parser.parse_args()
    main(config)