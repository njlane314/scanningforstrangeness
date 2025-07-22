import os
import argparse
import logging
import numpy as np
import uproot

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter

try:
    import MinkowskiEngine as ME
except ImportError:
    raise ImportError(
        "MinkowskiEngine not found. Please install it, e.g., "
        "by following the instructions at https://github.com/NVIDIA/MinkowskiEngine"
    )

# ==============================================================================
# ## --- DATA LOADING --- ##
# ==============================================================================

class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, root_file_path, tree_names, is_anomaly):
        self.root_file_path = root_file_path
        self.tree_names = tree_names
        self.is_anomaly = is_anomaly
        self.file_handle = None
        self.events = []
        self._scan_root_file()

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self):
        logging.info(f"Scanning ROOT file: {self.root_file_path} for trees: {self.tree_names}")
        with uproot.open(self.root_file_path) as file:
            for tree_name in self.tree_names:
                if tree_name not in file:
                    logging.warning(f"TTree '{tree_name}' not found.")
                    continue
                num_entries = file[tree_name].num_entries
                logging.info(f"Found {num_entries} events in TTree '{tree_name}'.")
                for i in range(num_entries):
                    self.events.append((tree_name, i))
        logging.info(f"Loaded {len(self.events)} event indices for anomaly type: {self.is_anomaly}.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        flattened_image = tree['detector_image_w'].array(entry_start=entry_index, entry_stop=entry_index + 1, library="np")[0]
        rows, cols = np.where(flattened_image.reshape(512, 512) > 0)
        
        coords_raw = np.stack([rows, cols], axis=1)

        mask = (coords_raw[:, 0] >= 0) & (coords_raw[:, 0] < 512) & \
               (coords_raw[:, 1] >= 0) & (coords_raw[:, 1] < 512)
        coords_filtered = coords_raw[mask]

        if coords_filtered.shape[0] < 10: 
            return None
            
        coords = coords_filtered.astype(np.int32)
        feats = np.ones((len(coords), 1), dtype=np.float32)
        
        return (coords, feats, self.is_anomaly)

def collate_fn_one_class(list_data):
    list_data = [data for data in list_data if data is not None]
    if not list_data:
        return None
    
    coords_list, feats_list, labels_list = zip(*list_data)
    
    batched_coords = ME.utils.batched_coordinates(coords_list)
    batched_feats = torch.from_numpy(np.vstack(feats_list))
    labels = torch.LongTensor(labels_list)
    
    return {
        "coords": batched_coords,
        "feats": batched_feats,
        "labels": labels
    }


def make_data_loader(phase, config, tree_names, is_anomaly):
    is_train = phase == 'train'
    dset = AnomalyDataset(
        root_file_path=config.root_file_path,
        tree_names=tree_names,
        is_anomaly=is_anomaly
    )
    
    return torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        collate_fn=collate_fn_one_class,
        num_workers=config.num_workers,
        sampler=None,
        shuffle=is_train,
        pin_memory=True,
        drop_last=is_train
    )

# ==============================================================================
# ## --- MODEL ARCHITECTURE (FCDD with UNet) --- ##
# ==============================================================================

class MinkUNet(nn.Module):
    def __init__(self, in_channels, out_channels, D=2):
        super(MinkUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D

        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # Encoder
        self.conv1 = ME.MinkowskiConvolution(in_channels, 32, kernel_size=3, stride=1, dimension=D, bias=False)
        self.norm1 = ME.MinkowskiBatchNorm(32)
        self.pool1 = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)

        self.conv2 = ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=1, dimension=D, bias=False)
        self.norm2 = ME.MinkowskiBatchNorm(64)
        self.pool2 = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)

        # Bottleneck
        self.conv3 = ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=1, dimension=D, bias=False)
        self.norm3 = ME.MinkowskiBatchNorm(128)

        # Decoder
        self.conv_tr2 = ME.MinkowskiConvolutionTranspose(128, 64, kernel_size=2, stride=2, dimension=D, bias=False)
        self.norm_tr2 = ME.MinkowskiBatchNorm(64)

        self.conv_tr1 = ME.MinkowskiConvolutionTranspose(64 + 64, 32, kernel_size=2, stride=2, dimension=D, bias=False)
        self.norm_tr1 = ME.MinkowskiBatchNorm(32)
        
        # Final layer to produce the anomaly score at the original resolution
        self.final_conv = ME.MinkowskiConvolution(32 + 32, out_channels, kernel_size=1, dimension=D, bias=False)

    def forward(self, x):
        # Encoder
        out_s1 = self.relu(self.norm1(self.conv1(x)))
        out_s2 = self.pool1(out_s1)
        
        out_s2 = self.relu(self.norm2(self.conv2(out_s2)))
        out_s4 = self.pool2(out_s2)
        
        # Bottleneck
        out_s4 = self.relu(self.norm3(self.conv3(out_s4)))

        # Decoder
        out = self.relu(self.norm_tr2(self.conv_tr2(out_s4)))
        out = ME.cat(out, out_s2) # Skip connection
        
        out = self.relu(self.norm_tr1(self.conv_tr1(out)))
        out = ME.cat(out, out_s1) # Skip connection

        return self.final_conv(out)


class FCDDLoss(nn.Module):
    def __init__(self):
        super(FCDDLoss, self).__init__()

    def forward(self, anomaly_map, labels):
        total_loss = 0.0
        batch_size = labels.shape[0]
        
        coords = anomaly_map.C
        features = anomaly_map.F
        
        huber_losses = torch.sqrt(features**2 + 1) - 1
        
        for i in range(batch_size):
            event_mask = (coords[:, 0] == i)
            
            if not torch.any(event_mask):
                continue
            
            mean_huber_loss_i = torch.mean(huber_losses[event_mask])
            
            if labels[i] == 0:  # Nominal
                image_loss = mean_huber_loss_i
            else:  # Anomalous
                image_loss = -torch.log(1 - torch.exp(-mean_huber_loss_i) + 1e-9)
                
            total_loss += image_loss
            
        return total_loss / batch_size

# ==============================================================================
# ## --- SCRIPT LOGIC --- ##
# ==============================================================================

def train(net, dataloader, device, config):
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    criterion = FCDDLoss()
    net.train()

    train_iter = iter(dataloader)
    logging.info("Starting FCDD training...")
    
    for i in range(config.max_iter):
        try:
            data_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            data_dict = next(train_iter)
            
        if data_dict is None: continue

        optimizer.zero_grad()
        
        sin = ME.SparseTensor(
            features=data_dict["feats"].to(device), 
            coordinates=data_dict["coords"].to(device)
        )
        labels = data_dict["labels"].to(device)
        
        anomaly_map = net(sin)
        
        loss = criterion(anomaly_map, labels)
        
        loss.backward()
        optimizer.step()

        if i % config.stat_freq == 0:
            logging.info(f"Iter: {i}, Loss: {loss.item():.4f}")

        if i > 0 and i % config.val_freq == 0:
            logging.info(f"Saving weights to {config.weights}")
            torch.save({'state_dict': net.state_dict()}, config.weights)
            scheduler.step()

# ==============================================================================
# ## --- EVALUATION AND PLOTTING --- ##
# ==============================================================================

def plot_roc_curve(bg_scores, sig_scores, output_dir):
    y_true = np.concatenate([np.zeros(len(bg_scores)), np.ones(len(sig_scores))])
    y_scores = np.concatenate([bg_scores, sig_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    save_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved ROC curve to {save_path} with AUC: {roc_auc:.4f}")


def calculate_and_plot_scores(net, bg_loader, sig_loader, device, config):
    net.eval()
    output_dir = "anomaly_comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    def get_scores(dataloader):
        scores = []
        with torch.no_grad():
            for i, data_dict in enumerate(dataloader):
                if config.max_eval_batches is not None and i >= config.max_eval_batches:
                    break
                if data_dict is None: continue
                sin = ME.SparseTensor(features=data_dict["feats"].to(device), coordinates=data_dict["coords"].to(device))
                anomaly_map = net(sin)
                for b_idx in torch.unique(anomaly_map.C[:, 0]):
                    event_mask = anomaly_map.C[:, 0] == b_idx
                    event_score = torch.sum(anomaly_map.F[event_mask]).item()
                    scores.append(event_score)
        return scores

    logging.info("Calculating scores for BACKGROUND events...")
    bg_scores = get_scores(bg_loader)
    logging.info("Calculating scores for SIGNAL events...")
    sig_scores = get_scores(sig_loader)

    plt.figure(figsize=(10, 6))
    if bg_scores and sig_scores:
        min_val = min(np.min(bg_scores), np.min(sig_scores))
        max_val = max(np.max(bg_scores), np.max(sig_scores))
        plt.hist(bg_scores, bins=50, range=(min_val, max_val), alpha=0.7, label='Background (Nominal)', density=True, color='blue')
        plt.hist(sig_scores, bins=50, range=(min_val, max_val), alpha=0.7, label='Signal (Anomaly)', density=True, color='red')
    
    plt.xlabel("Mean Anomaly Score")
    plt.ylabel("Normalized Event Count")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.yscale('log')
    save_path = os.path.join(output_dir, "anomaly_score_distribution.png")
    plt.savefig(save_path)
    plt.close()

    if bg_scores and sig_scores:
        plot_roc_curve(bg_scores, sig_scores, output_dir)


def visualize_heatmaps(net, dataloader, device, config, loader_name=""):
    net.eval()
    output_dir = f"visualizations_{loader_name}"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving '{loader_name}' diagnostic visualizations to '{output_dir}/'")
    
    with torch.no_grad():
        for i, data_dict in enumerate(dataloader):
            if i >= config.max_visualization: break
            if data_dict is None: continue

            sin = ME.SparseTensor(features=data_dict["feats"].to(device), coordinates=data_dict["coords"].to(device))
            anomaly_map_sparse = net(sin)
            
            b_idx = 0 
            
            input_mask = sin.C[:, 0] == b_idx
            if not torch.any(input_mask):
                continue
            
            canvas_shape = (512, 512)
            gt_map = np.zeros(canvas_shape)
            heatmap = np.zeros(canvas_shape)
            
            gt_coords_b = sin.C[input_mask, 1:].cpu().numpy()
            gt_map[gt_coords_b[:, 0], gt_coords_b[:, 1]] = 1
            
            output_mask = anomaly_map_sparse.C[:, 0] == b_idx
            if torch.any(output_mask):
                # The new UNet outputs scores at the original resolution
                out_coords_b = anomaly_map_sparse.C[output_mask, 1:].cpu().numpy()
                raw_scores_b = anomaly_map_sparse.F[output_mask].cpu().numpy()
                
                final_scores_b = np.sqrt(raw_scores_b**2 + 1) - 1
                
                # Directly create a sparse map at the original resolution
                temp_sparse_map = np.zeros(canvas_shape)
                temp_sparse_map[out_coords_b[:, 0], out_coords_b[:, 1]] = final_scores_b.squeeze()

                # Apply a gentle Gaussian filter to make points more visible
                heatmap = gaussian_filter(temp_sparse_map, sigma=2.0)

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Event #{i+1} - Type: {loader_name.upper()}", fontsize=16)
            
            axes[0].imshow(gt_map, cmap='gray')
            axes[0].set_title("Input Sparsity Pattern")

            vmax = max(1e-6, np.max(heatmap))
            im = axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=vmax)
            axes[1].set_title("Anomaly Heatmap")
            
            fig.colorbar(im, ax=axes[1])
            plt.savefig(os.path.join(output_dir, f"heatmap_{i+1}.png"))
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_file_path", type=str, required=True)
    parser.add_argument("--weights", type=str, default="fcdd_model.pth")
    parser.add_argument("--background_trees", nargs='+', type=str, required=True)
    parser.add_argument("--signal_trees", nargs='+', type=str, required=True)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--val_freq", type=int, default=500)
    parser.add_argument("--stat_freq", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_false', dest='train')
    parser.add_argument("--max_eval_batches", type=int, default=10)
    parser.add_argument("--max_visualization", type=int, default=10)
    parser.set_defaults(train=False)

    config = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the new UNet model
    net = MinkUNet(in_channels=1, out_channels=1, D=2)
    net.to(device)

    if config.train:
        bg_dset = AnomalyDataset(config.root_file_path, config.background_trees, is_anomaly=0)
        sig_dset = AnomalyDataset(config.root_file_path, config.signal_trees, is_anomaly=1)
        combined_dset = torch.utils.data.ConcatDataset([bg_dset, sig_dset])
        
        dataloader = torch.utils.data.DataLoader(
            combined_dset,
            batch_size=config.batch_size,
            collate_fn=collate_fn_one_class,
            num_workers=config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        train(net, dataloader, device, config)
    else:
        if not os.path.exists(config.weights):
            raise FileNotFoundError(f"Weights file not found: {config.weights}")
        logging.info(f"Loading weights from {config.weights}")
        checkpoint = torch.load(config.weights, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        bg_loader = make_data_loader("test", config, config.background_trees, is_anomaly=0)
        sig_loader = make_data_loader("test", config, config.signal_trees, is_anomaly=1)

        calculate_and_plot_scores(net, bg_loader, sig_loader, device, config)
        
        visualize_heatmaps(net, bg_loader, device, config, loader_name="background")
        visualize_heatmaps(net, sig_loader, device, config, loader_name="signal")