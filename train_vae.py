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

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc

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
        logging.info(f"Loaded {len(self.events)} event indices for phase '{self.phase}'.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        flattened_image = tree['detector_image_w'].array(entry_start=entry_index, entry_stop=entry_index + 1, library="np")[0]
        rows, cols = np.where(flattened_image.reshape(512, 512) > 0)
        if rows.size < 10: return None 
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
    # In evaluation mode, we don't need infinite samplers or shuffling
    is_train = phase == 'train'
    dset = ImageDataset(phase, config=config)
    return torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        sampler=InfSampler(dset, shuffle=True) if is_train else None,
        shuffle=False if is_train else False, # Don't shuffle test data
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
# ## --- MODEL ARCHITECTURE (Variational Autoencoder) --- ##
# ==============================================================================

class Encoder(nn.Module):
    CHANNELS = [1, 16, 32, 64, 128, 256]
    def __init__(self, D=2, latent_dim=256):
        super().__init__()
        ch = self.CHANNELS
        self.net = nn.Sequential(
            self.make_block(ch[0], ch[1], D),
            self.make_block(ch[1], ch[2], D),
            self.make_block(ch[2], ch[3], D),
            self.make_block(ch[3], ch[4], D),
            self.make_block(ch[4], ch[5], D),
        )
        self.fc_mu = ME.MinkowskiConvolution(ch[5], latent_dim, kernel_size=1, dimension=D)
        self.fc_log_var = ME.MinkowskiConvolution(ch[5], latent_dim, kernel_size=1, dimension=D)

    def make_block(self, in_channels, out_channels, D):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
        )

    def forward(self, x):
        x = self.net(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    CHANNELS = [256, 128, 64, 32, 16, 8]
    def __init__(self, D=2, latent_dim=256):
        super().__init__()
        ch = [latent_dim] + self.CHANNELS[1:]
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
        if self.training: # In training mode, we generate targets for loss calculation
            # Initial classification and pruning
            out_cls.append(self.cls_layers[0](out))
            target = self.get_target(out, target_key)
            targets.append(target)
            keep = target
            if self.pruning is not None and keep.sum() > 0:
                out = self.pruning(out, keep)

            # Upsampling blocks
            for i, block in enumerate(self.blocks):
                out = block(out)
                out_cls.append(self.cls_layers[i+1](out))
                target = self.get_target(out, target_key)
                targets.append(target)
                keep = target
                if self.pruning is not None and keep.sum() > 0:
                    out = self.pruning(out, keep)
        else: # In evaluation mode, we don't need to generate targets
            # Initial classification and pruning
            out_cls.append(self.cls_layers[0](out))
            keep = (out_cls[-1].F.squeeze() > 0)
            if self.pruning is not None and keep.sum() > 0:
                out = self.pruning(out, keep)

            # Upsampling blocks
            for i, block in enumerate(self.blocks):
                out = block(out)
                out_cls.append(self.cls_layers[i+1](out))
                keep = (out_cls[-1].F.squeeze() > 0)
                if self.pruning is not None and keep.sum() > 0:
                    out = self.pruning(out, keep)

        return out_cls, targets

def weights_init(m):
    if isinstance(m, ME.MinkowskiConvolution):
        ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
    if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

class VAE(nn.Module):
    def __init__(self, D=2, latent_dim=256):
        super().__init__()
        logging.info(f"Initializing VAE with D={D}, latent_dim={latent_dim}")
        self.encoder = Encoder(D=D, latent_dim=latent_dim)
        self.decoder = Decoder(D=D, latent_dim=latent_dim)
        self.apply(weights_init)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var.F)
        eps = torch.randn_like(std)
        return ME.SparseTensor(
            features=mu.F + eps * std,
            coordinate_map_key=mu.coordinate_map_key,
            coordinate_manager=mu.coordinate_manager
        )

    def forward(self, sinput, gen_targets=True):
        target_key = sinput.coordinate_map_key
        mu, log_var = self.encoder(sinput)

        if self.training:
            z = self.reparameterize(mu, log_var)
        else: # During inference, we can use the mean for a deterministic output
            z = mu

        # The decoder's forward pass needs to know whether to generate targets
        self.decoder.training = gen_targets
        out_cls, targets = self.decoder(z, target_key)
        return out_cls, targets, mu, log_var

# ==============================================================================
# ## --- MAIN SCRIPT LOGIC --- ##
# ==============================================================================

def train(net, dataloader, device, config):
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    crit = nn.BCEWithLogitsLoss()
    net.train()

    train_iter = iter(dataloader)
    logging.info("Starting Variational Autoencoder training...")
    
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

        out_cls, targets, mu, log_var = net(sin, gen_targets=True)
        
        bce_loss = 0
        for out, target in zip(out_cls, targets):
            bce_loss += crit(out.F.squeeze(), target.float())
        bce_loss /= len(out_cls)
        
        kl_div = -0.5 * torch.sum(1 + log_var.F - mu.F.pow(2) - log_var.F.exp())
        kl_div /= mu.F.shape[0]

        loss = bce_loss + config.kl_weight * kl_div
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        if i % config.stat_freq == 0:
            logging.info(
                f"Iter: {i}, Loss: {loss.item():.4f}, "
                f"BCE: {bce_loss.item():.4f}, KL: {kl_div.item():.4f}"
            )

        if i > 0 and i % config.val_freq == 0:
            logging.info(f"Saving weights to {config.weights}")
            torch.save({'state_dict': net.state_dict()}, config.weights)
            scheduler.step()

# ==============================================================================
# ## --- NEW: SOPHISTICATED EVALUATION AND PLOTTING --- ##
# ==============================================================================

def plot_roc_curve(bg_scores, sig_scores, score_name, output_dir):
    """
    Calculates and plots the Receiver Operating Characteristic (ROC) curve and AUC.

    Args:
        bg_scores (list or np.array): Anomaly scores for background events.
        sig_scores (list or np.array): Anomaly scores for signal events.
        score_name (str): The name of the score being evaluated (e.g., 'bce', 'total').
        output_dir (str): The directory where the plot will be saved.
    """
    # Combine scores and create true labels.
    # By convention, background is the "negative" class (0) and signal is the "positive" class (1).
    y_true = np.concatenate([np.zeros(len(bg_scores)), np.ones(len(sig_scores))])
    y_scores = np.concatenate([bg_scores, sig_scores])

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Efficiency)')
    plt.ylabel('True Positive Rate (Signal Efficiency)')
    plt.title(f'Receiver Operating Characteristic ({score_name.upper()})')
    plt.legend(loc="lower right")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    save_path = os.path.join(output_dir, f"roc_curve_{score_name}.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved {score_name.upper()} ROC curve to {save_path} with AUC: {roc_auc:.4f}")

def calculate_scores(net, dataloader, device, config):
    """Calculates three VAE anomaly scores for all events in a dataloader."""
    net.eval()
    crit = nn.BCEWithLogitsLoss(reduction='none') # Get per-pixel loss
    
    all_scores = {
        'bce': [],
        'kl': [],
        'total': []
    }

    with torch.no_grad():
        for i, data_dict in enumerate(dataloader):
            if config.max_eval_batches is not None and i >= config.max_eval_batches:
                logging.info(f"Reached max_eval_batches limit of {config.max_eval_batches}. Stopping.")
                break

            if (i+1) % 10 == 0: # NEW: Add some progress logging
                logging.info(f"Processing batch {i+1}/{len(dataloader)}...")

            if data_dict is None: continue

            sin = ME.SparseTensor(
                features=data_dict["feats"].to(device),
                coordinates=data_dict["coords"].to(device)
            )
            
            # We need to generate targets to calculate BCE loss, but not for KL
            out_cls, targets, mu, log_var = net(sin, gen_targets=True)
            
            # Get batch index from coordinate tensor (column 0)
            batch_indices = mu.C[:, 0]
            
            # --- Calculate scores per event in the batch ---
            for b_idx in torch.unique(batch_indices):
                # 1. Reconstruction Loss (BCE)
                bce_per_event = 0
                for out, target in zip(out_cls, targets):
                    # Mask to select features/targets for the current event
                    event_mask = out.C[:, 0] == b_idx
                    if torch.sum(event_mask) == 0: continue
                    
                    bce_pixels = crit(out.F[event_mask].squeeze(), target.float()[event_mask])
                    bce_per_event += torch.mean(bce_pixels).item() # Average over pixels
                bce_per_event /= len(out_cls) # Average over layers
                all_scores['bce'].append(bce_per_event)

                # 2. KL Divergence
                event_mask_kl = mu.C[:, 0] == b_idx
                kl_per_event = -0.5 * torch.sum(1 + log_var.F[event_mask_kl] - mu.F[event_mask_kl].pow(2) - log_var.F[event_mask_kl].exp())
                all_scores['kl'].append(kl_per_event.item())

                # 3. Total Loss (ELBO)
                total_loss = bce_per_event + config.kl_weight * kl_per_event.item()
                all_scores['total'].append(total_loss)

    return all_scores



def evaluate_and_plot_scores(net, bg_loader, sig_loader, device, config):
    """
    Calculates anomaly scores for background and signal samples,
    plots comparison histograms, and generates ROC curves with AUC values.
    """
    logging.info("Calculating scores for BACKGROUND events...")
    bg_scores = calculate_scores(net, bg_loader, device, config)

    logging.info("Calculating scores for SIGNAL events...")
    sig_scores = calculate_scores(net, sig_loader, device, config)

    output_dir = "anomaly_comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving anomaly comparison plots to '{output_dir}/'")

    # --- Plotting ---
    for score_name in ['bce', 'kl', 'total']:
        # Ensure there are scores to plot
        if not bg_scores[score_name] or not sig_scores[score_name]:
            logging.warning(f"Skipping plots for '{score_name}' due to empty score list.")
            continue
            
        # --- 1. Plot Histograms (existing code) ---
        plt.figure(figsize=(10, 6))
        min_val = min(np.min(bg_scores[score_name]), np.min(sig_scores[score_name]))
        max_val = max(np.max(bg_scores[score_name]), np.max(sig_scores[score_name]))

        plt.hist(bg_scores[score_name], bins=50, range=(min_val, max_val),
                 alpha=0.7, label='Background', density=True, color='blue')
        plt.hist(sig_scores[score_name], bins=50, range=(min_val, max_val),
                 alpha=0.7, label='Signal', density=True, color='red')

        plt.xlabel(f"Anomaly Score ({score_name.upper()})")
        plt.ylabel("Normalized Event Count")
        plt.title(f"Anomaly Score Distribution ({score_name.upper()})")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.yscale('log')

        save_path = os.path.join(output_dir, f"anomaly_scores_{score_name}.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved {score_name.upper()} comparison plot to {save_path}")

        # --- 2. NEW: Generate ROC Curve and AUC ---
        plot_roc_curve(
            bg_scores[score_name],
            sig_scores[score_name],
            score_name,
            output_dir
        )

# ==============================================================================
# ## --- NEW: LATENT SPACE VISUALIZATION --- ##
# ==============================================================================

# ==============================================================================
# ## --- CORRECTED: LATENT SPACE VISUALIZATION --- ##
# ==============================================================================

def evaluate_and_visualize_latent_space(net, bg_loader, sig_loader, device, config):
    """
    Extracts the latent space representations (mu) for signal and background,
    aggregates them to get one vector per event, reduces dimensionality
    with t-SNE, and plots the result.
    """
    net.eval()
    latent_vectors = []
    labels = []

    # --- Helper function to process a batch ---
    def process_batch(data_dict, label_name):
        if data_dict is None: return
        sin = ME.SparseTensor(
            features=data_dict["feats"].to(device),
            coordinates=data_dict["coords"].to(device)
        )
        mu, _ = net.encoder(sin)
        
        # Get coordinates and features, move to CPU for numpy
        coords = mu.C.cpu().numpy()
        feats = mu.F.cpu().numpy()
        
        # Get the unique event IDs present in this batch
        # The batch index is the first column of the coordinates
        batch_event_ids = np.unique(coords[:, 0])
        
        # For each event, aggregate its latent vectors
        for event_id in batch_event_ids:
            # Find all points/vectors belonging to this single event
            event_mask = (coords[:, 0] == event_id)
            event_vectors = feats[event_mask]
            
            # Aggregate into a single vector by taking the mean
            aggregated_vector = np.mean(event_vectors, axis=0)
            
            # Now we have one vector per event
            latent_vectors.append(aggregated_vector)
            labels.append(label_name)

    logging.info("Extracting latent space for BACKGROUND events...")
    with torch.no_grad():
        for i, data_dict in enumerate(bg_loader):
            if config.max_eval_batches is not None and i >= config.max_eval_batches: break
            process_batch(data_dict, 'Background')

    logging.info("Extracting latent space for SIGNAL events...")
    with torch.no_grad():
        for i, data_dict in enumerate(sig_loader):
            if config.max_eval_batches is not None and i >= config.max_eval_batches: break
            process_batch(data_dict, 'Signal')

    if not latent_vectors:
        logging.error("No latent vectors were extracted. Cannot create plot.")
        return

    # Combine all batch results into a single numpy array
    latent_vectors = np.array(latent_vectors)
    
    logging.info(f"Total aggregated latent vectors extracted: {len(latent_vectors)}. Performing t-SNE...")

    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(latent_vectors)

    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    df['label'] = labels # Now the lengths will match!

    logging.info("Plotting t-SNE of latent space...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette={"Background": "blue", "Signal": "red"},
        data=df,
        legend="full",
        alpha=0.6
    )
    plt.title('t-SNE Projection of the Latent Space (Aggregated per Event)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_dir = "anomaly_comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "latent_space_tsne.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved latent space t-SNE plot to {save_path}")

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
            out_cls, _, _, _ = net(sin)
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
    parser.add_argument("--root_file_path", type=str, required=True, help="Path to the ROOT file containing event data.")
    parser.add_argument("--weights", type=str, default="neutrino_vae_final.pth", help="Path to saved model weights.")
    
    # MODIFIED: Arguments to specify signal and background tree names
    parser.add_argument("--background_trees", nargs='+', type=str, default=['mc_inclusive_run1_fhc'], help="Name(s) of the TTree(s) for background events.")
    parser.add_argument("--signal_trees", nargs='+', type=str, default=['mc_strangeness_run1_fhc'], help="Name(s) of the TTree(s) for signal events.")
    
    # Training Config
    parser.add_argument("--max_iter", type=int, default=20000)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--kl_weight", type=float, default=1e-6, help="Weight for the KL divergence term in the VAE loss.")
    
    # Logging and Validation
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--stat_freq", type=int, default=100)
    
    # Execution Config
    parser.add_argument("--num_workers", type=int, default=4)
    # MODIFIED: Default mode is now test, not train
    parser.add_argument("--train", dest='train_mode', action='store_true', help="Run in training mode.")
    parser.add_argument("--test", dest='train_mode', action='store_false', help="Run in testing/evaluation mode.")
    parser.add_argument("--max_eval_batches", type=int, default=30, help="Limit the number of batches to process during evaluation for a quick test.")
    parser.add_argument("--max_visualization", type=int, default=10, help="Maximum number of visualizations to generate during evaluation.")
    parser.set_defaults(train_mode=False)

    config = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    latent_dim = 64
    net = VAE(D=2, latent_dim=latent_dim)
    net.to(device)

    if config.train_mode:
        # For training, use the background trees
        config.tree_names = config.background_trees
        dataloader = make_data_loader("train", config)
        train(net, dataloader, device, config)
    else: # Evaluation mode
        if not os.path.exists(config.weights):
            raise FileNotFoundError(f"Weights file not found: {config.weights}")
        logging.info(f"Loading weights from {config.weights}")
        checkpoint = torch.load(config.weights, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        
        # --- Create separate dataloaders for background and signal ---
        # Create a config object for background
        config_bg = argparse.Namespace(**vars(config))
        config_bg.tree_names = config.background_trees
        bg_loader = make_data_loader("test_bg", config_bg)
        
        # Create a config object for signal
        config_sig = argparse.Namespace(**vars(config))
        config_sig.tree_names = config.signal_trees
        sig_loader = make_data_loader("test_sig", config_sig)

        # Run the sophisticated evaluation
        evaluate_and_plot_scores(net, bg_loader, sig_loader, device, config)
        #evaluate_and_visualize_latent_space(net, bg_loader, sig_loader, device, config)
        visualize(net, bg_loader, device, config)
