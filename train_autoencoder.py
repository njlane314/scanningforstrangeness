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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import MinkowskiEngine as ME

def seed_all(random_seed):
    """Sets the random seeds for reproducibility."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

# ==============================================================================
# ## --- VAE Model Implementation --- ##
# ==============================================================================

class Encoder(nn.Module):
    """A simple convolutional encoder for sparse 2D data."""
    def __init__(self, in_channels=1, latent_dim=64, D=2):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 16, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(16, 32, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiGlobalPooling(),
        )
        self.mean_linear = ME.MinkowskiLinear(64, latent_dim)
        self.log_var_linear = ME.MinkowskiLinear(64, latent_dim)

    def forward(self, x):
        x = self.network(x)
        mean = self.mean_linear(x)
        log_var = self.log_var_linear(x)
        return mean, log_var

class Decoder(nn.Module):
    """A simple convolutional decoder for sparse 2D data."""
    def __init__(self, out_channels=1, latent_dim=64, D=2):
        super(Decoder, self).__init__()
        self.network = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(latent_dim, 64, kernel_size=4, stride=1, dimension=D),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolutionTranspose(64, 32, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolutionTranspose(32, 16, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolutionTranspose(16, out_channels, kernel_size=3, stride=2, dimension=D),
        )

    def forward(self, x, target_key):
        return self.network(x)

class VAE(nn.Module):
    """A Variational Autoencoder that combines the Encoder and Decoder."""
    def __init__(self, in_channels=1, latent_dim=64, D=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, D)
        self.decoder = Decoder(in_channels, latent_dim, D)

    def reparameterize(self, mean, log_var):
        # The reparameterization trick to allow backpropagation
        std = torch.exp(0.5 * log_var.F)
        eps = torch.randn_like(std)
        return ME.SparseTensor(mean.F + eps * std, coordinate_map_key=mean.coordinate_map_key, coordinate_manager=mean.coordinate_manager)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decoder(z, x.coordinate_map_key)
        return reconstructed, mean, log_var

class TripleVAE(nn.Module):
    """Three parallel VAEs for the U, V, and W planes."""
    def __init__(self, in_channels=1, latent_dim=64, D=2):
        super(TripleVAE, self).__init__()
        self.vae_u = VAE(in_channels, latent_dim, D)
        self.vae_v = VAE(in_channels, latent_dim, D)
        self.vae_w = VAE(in_channels, latent_dim, D)

    def forward(self, x_u, x_v, x_w):
        recon_u, mean_u, log_var_u = self.vae_u(x_u)
        recon_v, mean_v, log_var_v = self.vae_v(x_v)
        recon_w, mean_w, log_var_w = self.vae_w(x_w)
        return (recon_u, mean_u, log_var_u), (recon_v, mean_v, log_var_v), (recon_w, mean_w, log_var_w)

# ==============================================================================
# ## --- Data Handling and Training Logic --- ##
# ==============================================================================

class NeutrinoDataset(Dataset):
    """Loads data for three detector planes."""
    def __init__(self, root_file_path, tree_names, image_branches, label_branch, image_shape=(512, 512)):
        self.root_file_path = root_file_path
        self.image_branches = image_branches
        self.label_branch = label_branch
        self.image_shape = image_shape
        self.file_handle = None
        self._scan_root_file(tree_names, label_branch)

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self, tree_names, label_branch):
        logging.info(f"Scanning ROOT file: {self.root_file_path}...")
        self.events = []
        all_labels = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file:
                    logging.warning(f"TTree '{tree_name}' not found. Skipping.")
                    continue
                tree = file[tree_name]
                num_entries = tree.num_entries
                logging.info(f"Scanning TTree '{tree_name}' with {num_entries} entries.")
                labels_array = tree[label_branch].array(library="np")
                for i in range(num_entries):
                    self.events.append((tree_name, i))
                all_labels.append(labels_array)

        if not self.events:
            raise RuntimeError("No events loaded. Check TTree names and file content.")
        
        raw_labels = np.concatenate(all_labels)
        self.labels = np.isin(raw_labels.astype(int), [10, 11]).astype(int).flatten()
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        logging.info(f"Full dataset label distribution: {dict(zip(unique_labels, counts))}")
        logging.info(f"Scan complete. Found {len(self.events)} total events.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        true_label = self.labels[idx]
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        
        plane_data = []
        for branch_name in self.image_branches:
            flattened_image = tree[branch_name].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
            image_grid = np.array(flattened_image, dtype=np.float32).reshape(self.image_shape)
            rows, cols = np.where(image_grid > 1.0)
            if rows.size == 0:
                coords = torch.empty((0, 2), dtype=torch.int32)
                feats = torch.empty((0, 1), dtype=torch.float32)
            else:
                coords = torch.from_numpy(np.stack([rows, cols], axis=1)).int()
                feats = torch.from_numpy(np.log(image_grid[rows, cols])[:, np.newaxis]).float()
            plane_data.append((coords, feats))
        
        return tuple(plane_data), torch.tensor([true_label], dtype=torch.long)

def vae_collate_fn(batch):
    """Collate function for the VAE."""
    filtered_batch = [item for item in batch if all(c.shape[0] > 0 for c, f in item[0])]
    if not filtered_batch:
        return None

    inputs, labels = zip(*filtered_batch)
    u_coords, u_feats = ME.utils.sparse_collate(*zip(*[p[0] for p in inputs]))
    v_coords, v_feats = ME.utils.sparse_collate(*zip(*[p[1] for p in inputs]))
    w_coords, w_feats = ME.utils.sparse_collate(*zip(*[p[2] for p in inputs]))
    collated_labels = torch.cat(labels)
    return (u_coords, u_feats), (v_coords, v_feats), (w_coords, w_feats), collated_labels

def train(net, data_loader, device, config):
    """Training loop for the VAE."""
    net.train()
    optimiser = optim.Adam(net.parameters(), lr=config.lr)
    logging.info(f"--- Starting VAE Training for {config.max_steps} steps ---")
    train_iter = iter(data_loader)

    for i in range(config.max_steps):
        optimiser.zero_grad()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(data_loader)
            batch = next(train_iter)
        
        if batch is None: continue

        (u_coords, u_feats), (v_coords, v_feats), (w_coords, w_feats), labels = batch

        s_in_u = ME.SparseTensor(u_feats, coordinates=u_coords, device=device)
        s_in_v = ME.SparseTensor(v_feats, coordinates=v_coords, device=device)
        s_in_w = ME.SparseTensor(w_feats, coordinates=w_coords, device=device)
        
        (recon_u, mean_u, log_var_u), (recon_v, mean_v, log_var_v), (recon_w, mean_w, log_var_w) = net(s_in_u, s_in_v, s_in_w)

        # Reconstruction Loss on the union of coordinates
        diff_u = recon_u - s_in_u
        diff_v = recon_v - s_in_v
        diff_w = recon_w - s_in_w
        recon_loss = torch.sum(torch.abs(diff_u.F)) + torch.sum(torch.abs(diff_v.F)) + torch.sum(torch.abs(diff_w.F))

        # KL Divergence
        kld = -0.5 * torch.sum(1 + log_var_u.F - mean_u.F.pow(2) - log_var_u.F.exp())
        kld += -0.5 * torch.sum(1 + log_var_v.F - mean_v.F.pow(2) - log_var_v.F.exp())
        kld += -0.5 * torch.sum(1 + log_var_w.F - mean_w.F.pow(2) - log_var_w.F.exp())

        # --- FIX: Get batch size from the collated labels tensor ---
        batch_size = len(labels)
        loss = (recon_loss + kld) / batch_size
        
        loss.backward()
        optimiser.step()

        if (i + 1) % config.stat_freq == 0:
            logging.info(f"Iter: {i + 1}, Loss: {loss.item():.3e}")

def test_anomaly_detection(net, data_loader, device):
    """Evaluates the VAE's ability to detect anomalies."""
    net.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue
            
            (u_coords, u_feats), (v_coords, v_feats), (w_coords, w_feats), labels = batch

            s_in_u = ME.SparseTensor(u_feats, coordinates=u_coords, device=device)
            s_in_v = ME.SparseTensor(v_feats, coordinates=v_coords, device=device)
            s_in_w = ME.SparseTensor(w_feats, coordinates=w_coords, device=device)
            
            (recon_u, mean_u, log_var_u), (recon_v, mean_v, log_var_v), (recon_w, mean_w, log_var_w) = net(s_in_u, s_in_v, s_in_w)

            # --- FIX: Get batch size from the collated labels tensor ---
            batch_size = len(labels)

            # 1. Reconstruction Loss per event
            diff_u = recon_u - s_in_u
            diff_v = recon_v - s_in_v
            diff_w = recon_w - s_in_w

            point_errors_u = torch.abs(diff_u.F).sum(dim=1)
            point_errors_v = torch.abs(diff_v.F).sum(dim=1)
            point_errors_w = torch.abs(diff_w.F).sum(dim=1)

            event_recon_loss_u = torch.bincount(diff_u.C[:, 0].long(), weights=point_errors_u, minlength=batch_size)
            event_recon_loss_v = torch.bincount(diff_v.C[:, 0].long(), weights=point_errors_v, minlength=batch_size)
            event_recon_loss_w = torch.bincount(diff_w.C[:, 0].long(), weights=point_errors_w, minlength=batch_size)
            
            total_recon_loss = event_recon_loss_u + event_recon_loss_v + event_recon_loss_w

            # 2. KLD per event
            kld_u = -0.5 * torch.sum(1 + log_var_u.F - mean_u.F.pow(2) - log_var_u.F.exp(), dim=1)
            kld_v = -0.5 * torch.sum(1 + log_var_v.F - mean_v.F.pow(2) - log_var_v.F.exp(), dim=1)
            kld_w = -0.5 * torch.sum(1 + log_var_w.F - mean_w.F.pow(2) - log_var_w.F.exp(), dim=1)
            
            total_kld = kld_u + kld_v + kld_w

            # Anomaly score is the total loss (negative ELBO)
            anomaly_scores = total_recon_loss + total_kld
            
            all_scores.extend(anomaly_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores)
    auc = metrics.auc(fpr, tpr)
    
    logging.info(f"Anomaly Detection AUC: {auc:.4f}")
    return auc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(description="MinkowskiEngine VAE Anomaly Detection")
    parser.add_argument("--root_file_path", type=str, required=True, help="Path to the input ROOT file.")
    parser.add_argument("--tree_names", nargs='+', required=True, help="List of TTree names to process.")
    parser.add_argument("--image_branches", nargs=3, default=["detector_image_u", "detector_image_v", "detector_image_w"], help="Three branch names for the U, V, and W detector images.")
    parser.add_argument("--label_branch", type=str, default="inclusive_strange_channels", help="Branch name for the event labels.")
    parser.add_argument("--weight_branch", type=str, default="base_event_weight", help="Dummy argument, not used.")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stat_freq", type=int, default=1)
    parser.add_argument("--weights", type=str, default="neutrino_vae_model.pth")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--test_split_size", type=int, default=1000, help="Number of events to use for the test set.")
    config = parser.parse_args()

    logging.info(f"Running with configuration: {config}")
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    full_dataset = NeutrinoDataset(
        root_file_path=config.root_file_path,
        tree_names=config.tree_names,
        image_branches=config.image_branches,
        label_branch=config.label_branch,
    )
    
    background_indices = np.where(full_dataset.labels == 0)[0]
    train_dataset = Subset(full_dataset, background_indices)
    
    all_indices = np.arange(len(full_dataset))
    _, test_indices = train_test_split(
        all_indices,
        test_size=config.test_split_size,
        random_state=config.seed,
        stratify=full_dataset.labels
    )
    test_dataset = Subset(full_dataset, test_indices)

    logging.info(f"Training on {len(train_dataset)} background events.")
    logging.info(f"Testing on {len(test_dataset)} mixed events.")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=vae_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=vae_collate_fn)

    net = TripleVAE().to(device)
    logging.info(f"Model: {net}")
    
    train(net, train_loader, device, config)
    
    logging.info("--- Final Test Evaluation ---")
    auc = test_anomaly_detection(net, test_loader, device)
    logging.info(f"Final Anomaly Detection AUC on test set: {auc:.4f}")

    torch.save(net.state_dict(), config.weights)
    logging.info(f"Model saved to {config.weights}")