# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import MinkowskiEngine as ME
import uproot
from sklearn.metrics import classification_report, roc_auc_score

# --- (Parser remains the same) ---
parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--val_freq", type=int, default=100)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--stat_freq", type=int, default=1)
parser.add_argument("--weights", type=str, default="neutrino_vae_vmf.pth")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--root_file_path", type=str, required=True)
parser.add_argument("--tree_names", type=str, required=True)
parser.add_argument("--plane", type=str, default='u')
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--beta", type=float, default=10) # Lower beta is often better for vMF VAEs


class HypersphericalVAE(ME.MinkowskiNetwork):
    CHANNELS = [16, 32, 64, 128]

    def __init__(self, in_channel=1, latent_dim=64, D=2):
        super().__init__(D)
        if latent_dim < 2:
            raise ValueError("Latent dim must be at least 2 for hyperspherical VAE.")
        ch = self.CHANNELS
        self.latent_dim = latent_dim
        
        # == Encoder (Outputs vMF parameters) ==
        self.enc_b1 = self._make_block(in_channel, ch[0], stride=2)
        self.enc_b2 = self._make_block(ch[0], ch[1], stride=2)
        self.enc_b3 = self._make_block(ch[1], ch[2], stride=2)
        self.enc_b4 = self._make_block(ch[2], ch[3], stride=2)
        
        self.global_pool = ME.MinkowskiGlobalPooling()
        # Learns the mean direction vector
        self.fc_mu = nn.Linear(ch[3], latent_dim)
        # Learns the concentration parameter (kappa)
        self.fc_kappa = nn.Linear(ch[3], 1)

        # == Decoder (Takes a point on the hypersphere as input) ==
        self.dec_fc = nn.Linear(latent_dim, ch[3])
        self.dec_b4 = self._make_up_block(ch[3] * 2, ch[2])
        self.dec_b3 = self._make_up_block(ch[2] * 2, ch[1])
        self.dec_b2 = self._make_up_block(ch[1] * 2, ch[0])
        self.dec_b1 = self._make_up_block(ch[0] * 2, ch[0])

        self.final_conv = ME.MinkowskiConvolution(ch[0], in_channel, kernel_size=1, dimension=D)
        self.weight_initialization()

    def _make_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=3, stride=stride, dimension=self.D),
            ME.MinkowskiBatchNorm(out_ch),
            ME.MinkowskiLeakyReLU())

    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_ch, out_ch, kernel_size=2, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(out_ch),
            ME.MinkowskiLeakyReLU())

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, (ME.MinkowskiConvolution, ME.MinkowskiConvolutionTranspose)):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _vmf_sample(self, mu, kappa):
        """
        Reparameterization trick for vMF distribution.
        Based on https://arxiv.org/abs/1805.08528
        """
        w = self._sample_w(kappa)
        v = torch.randn(mu.size(0), self.latent_dim - 1, device=mu.device)
        v = F.normalize(v, p=2, dim=1)

        # Create a vector orthogonal to mu
        u = F.normalize(torch.randn_like(mu), p=2, dim=1)
        proj_mu_u = (mu * u).sum(dim=1, keepdim=True) * mu
        u_ortho = F.normalize(u - proj_mu_u, p=2, dim=1)
        
        # Combine to form the sample
        return w * mu + torch.sqrt(1 - w**2) * u_ortho
    
    def _sample_w(self, kappa):
        """Samples the W component for vMF sampling."""
        dim = self.latent_dim
        b = (dim - 1) / (2 * kappa + torch.sqrt(4 * kappa**2 + (dim - 1)**2))
        x = (1 - b) / (1 + b)
        c = kappa * x + (dim - 1) * torch.log(1 - x**2)

        # Rejection sampling
        while True:
            z = torch.distributions.Beta((dim - 1) / 2, (dim - 1) / 2).sample((kappa.size(0),)).to(kappa.device)
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = torch.rand_like(w)
            
            accept_mask = (kappa * w + (dim - 1) * torch.log(1 - x * w) - c) > torch.log(u)
            if accept_mask.all():
                return w.unsqueeze(1)

    def forward(self, x):
        enc1 = self.enc_b1(x)
        enc2 = self.enc_b2(enc1)
        enc3 = self.enc_b3(enc2)
        enc4 = self.enc_b4(enc3)
        
        pooled = self.global_pool(enc4)
        
        # Predict vMF parameters
        mu = F.normalize(self.fc_mu(pooled.F), p=2, dim=1)
        kappa = F.softplus(self.fc_kappa(pooled.F)).squeeze() + 1e-6 # Ensure kappa is positive

        # Sample from the hypersphere
        z = self._vmf_sample(mu, kappa) if self.training else mu
        
        z_feats = self.dec_fc(z)
        batch_indices = enc4.C[:, 0].long()
        z_feats_broadcasted = z_feats[batch_indices]
        s_z = ME.SparseTensor(
            features=z_feats_broadcasted,
            coordinate_map_key=enc4.coordinate_map_key,
            coordinate_manager=enc4.coordinate_manager
        )

        dec4 = self.dec_b4(ME.cat(s_z, enc4))
        dec3 = self.dec_b3(ME.cat(dec4, enc3))
        dec2 = self.dec_b2(ME.cat(dec3, enc2))
        dec1 = self.dec_b1(ME.cat(dec2, enc1))
        recon = self.final_conv(dec1)

        return {"recon": recon, "mu": mu, "kappa": kappa}

# --- (NeutrinoDataset and collate_fn are unchanged) ---
class NeutrinoDataset(torch.utils.data.Dataset):
    def __init__(self, root_file_path, tree_names, image_branch, label_branch='inclusive_strange_channels'):
        self.root_file_path = root_file_path
        self.image_branch = image_branch
        self.label_branch = label_branch
        self.file_handle = None
        self._scan_root_file(tree_names.split(','))

    def _get_file_handle(self):
        if not hasattr(self, 'file_handle') or self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self, tree_names):
        self.events = []
        self.labels = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file: continue
                tree = file[tree_name]
                labels_array = tree[self.label_branch].array(library="np")
                for i in range(tree.num_entries):
                    label = 1 if labels_array[i] in [10, 11] else 0
                    self.events.append((tree_name, i))
                    self.labels.append(label)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        flattened_image = tree[self.image_branch].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
        image_grid = np.array(flattened_image, dtype=np.float32).reshape(512, 512)
        rows, cols = np.nonzero(image_grid > 1e-6)
        if rows.size == 0: return None
        coords = torch.from_numpy(np.stack([rows, cols], axis=1)).int()
        feats = torch.from_numpy(np.log10(image_grid[rows, cols] + 1e-6)[:, np.newaxis]).float()
        feats = (feats - feats.mean()) / (feats.std() + 1e-8)
        return coords, feats, self.labels[idx]

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None, None, None
    coords_list, feats_list, labels_list = zip(*batch)
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_list, feats_list)
    return coords_batch, feats_batch, np.array(labels_list)

def make_data_loader(config, background_only=False, sampler=None, shuffle=True):
    dataset = NeutrinoDataset(
        config.root_file_path, config.tree_names, f'detector_image_{config.plane}')
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=config.num_workers, collate_fn=collate_fn)


def vae_loss(outputs, original_input, latent_dim, beta):
    # Reconstruction loss
    recon_loss = F.mse_loss(outputs["recon"].F, original_input.F, reduction='mean')
    
    # vMF KL Divergence
    mu, kappa = outputs["mu"], outputs["kappa"]
    
    # KL divergence between a vMF(mu, kappa) and a uniform vMF prior
    # Based on https://arxiv.org/abs/1703.01488
    kl_div = kappa * ( (latent_dim - 1) / (2 * kappa + torch.sqrt(4 * kappa**2 + (latent_dim - 1)**2)) ) \
           + (latent_dim / 2 - 1) * torch.log(kappa) \
           - (latent_dim / 2) * np.log(2 * np.pi) \
           - torch.log( (latent_dim - 1) / (2 * kappa + torch.sqrt(4 * kappa**2 + (latent_dim - 1)**2)) ) \
           - (latent_dim / 2 - 1) * torch.log(1 - ( (latent_dim - 1) / (2 * kappa + torch.sqrt(4 * kappa**2 + (latent_dim - 1)**2)) )**2)
    
    kl_loss = kl_div.mean()
    
    return recon_loss + beta * kl_loss


def calculate_validation_auc(net, device, config):
    net.eval()
    val_dataset = NeutrinoDataset(
        config.root_file_path, config.tree_names, f'detector_image_{config.plane}')
    signal_indices = np.where(val_dataset.labels == 1)[0]
    background_indices = np.where(val_dataset.labels == 0)[0]

    if len(signal_indices) < 500 or len(background_indices) < 500:
        print("Warning: Not enough samples for validation AUC.")
        return 0.5

    chosen_signal = np.random.choice(signal_indices, 500, replace=False)
    chosen_background = np.random.choice(background_indices, 500, replace=False)
    val_indices = np.concatenate([chosen_signal, chosen_background])
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_loader = make_data_loader(config, sampler=val_sampler, shuffle=False)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for coords, feats, labels_np in val_loader:
            if coords is None: continue
            input_tensor = ME.SparseTensor(feats, coords, device=device)
            outputs = net(input_tensor)
            
            batch_indices = input_tensor.C[:, 0].unique()
            for i, batch_idx in enumerate(batch_indices):
                item_mask = (input_tensor.C[:, 0] == batch_idx)
                recon_err = F.mse_loss(outputs['recon'].F[item_mask], input_tensor.F[item_mask]).item()
                all_scores.append(recon_err)
                all_labels.append(labels_np[i])

    return roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5


def train(net, device, config):
    optimizer = optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler()
    train_loader = make_data_loader(config, background_only=True)
    train_iter = iter(train_loader)

    running_loss, best_auc = 0.0, 0.5

    for i in range(config.max_steps):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        
        try:
            coords, feats, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            coords, feats, _ = next(train_iter)

        if coords is None: continue
        input_tensor = ME.SparseTensor(feats, coords, device=device)
        
        with autocast():
            outputs = net(input_tensor)
            loss = vae_loss(outputs, input_tensor, config.latent_dim, config.beta)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i == 0: running_loss = loss.item()
        else: running_loss = 0.99 * running_loss + 0.01 * loss.item()

        if i % config.stat_freq == 0:
            print(f"Iter: {i}, Batch Loss: {loss.item():.4f}, Avg Loss: {running_loss:.4f}")

        if i > 0 and i % config.val_freq == 0:
            val_auc = calculate_validation_auc(net, device, config)
            print(f"--- Validation at Iter {i}: AUC = {val_auc:.4f} ---")
            if val_auc > best_auc:
                best_auc = val_auc
                print(f"*** New best AUC: {best_auc:.4f}. Saving model weights. ***")
                torch.save(net.state_dict(), config.weights)

if __name__ == "__main__":
    config = parser.parse_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = HypersphericalVAE(latent_dim=config.latent_dim).to(device)
    
    print("--- Starting Training ---")
    train(net, device, config)
    
    print("\n--- Final Anomaly Detection on Test Set ---")
    net.load_state_dict(torch.load(config.weights))
    test_loader = make_data_loader(config, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for coords, feats, labels_np in test_loader:
             if coords is None: continue
             input_tensor = ME.SparseTensor(feats, coords, device=device)
             outputs = net(input_tensor)
             batch_indices = input_tensor.C[:, 0].unique()
             for i, batch_idx in enumerate(batch_indices):
                 item_mask = (input_tensor.C[:, 0] == batch_idx)
                 recon_err = F.mse_loss(outputs['recon'].F[item_mask], input_tensor.F[item_mask]).item()
                 scores.append(recon_err)
                 labels.append(labels_np[i])

    scores, labels = np.array(scores), np.array(labels)
    if len(np.unique(labels)) > 1:
        background_scores = scores[labels == 0]
        threshold = np.percentile(background_scores, 95) if len(background_scores) > 0 else np.median(scores)
        predictions = (scores > threshold).astype(int)
        
        print("\n--- Anomaly Detection Results ---")
        print(f"Final Test AUC: {roc_auc_score(labels, scores):.4f}")
        print(f"Threshold set to: {threshold:.4f}")
        print(classification_report(labels, predictions, target_names=["Background (0)", "Signal (1)"]))
    else:
        print("Not enough class diversity in test set for a full report.")