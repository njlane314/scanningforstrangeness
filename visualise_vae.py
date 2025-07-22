import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import uproot

import torch
import torch.nn as nn
import MinkowskiEngine as ME

# ==============================================================================
# ## --- MODEL AND DATA STRUCTURES (UNCHANGED) --- ##
# ==============================================================================

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.transform = transform
        self.cache = {}
        self.root_file_path = config.root_file_path
        self.tree_names = config.tree_names
        self.file_handle = None
        self._scan_root_file()

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self):
        logging.info(f"Scanning ROOT file for TTrees: {self.tree_names}...")
        all_events = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in self.tree_names:
                if tree_name not in file:
                    logging.warning(f"TTree '{tree_name}' not found. Skipping.")
                    continue
                tree = file[tree_name]
                num_entries = tree.num_entries
                logging.info(f"Loading {num_entries} events from TTree '{tree_name}'.")
                for i in range(num_entries):
                    all_events.append((tree_name, i))
        if not all_events:
            raise RuntimeError("No events loaded. Check TTree names and file content.")
        self.events = all_events
        logging.info(f"Scan complete. Found {len(self.events)} total events.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        
        flattened_image = tree['detector_image_w'].array(entry_start=entry_index, entry_stop=entry_index + 1, library="np")[0]
        label_array = tree['inclusive_strange_channels'].array(entry_start=entry_index, entry_stop=entry_index + 1, library="np")
        label = label_array[0] if len(label_array) > 0 else -1 # Handle empty label

        image_shape = (512, 512)
        image_grid = flattened_image.astype(np.float32).reshape(image_shape)
        rows, cols = np.where(image_grid > 0)

        if rows.size == 0:
            logging.warning(f"Skipping event {idx}: no features found in the image.")
            return None

        coords = np.stack([rows, cols], axis=1).astype(np.int32)
        feats = image_grid[rows, cols][:, np.newaxis]
        
        self.cache[idx] = (coords, feats, label)
        return (coords, feats, label)

class Encoder(nn.Module):
    CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    def __init__(self):
        super().__init__()
        ch = self.CHANNELS
        self.block1 = self.make_block(1, ch[0])
        self.block2 = self.make_block(ch[0], ch[1])
        self.block3 = self.make_block(ch[1], ch[2])
        self.block4 = self.make_block(ch[2], ch[3])
        self.block5 = self.make_block(ch[3], ch[4])
        self.block6 = self.make_block(ch[4], ch[5])
        self.block7 = self.make_block(ch[5], ch[6])
        self.global_pool = ME.MinkowskiGlobalPooling()
        self.linear_mean = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.linear_log_var = ME.MinkowskiLinear(ch[6], ch[6], bias=True)

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=2, dimension=2),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=2),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
        )
    def forward(self, sinput):
        out = self.block1(sinput)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.global_pool(out)
        mean = self.linear_mean(out)
        log_var = self.linear_log_var(out)
        return mean, log_var

class Decoder(nn.Module):
    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]
    def __init__(self):
        super().__init__()
        ch = self.CHANNELS
        self.resolution = 128
        self.block1, self.block1_cls = self.make_block(ch[0], ch[1], first_block=True)
        self.block2, self.block2_cls = self.make_block(ch[1], ch[2])
        self.block3, self.block3_cls = self.make_block(ch[2], ch[3])
        self.block4, self.block4_cls = self.make_block(ch[3], ch[4])
        self.block5, self.block5_cls = self.make_block(ch[4], ch[5])
        self.block6, self.block6_cls = self.make_block(ch[5], ch[6])
        self.pruning = ME.MinkowskiPruning()

    def make_block(self, in_channels, out_channels, first_block=False):
        layers = []
        if first_block:
             layers.extend([ ME.MinkowskiGenerativeConvolutionTranspose(in_channels, in_channels, kernel_size=2, stride=2, dimension=2),
                 ME.MinkowskiBatchNorm(in_channels), ME.MinkowskiELU(),
                 ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=3, dimension=2),
                 ME.MinkowskiBatchNorm(in_channels), ME.MinkowskiELU() ])
        layers.extend([ ME.MinkowskiGenerativeConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=2),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=2),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU() ])
        block = nn.Sequential(*layers)
        cls_layer = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=2)
        return block, cls_layer

    @torch.no_grad()
    def get_target(self, out, target_key):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(out.coordinate_map_key, strided_target_key, kernel_size=1, region_type=1)
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def forward(self, z_glob, target_key):
        out_cls, targets = [], []
        z = ME.SparseTensor(features=z_glob.F, coordinates=z_glob.C, tensor_stride=self.resolution, coordinate_manager=z_glob.coordinate_manager)
        blocks = [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]
        cls_layers = [self.block1_cls, self.block2_cls, self.block3_cls, self.block4_cls, self.block5_cls, self.block6_cls]
        out = z
        for i, (block, cls_layer) in enumerate(zip(blocks, cls_layers)):
            out = block(out)
            out_cl = cls_layer(out)
            target = self.get_target(out, target_key)
            targets.append(target)
            out_cls.append(out_cl)
            if self.training:
                keep = (out_cl.F > 0).squeeze() + target
            else:
                keep = (out_cl.F > 0).squeeze()
            if i < len(blocks) - 1 and keep.sum() > 0:
                 out = self.pruning(out, keep)
        return out_cls, targets, out

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, sinput, gt_target):
        means, log_vars = self.encoder(sinput)
        zs = means
        if self.training:
            zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target)
        return out_cls, targets, sout, means, log_vars, zs

# ==============================================================================
# ## --- VISUALIZATION SCRIPT MAIN LOGIC (MODIFIED) --- ##
# ==============================================================================

def sparse_tensor_to_dense_image(stensor, image_shape=(512, 512)):
    """Converts a MinkowskiEngine SparseTensor to a dense 2D numpy image."""
    coords = stensor.C.cpu().numpy()
    feats = stensor.F.cpu().numpy()
    dense_image = np.zeros(image_shape, dtype=np.float32)
    if len(coords) == 0:
        return dense_image
    rows, cols = coords[:, 1], coords[:, 2]
    dense_image[rows, cols] = feats.squeeze()
    return dense_image

def visualize_reconstruction(config):
    """Loads a model, then loops through a list of event indices to visualize and save each one."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # --- Part 1: Setup (runs only once) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading dataset...")
    dataset = ImageDataset(phase="visualize", config=config)
    
    logging.info(f"Loading trained model weights from: {config.weights}")
    model = VAE()
    checkpoint = torch.load(config.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # --- Part 2: Loop over requested indices ---
    for index in config.indices:
        print("-" * 50)
        logging.info(f"Processing event index: {index}")

        if index >= len(dataset):
            logging.warning(f"Skipping: Index {index} is out of bounds for the dataset of size {len(dataset)}.")
            continue

        data = dataset[index]
        if data is None:
            logging.warning(f"Skipping: Could not retrieve data for index {index} (it might be an empty image).")
            continue
        
        coords, feats, label = data

        # Prepare the input tensor for the model
        coords_tensor = torch.from_numpy(coords).int()
        feats_tensor = torch.from_numpy(feats).float()
        batched_coords = ME.utils.batched_coordinates([coords_tensor]).to(device)
        feats_tensor = feats_tensor.to(device)
        sinput = ME.SparseTensor(features=feats_tensor, coordinates=batched_coords)
        
        # Perform the forward pass (inference)
        with torch.no_grad():
            out_cls, _, _, _, _, _ = model(sinput, sinput.coordinate_map_key)

        sout_logits = out_cls[-1]
        sout_probs = ME.SparseTensor(
            features=torch.sigmoid(sout_logits.F),
            coordinate_map_key=sout_logits.coordinate_map_key,
            coordinate_manager=sout_logits.coordinate_manager
        )
        
        # Convert sparse tensors to dense images for plotting
        input_image = sparse_tensor_to_dense_image(sinput, image_shape=(512, 512))
        output_image = sparse_tensor_to_dense_image(sout_probs, image_shape=(512, 512))
        
        # Generate and save the visualization
        output_filename = config.output_pattern.format(index=index)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot Input
        im1 = axes[0].imshow(input_image, cmap='viridis', origin='lower', vmin=0)
        axes[0].set_title(f'Input Image (Index: {index})')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot Reconstructed Output
        im2 = axes[1].imshow(output_image, cmap='viridis', origin='lower', vmin=0, vmax=1)
        axes[1].set_title('Reconstructed Output (Probabilities)')
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_filename)
        logging.info(f"Image successfully saved to {output_filename}")
        plt.close(fig)

    print("-" * 50)
    logging.info("All requested events have been processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize input and output of a trained VAE model for multiple events."
    )
    parser.add_argument("--root_file_path", type=str, required=True, 
                        help="Path to the input ROOT file.")
    parser.add_argument("--tree_names", nargs='+', type=str, required=True, 
                        help="List of TTree names to process.")
    parser.add_argument("--weights", type=str, default="vae_weights.pth", 
                        help="Path to the saved model weights (.pth file).")
    # MODIFIED ARGUMENTS
    parser.add_argument("--indices", type=int, nargs='+', required=True, 
                        help="A space-separated list of event indices to visualize (e.g., --indices 10 42 99).")
    parser.add_argument("--output_pattern", type=str, default="reconstruction_event_{index}.png",
                        help="Pattern for saving output images. Use '{index}' as a placeholder for the event index.")
    
    config = parser.parse_args()
    visualize_reconstruction(config)