# visualize_concepts_final.py

import os
import sys
import argparse
import logging
import numpy as np
import uproot
import torch
import matplotlib.pyplot as plt

try:
    import MinkowskiEngine as ME
except ImportError:
    raise ImportError("MinkowskiEngine not found. Please install it.")

# Import model and data classes from your training script
from train_vae import VAE, ImageDataset, make_data_loader

def get_average_latent_vector(net, dataloader, device, num_samples=1000):
    """
    Calculates the average latent vector from a sample of the dataset.
    """
    net.eval()
    all_means = []
    num_processed = 0
    batch_size = dataloader.batch_size
    
    with torch.no_grad():
        for i, data_dict in enumerate(dataloader):
            if data_dict is None: continue
            
            logging.info(f"Processing batch {i+1} to average latent vectors...")
            
            sin = ME.SparseTensor(
                features=data_dict["feats"].to(device),
                coordinates=data_dict["coords"].to(device),
            )
            
            # ================================================================
            # ## --- THE FIX IS HERE (PART 1) --- ##
            # ================================================================
            # The VAE's forward pass returns 4 items in eval mode.
            # We only need the 'means' vector, which is the 3rd item.
            _, _, means, _ = net(sin, None)
            # ================================================================
            
            all_means.append(means.F.cpu())
            
            num_processed += sin.C[:, 0].max().item() + 1
            if num_processed >= num_samples:
                logging.info(f"Reached {num_processed} samples. Stopping averaging.")
                break
                
    if not all_means:
        raise ValueError("Could not extract any latent vectors. Is the dataset empty?")

    return torch.cat(all_means, dim=0).mean(dim=0, keepdim=True)


def generate_from_latent(net, latent_vector, device):
    """
    Generates a shape from a given latent vector using a robust method.
    """
    net.eval()
    
    # ================================================================
    # ## --- THE FIX IS HERE (PART 2) --- ##
    # ================================================================
    # We create a new, clean coordinate manager for this single generation task.
    # This avoids all the previous device and API mismatch errors.
    cm = ME.CoordinateManager(D=2)
    
    # The coordinate for the global vector is just (0,0) in a single batch
    latent_coords = torch.zeros((1, 3), dtype=torch.int32, device=device)
    latent_features = latent_vector.to(device)
    
    z_glob = ME.SparseTensor(
        features=latent_features,
        coordinates=latent_coords,
        coordinate_manager=cm
    )

    with torch.no_grad():
        # The target_key is a dummy placeholder for inference.
        out_cls, _ = net.decoder(z_glob, None)
    # ================================================================
        
    s_probs = torch.sigmoid(out_cls.F)
    pruning_mask = s_probs.squeeze() > 0.5
    
    if pruning_mask.sum() > 0:
        return ME.MinkowskiPruning()(out_cls, pruning_mask)
    else:
        # If no pixels are above the threshold, return the empty tensor
        return out_cls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the 'average' learned concept from a trained VAE."
    )
    parser.add_argument("--root_file_path", type=str, required=True)
    parser.add_argument("--tree_names", nargs='+', type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    config = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading weights from {config.weights}")
    net = VAE()
    checkpoint = torch.load(config.weights, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)

    # Use a Namespace to create a config object for the data loader
    dataloader_config = argparse.Namespace(**vars(config), train=False)
    dataloader = make_data_loader("test", dataloader_config)

    logging.info("Calculating the average latent vector for background events...")
    avg_latent_vector = get_average_latent_vector(net, dataloader, device)

    logging.info("Generating shape from the average vector...")
    generated_shape = generate_from_latent(net, avg_latent_vector, device)
    
    output_coords_decomposed, _ = generated_shape.decomposed_coordinates_and_features
    out_img = np.zeros((512, 512))

    if output_coords_decomposed:
        out_c = output_coords_decomposed[0].cpu().numpy()
        if out_c.size > 0:
            # Use the final tensor stride from generated_shape
            tensor_stride = generated_shape.tensor_stride[0]  # Should be 2
            # Scale both x and y coordinates
            scaled_coords = (out_c * tensor_stride).astype(int)
            
            # Ensure coordinates are within the 512x512 image bounds
            valid_idx = (scaled_coords[:, 0] >= 0) & (scaled_coords[:, 0] < 512) & \
                        (scaled_coords[:, 1] >= 0) & (scaled_coords[:, 1] < 512)
            valid_coords = scaled_coords[valid_idx]
            
            # Assign to image: out_img[y, x], where valid_coords[:, 0] is x, [:, 1] is y
            out_img[valid_coords[:, 1], valid_coords[:, 0]] = 1
    

    plt.figure(figsize=(8, 8))
    plt.imshow(out_img, cmap='gray_r', origin='lower')
    plt.title("Generated 'Platonic Ideal' of a Background Event")
    
    output_filename = "average_background_visualization.png"
    plt.savefig(output_filename)
    logging.info(f"Saved interpretable visualization to {output_filename}")
    plt.close()