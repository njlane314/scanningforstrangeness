import os
import sys
import argparse
import logging
import numpy as np
import torch

try:
    import MinkowskiEngine as ME
except ImportError:
    raise ImportError("MinkowskiEngine not found.")

# Import your data loading and model classes from your main script
from train_vae import ImageDataset, make_data_loader, VAE

def debug_model_coordinates(config):
    """
    Loads the model and one data point, then prints the coordinate
    ranges at every stage of the decoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained model
    if not os.path.exists(config.weights):
        logging.error(f"Weights file not found at {config.weights}")
        return
        
    net = VAE(D=2)
    net.to(device)
    checkpoint = torch.load(config.weights)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    logging.info(f"Successfully loaded weights from {config.weights}")

    # 2. Load one batch of data
    dataloader = make_data_loader("test", config)
    data_dict = next(iter(dataloader))
    if data_dict is None:
        logging.error("Failed to load data.")
        return

    # 3. Run a single forward pass
    with torch.no_grad():
        sin = ME.SparseTensor(
            features=data_dict["feats"].to(device),
            coordinates=data_dict["coords"].to(device)
        )
        
        logging.info("\n--- INPUT TENSOR ---")
        input_coords = sin.C[:, 1:].cpu().numpy()
        logging.info(f"Input MIN Coords: {np.min(input_coords, axis=0)}")
        logging.info(f"Input MAX Coords: {np.max(input_coords, axis=0)}")
        
        out_cls, _, _, _ = net(sin)
        
        logging.info("\n--- DECODER OUTPUTS ---")
        for i, sparse_out in enumerate(out_cls):
            if sparse_out.F.numel() > 0:
                coords = sparse_out.C[:, 1:].cpu().numpy()
                min_c = np.min(coords, axis=0)
                max_c = np.max(coords, axis=0)
                logging.info(f"Output Layer {i}: Stride={sparse_out.tensor_stride[0]}, Points={len(coords)}, MIN={min_c}, MAX={max_c}")
            else:
                logging.info(f"Output Layer {i}: Is Empty.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # These must match the settings used for training
    parser.add_argument("--root_file_path", type=str, required=True, help="Path to your ROOT file.")
    parser.add_argument("--tree_names", nargs='+', type=str, required=True, help="Name of the TTree(s).")
    parser.add_argument("--weights", type=str, default="neutrino_vae_final.pth", help="Path to your trained model weights.")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_mode", action='store_false') # Set to false for test mode
    
    config = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # This assumes your VAE classes are in a file named 'train_vae.py'
    # If your file is named something else, change it here.
    sys.path.append(os.getcwd())
    
    debug_model_coordinates(config)