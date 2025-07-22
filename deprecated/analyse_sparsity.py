import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import uproot
from torch.utils.data import Dataset
from tqdm import tqdm

# --- Start of Classes from Training Script ---

@dataclass
class Config:
    """Configuration class to hold dataset parameters."""
    ROOT_FILE_PATH: Path = Path("/gluster/data/dune/niclane/scanning/numucc_snapshot.root")
    TTREE_NAME: str = "mc_inclusive_run1_fhc"
    IMAGE_BRANCH: str = "raw_image_w"
    LABEL_BRANCH: str = "analysis_channel"
    EVENT_CATEGORIES: Dict[int, int] = field(default_factory=lambda: {
        20: 0, 21: 1, 22: 2, 23: 3, 30: 4, 31: 5
    })
    IMAGE_HEIGHT: int = 512
    IMAGE_WIDTH: int = 512
    ADC_THRESHOLD: float = 4.0

class ImageClassifierDataset(Dataset):
    """Dataset loader for reading and preprocessing event images from a ROOT file."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.file = uproot.open(self.cfg.ROOT_FILE_PATH)
        self.tree = self.file[cfg.TTREE_NAME]
        self.all_labels_raw = self.tree[self.cfg.LABEL_BRANCH].array(library="np")
        self.valid_indices = np.where(np.isin(self.all_labels_raw, list(self.cfg.EVENT_CATEGORIES.keys())))[0]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> np.ndarray:
        # Modified to only return the processed numpy image for analysis
        actual_idx = self.valid_indices[idx]
        raw_image = self.tree[self.cfg.IMAGE_BRANCH].array(entry_start=actual_idx, entry_stop=actual_idx+1)[0]
        image_np = np.fromiter(raw_image, dtype=np.float32).reshape(self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        image_np[image_np < self.cfg.ADC_THRESHOLD] = 0.0
        return image_np

# --- End of Classes from Training Script ---

def analyze_data_sparsity(sample_size=1000):
    """
    Calculates the sparsity (percentage of non-zero pixels) for a sample of images.
    """
    cfg = Config()
    dataset = ImageClassifierDataset(cfg)
    
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    occupancies = []
    total_pixels = cfg.IMAGE_HEIGHT * cfg.IMAGE_WIDTH
    
    # Take a random sample to analyze
    indices_to_check = np.random.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)

    print(f"Analyzing sparsity for {len(indices_to_check)} random images...")
    for idx in tqdm(indices_to_check):
        processed_image_np = dataset[idx]
        non_zero_pixels = np.count_nonzero(processed_image_np)
        occupancy = (non_zero_pixels / total_pixels) * 100
        occupancies.append(occupancy)

    occupancies = np.array(occupancies)
    
    print("\n--- DATA SPARSITY ANALYSIS ---")
    print(f"ADC Threshold used: {cfg.ADC_THRESHOLD}")
    print(f"Mean Occupancy: {np.mean(occupancies):.4f}%")
    print(f"Median Occupancy: {np.median(occupancies):.4f}%")
    print(f"Max Occupancy: {np.max(occupancies):.4f}%")
    print(f"Min Occupancy: {np.min(occupancies):.4f}%")
    
    # Count how many images are completely empty
    empty_images = np.sum(occupancies == 0)
    print(f"Number of completely empty images in sample: {empty_images} out of {len(indices_to_check)} ({empty_images/len(indices_to_check)*100:.2f}%)")
    print("--------------------------------")

if __name__ == '__main__':
    analyze_data_sparsity()