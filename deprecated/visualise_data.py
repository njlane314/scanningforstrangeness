import logging
import sys
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import uproot
from torch.utils.data import Dataset

# To make this script standalone, the necessary classes from your training
# script are included here.

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
        try:
            self.file = uproot.open(self.cfg.ROOT_FILE_PATH)
        except FileNotFoundError:
            logging.error(f"ROOT file not found at: {self.cfg.ROOT_FILE_PATH}")
            sys.exit(1)

        if cfg.TTREE_NAME not in self.file:
            raise KeyError(f"TTree '{cfg.TTREE_NAME}' not found in '{cfg.ROOT_FILE_PATH}'. Available keys: {self.file.keys()}")
        self.tree = self.file[cfg.TTREE_NAME]

        self.all_labels_raw = self.tree[self.cfg.LABEL_BRANCH].array(library="np")
        self.valid_indices = np.where(np.isin(self.all_labels_raw, list(self.cfg.EVENT_CATEGORIES.keys())))[0]

        if len(self.valid_indices) == 0:
            raise ValueError(f"Found 0 events matching the specified categories in TTree '{cfg.TTREE_NAME}'.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]
        entry = self.tree.arrays(
            [self.cfg.IMAGE_BRANCH, self.cfg.LABEL_BRANCH],
            entry_start=actual_idx,
            entry_stop=actual_idx + 1,
            library="np"
        )
        image_np = entry[self.cfg.IMAGE_BRANCH][0].astype(np.float32).reshape(1, self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        image_np[image_np < self.cfg.ADC_THRESHOLD] = 0.0
        image_np = np.log1p(image_np)
        image_tensor = torch.from_numpy(image_np)
        label_val = entry[self.cfg.LABEL_BRANCH][0]
        label = self.cfg.EVENT_CATEGORIES[label_val]
        return image_tensor, torch.tensor(label, dtype=torch.long)

# --- End of Classes from Training Script ---


def visualize_events(num_images: int = 12, output_filename: str = "event_visualization_grid.png"):
    """
    Loads a random sample of events from the dataset, visualizes them
    in a grid, and saves the grid to a PNG file.
    """
    logging.info("Initializing configuration and dataset for visualization...")
    cfg = Config()
    dataset = ImageClassifierDataset(cfg)
    logging.info(f"Successfully loaded dataset with {len(dataset)} valid events.")

    reverse_event_categories = {v: k for k, v in cfg.EVENT_CATEGORIES.items()}

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    logging.info(f"Plotting {min(num_images, len(dataset))} random events...")

    for i in range(len(axes)):
        if i < min(num_images, len(dataset)):
            random_index = random.randint(0, len(dataset) - 1)
            image_tensor, label_tensor = dataset[random_index]

            image_numpy = image_tensor.squeeze().numpy()
            label_index = label_tensor.item()
            raw_event_code = reverse_event_categories.get(label_index, "Unknown")

            ax = axes[i]
            im = ax.imshow(image_numpy, cmap='viridis', origin='lower', aspect='auto')
            ax.set_title(f"Label: {label_index} (Event Code: {raw_event_code})")
            fig.colorbar(im, ax=ax)
        
        axes[i].axis('off')

    plt.suptitle("Random Sample of Preprocessed Events", fontsize=24, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- CHANGED SECTION: Save the figure instead of showing it ---
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logging.info(f"Plot successfully saved to: {output_filename}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
    finally:
        plt.close(fig)  # Close the figure to free up memory


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        visualize_events(num_images=12)
    except (ValueError, KeyError) as e:
        logging.error(f"An error occurred: {e}")