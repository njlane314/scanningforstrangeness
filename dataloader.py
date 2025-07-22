import os
import sys
import logging
import numpy as np
import uproot
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import MinkowskiEngine as ME

class NeutrinoDataset(Dataset):
    def __init__(self, root_file_path, tree_names, image_branch, label_branch, weight_branch, quantization_size=1.0, image_shape=(512, 512)):
        self.root_file_path = root_file_path
        self.image_branch = image_branch
        self.label_branch = label_branch
        self.quantization_size = quantization_size
        self.image_shape = image_shape
        self.file_handle = None
        self._scan_root_file(tree_names, weight_branch)

    def _get_file_handle(self):
        if self.file_handle is None:
            self.file_handle = uproot.open(self.root_file_path)
        return self.file_handle

    def _scan_root_file(self, tree_names, weight_branch):
        logging.info(f"Scanning ROOT file: {self.root_file_path}...")
        self.events = []
        all_weights = []
        with uproot.open(self.root_file_path) as file:
            for tree_name in tree_names:
                if tree_name not in file:
                    logging.warning(f"TTree '{tree_name}' not found. Skipping.")
                    continue
                tree = file[tree_name]
                num_entries = tree.num_entries
                logging.info(f"Scanning TTree '{tree_name}' with {num_entries} entries.")
                weights = tree[weight_branch].array(library="np")
                for i in range(num_entries):
                    self.events.append((tree_name, i))
                all_weights.append(weights)
        if not self.events:
            raise RuntimeError("No events loaded. Check TTree names and file content.")
        self.weights = np.concatenate(all_weights)
        logging.info(f"Scan complete. Found {len(self.events)} total events.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        tree_name, entry_index = self.events[idx]
        file = self._get_file_handle()
        tree = file[tree_name]
        
        flattened_image = tree[self.image_branch].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
        label = tree[self.label_branch].array(entry_start=entry_index, entry_stop=entry_index + 1)[0]
            
        image_grid = np.array(flattened_image, dtype=np.float32).reshape(self.image_shape)
        
        rows, cols = np.where(image_grid > 1.0)
        
        if rows.size == 0:
            return (torch.empty((0, 2), dtype=torch.int32),
                    torch.empty((0, 1), dtype=torch.float32),
                    torch.tensor(label, dtype=torch.long))

        coords = np.stack([rows, cols], axis=1)
        feats = np.log(image_grid[rows, cols])[:, np.newaxis]

        return (torch.from_numpy(coords).int(),
                torch.from_numpy(feats).float(),
                torch.tensor(label, dtype=torch.long))

def make_data_loader(root_file_path, tree_names, image_branch, label_branch, weight_branch, quantization_size, batch_size, num_workers):
    dataset = NeutrinoDataset(root_file_path, tree_names, image_branch, label_branch, weight_branch, quantization_size)
    sampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=ME.utils.SparseCollation(),
        pin_memory=True,
    )

