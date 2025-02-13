import os
import uproot
import numpy as np
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, config):
        self.file_path = config.get("dataset.file_path")
        self.tree_name = config.get("dataset.tree_name")
        self.width = config.get("dataset.width")
        self.height = config.get("dataset.height")
        self.plane_labels = config.get("dataset.plane_labels")
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.root_file = uproot.open(self.file_path)
        self.tree = self.root_file[self.tree_name]
        self.input_data = self.tree["input_data"].array(library="np")
        self.truth_data = self.tree["truth_data"].array(library="np")
        self.planes = self.tree["planes"].array(library="np")
        self.event_type = self.tree["event_type"].array(library="np")

    def __len__(self):
        return len(self.input_data)

class SegmentationDataset(BaseDataset):
    def __init__(self, config, plane_index=0):
        super().__init__(config)
        self.plane_index = plane_index

    def __getitem__(self, idx):
        event_input = self.input_data[idx]
        event_truth = self.truth_data[idx]
        plane = event_input[self.plane_index]
        label = event_truth[self.plane_index]
        image = np.array(list(plane), dtype=np.float32).reshape(self.height, self.width)
        target = np.array(list(label), dtype=np.float32).reshape(self.height, self.width)
        image = torch.tensor(image).unsqueeze(0)
        target = torch.tensor(target).unsqueeze(0)
        return image, target

class ContrastiveDataset(BaseDataset):
    def __init__(self, config, background_only=True):
        super().__init__(config)
        self.background_only = background_only
        if self.background_only:
            self.indices = self._filter_indices()
        else:
            self.indices = None

    def _filter_indices(self):
        indices = []
        for i in range(len(self)):
            if self.event_type[i] == "background":
                indices.append(i)
        return indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return super().__len__()

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        event_input = self.input_data[real_idx]
        planes = []
        for i in range(len(event_input)):
            plane = event_input[i]
            image = np.array(list(plane), dtype=np.float32).reshape(self.height, self.width)
            image = torch.tensor(image).unsqueeze(0)
            planes.append(image)
        return torch.stack(planes, dim=0)