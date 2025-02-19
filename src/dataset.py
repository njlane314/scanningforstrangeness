import os
import uproot
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, config):
        self.path = config.get("dataset.path")
        self.tree = config.get("dataset.tree")
        self.width = config.get("dataset.dims.width")
        self.height = config.get("dataset.dims.height")
        self.plane = config.get("dataset.planes")
        self._load_data()
    def _load_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
        self.root_file = uproot.open(self.path)
        self.tree = self.root_file[self.tree]
        self.input_data = self.tree["input_data"].array(library="np")
        self.truth_data = self.tree["truth_data"].array(library="np")
        self.planes = self.tree["planes"].array(library="np")
        self.event_type = self.tree["event_type"].array(library="np")
    def __len__(self):
        return len(self.input_data)

class SegmentationDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.seg_classes = config.get("model.seg_classes")
    def __getitem__(self, idx):
        event_input = self.input_data[idx]
        event_truth = self.truth_data[idx]
        images = []
        targets = []
        for plane_idx in range(len(event_input)):
            plane_vector = event_input[plane_idx]
            label_vector = event_truth[plane_idx]
            plane = np.fromiter(plane_vector, dtype=np.float32, count=self.width * self.height).reshape(self.height, self.width)
            label = np.fromiter(label_vector, dtype=np.float32, count=self.width * self.height).reshape(self.height, self.width)
            images.append(plane)
            targets.append(label.astype(np.int64))
        images_np = np.stack(images, axis=0)
        targets_np = np.stack(targets, axis=0)
        one_hot_targets = []
        for plane_label in targets_np:
            plane_tensor = torch.tensor(plane_label, dtype=torch.long)
            one_hot = F.one_hot(plane_tensor, num_classes=self.seg_classes)
            one_hot = one_hot.permute(2, 0, 1)
            one_hot_targets.append(one_hot)
        one_hot_targets = torch.cat(one_hot_targets, dim=0)
        return torch.tensor(images_np), one_hot_targets

class ContrastiveDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, idx):
        event_input = self.input_data[idx]
        images = []
        for plane in event_input:
            image = np.fromiter(plane, dtype=np.float32, count=self.width * self.height).reshape(self.height, self.width)
            images.append(image)
        return torch.tensor(images)