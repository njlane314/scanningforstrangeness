import os
import uproot
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, config):
        self.path = config.get("dataset.path")
        self.file = config.get("dataset.file")
        self.tree_name = config.get("dataset.tree")
        self.width = config.get("dataset.width")
        self.height = config.get("dataset.height")
        self.seg_classes = config.get("train.segmentation_classes")
        self.file_path = os.path.join(self.path, self.file)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.root_file = uproot.open(self.file_path)
        self.tree = self.root_file[self.tree_name]
        self.run_array = self.tree["run"].array(library="np")
        self.subrun_array = self.tree["subrun"].array(library="np")
        self.event_array = self.tree["event"].array(library="np")
        self.type_array = self.tree["type"].array(library="np")
        self.num_events = len(self.run_array)
    def __len__(self):
        return self.num_events
    def __getitem__(self, idx):
        run = self.run_array[idx]
        subrun = self.subrun_array[idx]
        event = self.event_array[idx]
        input_data = self.tree["input"].array(library="np", entry_start=idx, entry_stop=idx+1)[0]
        truth_data = self.tree["truth"].array(library="np", entry_start=idx, entry_stop=idx+1)[0]
        images = []
        targets = []
        for plane_idx in range(len(input_data)):
            plane_vector = input_data[plane_idx]
            label_vector = truth_data[plane_idx]
            plane = np.fromiter(plane_vector, dtype=np.float32, count=self.width * self.height).reshape(self.height, self.width)
            label = np.fromiter(label_vector, dtype=np.float32, count=self.width * self.height).reshape(self.height, self.width)
            images.append(plane)
            targets.append(label.astype(np.int64))
        images_np = np.stack(images, axis=0)
        targets_np = np.stack(targets, axis=0)
        one_hot_targets = []
        for plane_label in targets_np:
            plane_tensor = torch.tensor(plane_label, dtype=torch.long)
            one_hot = F.one_hot(plane_tensor, num_classes=self.seg_classes).permute(2, 0, 1)
            one_hot_targets.append(one_hot)
        one_hot_targets = torch.cat(one_hot_targets, dim=0)
        return torch.tensor(images_np), one_hot_targets, run, subrun, event