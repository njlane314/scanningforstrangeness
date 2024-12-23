import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F

class ImageDataLoader():
    def __init__(self, input_dir, target_dir, batch_size, train_pct=None, valid_pct=0.1, test_pct=0.0, transform=False, device=torch.device('cuda:0')):
        assert (valid_pct + test_pct) < 1.0

        self.input_dir = input_dir
        self.target_dir = target_dir

        input_files = np.array(next(os.walk(self.input_dir))[2])

        n_files = len(input_files)
        valid_size = int(n_files * valid_pct)
        train_size = n_files - valid_size if train_pct is None else int(n_files * train_pct)

        sample = np.random.permutation(n_files)
        train_sample = sample[valid_size:] if not train_size else sample[valid_size:valid_size + train_size]
        valid_sample = sample[:valid_size]

        self.train_ds = ImageDataset(self.input_dir, self.target_dir, input_files[train_sample], transform, device)
        self.valid_ds = ImageDataset(self.input_dir, self.target_dir, input_files[valid_sample], None, device)

        self.valid_signature_ds, self.valid_background_ds = self.split_validation_set()

        event_labels = self.get_event_labels(self.train_ds)
        signature_weight = 0.9
        background_weight = 0.1
        weights = [signature_weight if label == 1 else background_weight for label in event_labels]

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0)
        self.valid_signature_dl = DataLoader(self.valid_signature_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
        self.valid_background_dl = DataLoader(self.valid_background_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    def get_event_labels(self, dataset):
        event_labels = []
        for _, target in tqdm(dataset, desc="Processing Event Labels"):
            event_contains_signature = (target == 2).any().item()
            event_labels.append(1 if event_contains_signature else 0)
        return event_labels

    def split_validation_set(self):
        signature_files = []
        background_files = []
        
        for idx, (_, target) in enumerate(tqdm(self.valid_ds, desc="Splitting Validation Set")):
            if (target == 2).any().item():
                signature_files.append(self.valid_ds.filenames[idx])
            else:
                background_files.append(self.valid_ds.filenames[idx])

        valid_signature_ds = ImageDataset(self.input_dir, self.target_dir, signature_files, transform=False, device=self.valid_ds.device)
        valid_background_ds = ImageDataset(self.input_dir, self.target_dir, background_files, transform=False, device=self.valid_ds.device)

        return valid_signature_ds, valid_background_ds

    def count_classes(self, num_classes):
        count = np.zeros(num_classes)
        for batch in tqdm(self.train_dl, desc="Counting Classes"):
            _, truth = batch
            unique = torch.unique(truth)
            counts = torch.stack([(truth == x_u).sum() for x_u in unique])
            unique = [u.item() for u in unique]
            counts = [c.item() for c in counts]
            this_dict = dict(zip(unique, counts))
            for key in this_dict:
                count[key] += this_dict[key]
        return count


class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, filenames, transform=False, device=torch.device('cuda:0')):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.filenames = filenames
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_name = os.path.join(self.input_dir, self.filenames[idx])
        with open(input_name, 'rb') as file:
            input = np.load(file)['arr_0']

        target_name = os.path.join(self.target_dir, self.filenames[idx])
        with open(target_name, 'rb') as file:
            target = np.load(file)['arr_0']

        input = torch.as_tensor(np.expand_dims(input, axis=0), dtype=torch.float).to(self.device)
        target = torch.as_tensor(target, dtype=torch.long)

        if self.transform:
            should_hflip = torch.rand(1) > 0.5
            should_vflip = torch.rand(1) > 0.5
            should_transpose = torch.rand(1) > 0.5

            if should_hflip:
                input = torch.flip(input, [2])  
                target = torch.flip(target, [1])
            if should_vflip:
                input = torch.flip(input, [1])  
                target = torch.flip(target, [0])
            if should_transpose:
                input = input.transpose(1, 2)
                target = target.transpose(0, 1)

        return input, target