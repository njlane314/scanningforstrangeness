import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def nan_mean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()

    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

class SegmentationData(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, transform=False, device=torch.device('cuda:0'), max_value=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.filenames = filenames
        self.device = device
        self.max_value = max_value 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.filenames[idx])
        with open(img_name, 'rb') as file:
            image = np.load(file)['arr_0']

        mask_name = os.path.join(self.mask_dir, self.filenames[idx])
        with open(mask_name, 'rb') as file:
            mask = np.load(file)['arr_0']

        image = torch.as_tensor(np.expand_dims(image, axis=0), dtype=torch.float).to(self.device)
        mask = torch.as_tensor(mask, dtype=torch.long).to(self.device)

        if self.max_value:
            image = image / self.max_value

        if self.transform:
            should_hflip = torch.rand(1) > 0.5
            should_vflip = torch.rand(1) > 0.5
            should_transpose = torch.rand(1) > 0.5

            if should_hflip:
                image = torch.flip(image, [2])  
                mask = torch.flip(mask, [1])
            if should_vflip:
                image = torch.flip(image, [1])  
                mask = torch.flip(mask, [0])
            if should_transpose:
                image = image.transpose(1, 2)
                mask = mask.transpose(0, 1)

        return image, mask

class SegmentationDataLoader():
    def __init__(self, root_dir, view, batch_size, train_pct=None, valid_pct=0.1,
                 test_pct=0.0, transform=False, device=torch.device('cuda:0')):
        assert (valid_pct + test_pct) < 1.0

        image_dir = os.path.join(root_dir, f"images_{view}/input")
        mask_dir = os.path.join(root_dir, f"images_{view}/target")

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Directory not found: {mask_dir}")

        image_filenames = np.array(next(os.walk(image_dir))[2])
        if len(image_filenames) == 0:
            raise FileNotFoundError(f"No image files found in directory: {image_dir}")
        else:
            print(f"Found {len(image_filenames)} image files in {image_dir}")

        n_files = len(image_filenames)
        valid_size = int(n_files * valid_pct)
        train_size = n_files - valid_size if train_pct is None else int(n_files * train_pct)

        sample = np.random.permutation(n_files)
        train_sample = sample[valid_size:] if not train_size else sample[valid_size:valid_size + train_size]
        valid_sample = sample[:valid_size]

        self.max_value = None

        self.train_ds = SegmentationData(image_dir, mask_dir, image_filenames[train_sample], transform, device)
        self.valid_ds = SegmentationData(image_dir, mask_dir, image_filenames[valid_sample], None, device)

        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    def find_max_value(self):
        max_value = -float('inf')
        for batch in tqdm(self.train_dl, desc="Finding max value"):
            images, _ = batch
            max_value = max(max_value, images.max().item())
        return max_value

    def apply_normalization(self, max_value):
        self.max_value = max_value

        self.train_ds = SegmentationData(self.train_ds.image_dir, self.train_ds.mask_dir,
                                         self.train_ds.filenames, self.train_ds.transform,
                                         self.train_ds.device, max_value=self.max_value)

        self.valid_ds = SegmentationData(self.valid_ds.image_dir, self.valid_ds.mask_dir,
                                         self.valid_ds.filenames, self.valid_ds.transform,
                                         self.valid_ds.device, max_value=self.max_value)

        self.train_dl = DataLoader(self.train_ds, batch_size=self.train_dl.batch_size,
                                   shuffle=True, drop_last=True, num_workers=0)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=self.valid_dl.batch_size,
                                   shuffle=False, drop_last=True, num_workers=0)

