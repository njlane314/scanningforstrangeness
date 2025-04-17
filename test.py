# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet as scn

# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Not using CUDA.")

# Define the UResNet class
class UResNet(torch.nn.Module):
    def __init__(self, img_size: int, num_classes: int):
        super(UResNet, self).__init__()
        self.dimensions = 2
        self.spatial_size = (img_size, img_size)
        self.kernel_size = 2
        self.input_features = 1
        self.output_features = 16
        self.filter_size = 3
        self.repetitions = 2
        self.planes = [self.output_features] + [(2 ** i) * self.output_features for i in range(1, self.filter_size)]
        self.num_classes = num_classes
        self.img_size = img_size  
        self._build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Model initialized on device: {self.device}")

    def _build_model(self) -> None:
        print("Building model...")
        self.model = scn.Sequential()
        self.model.add(scn.InputLayer(self.dimensions, self.spatial_size, mode=3))
        print(f"Added InputLayer: dimensions={self.dimensions}, spatial_size={self.spatial_size}")
        self.model.add(scn.SubmanifoldConvolution(self.dimensions, self.input_features, self.output_features, self.filter_size, False))
        print(f"Added SubmanifoldConvolution: in={self.input_features}, out={self.output_features}")
        self.model.add(scn.UNet(self.dimensions, self.repetitions, self.planes, residual_blocks=True, downsample=[self.kernel_size, 2]))
        print(f"Added UNet: planes={self.planes}")
        self.model.add(scn.BatchNormReLU(self.planes[0]))
        print(f"Added BatchNormReLU: features={self.planes[0]}")
        self.model.add(scn.OutputLayer(self.dimensions))
        print("Added OutputLayer")
        self.linear = torch.nn.Linear(self.planes[0], self.num_classes)
        print(f"Added Linear layer: in={self.planes[0]}, out={self.num_classes}")

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        coords, features = input
        print(f"Forward pass: coords shape: {coords.shape}, dtype: {coords.dtype}, device: {coords.device}")
        print(f"Forward pass: features shape: {features.shape}, dtype: {features.dtype}, device: {features.device}")
        print(f"coords min: {coords.min().item()}, max: {coords.max().item()}")
        
        # Validate input ranges for SparseConvNet
        batch_idx = coords[:, 0]  # Batch index
        spatial_coords = coords[:, 1:3]  # y, x coordinates
        assert spatial_coords.min() >= 0 and spatial_coords.max() < self.img_size, \
            f"Spatial coords out of bounds: min {spatial_coords.min()}, max {spatial_coords.max()}"
        assert batch_idx.min() >= 0, f"Batch index negative: min {batch_idx.min()}"
        
        x = self.model((coords, features))
        print(f"Model output shape: {x.shape}")
        assert x.shape[1] == self.planes[0], f"Expected {self.planes[0]} channels, got {x.shape[1]}"
        
        x = self.linear(x)
        print(f"Final output shape: {x.shape}")
        return x

# Instantiate UResNet
# img_size=60 to accommodate max x (~59), num_classes=32 to match original 32 channels
model = UResNet(img_size=60, num_classes=32)

# Messages remain the same
msgs = [[" X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
         " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
         " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
         " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
         " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "],
        [" XXX              XXXXX      x   x     x  xxxxx  xxx ",
         " X  X  X   XXX       X       x   x x   x  x     x  x ",
         " XXX                X        x   xxxx  x  xxxx   xxx ",
         " X     X   XXX       X       x     x   x      x    x ",
         " X     X          XXXX   x   x     x   x  xxxx     x "]]

# Create Nx3 and Nx1 vectors for InputLayer approach
locations = []
features = []
for batchIdx, msg in enumerate(msgs):
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                locations.append([y, x, batchIdx])
                features.append([1])
locations = torch.LongTensor(locations)  # Stays on CPU
features = torch.FloatTensor(features).to(device)  # Moves to GPU if available

# Pass directly to UResNet (no separate InputLayer needed)
output = model([locations, features])
print('Output shape:', output.shape)

# Create batch for BLInputLayer approach
batch = []
for batchIdx, msg in enumerate(msgs):
    l, f = [], []
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                l.append([y, x])  # Locations
                f.append([1])     # Features
    batch.append([torch.LongTensor(l), torch.FloatTensor(f)])
batch = scn.prepare_BLInput(batch)  # Combines into [coords, features]
batch[1] = batch[1].to(device)  # Features to device

# Pass directly to UResNet (no separate BLInputLayer needed)
output = model(batch)
print('Output shape:', output.shape)