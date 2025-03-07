import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def maxpool() -> nn.MaxPool2d:
    """2×2 max pooling with stride 2."""
    return nn.MaxPool2d(kernel_size=2, stride=2)

def dropout(prob: float) -> nn.Dropout:
    """Dropout layer with the given probability."""
    return nn.Dropout(prob)

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.identity(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.relu(out)
        return out

class TransposeConvBlock(nn.Module):
    """
    An upsampling block using transposed convolution, followed by GroupNorm and ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 padding: int = 1, output_padding: int = 1) -> None:
        super().__init__()
        self.tconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=2,
            padding=padding, output_padding=output_padding, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tconv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class UResNet(nn.Module):
    """
    URes-Net: a U-shaped residual network for segmentation or image-to-image tasks.
    """
    def __init__(self, in_channels: int, num_classes: int, base_filters: int = 16, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.pool = maxpool()
        self.dropout = dropout(drop_prob)
        
        # Encoder (downsampling)
        self.conv1 = ConvBlock(in_channels, base_filters)
        self.conv2 = ConvBlock(base_filters, 2 * base_filters)
        self.conv3 = ConvBlock(2 * base_filters, 4 * base_filters)
        self.conv4 = ConvBlock(4 * base_filters, 8 * base_filters)
        
        # Decoder (upsampling)
        self.bridge = ConvBlock(8 * base_filters, 16 * base_filters)
        self.up4 = TransposeConvBlock(16 * base_filters, 8 * base_filters)
        self.conv4_dec = ConvBlock(16 * base_filters, 8 * base_filters)
        self.up3 = TransposeConvBlock(8 * base_filters, 4 * base_filters)
        self.conv3_dec = ConvBlock(8 * base_filters, 4 * base_filters)
        self.up2 = TransposeConvBlock(4 * base_filters, 2 * base_filters)
        self.conv2_dec = ConvBlock(4 * base_filters, 2 * base_filters)
        self.up1 = TransposeConvBlock(2 * base_filters, base_filters)
        self.conv1_dec = ConvBlock(2 * base_filters, base_filters)
        self.out_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder: apply conv blocks and store outputs for skip connections.
        conv_stack = {}
        x = self.conv1(x)
        conv_stack['s1'] = x.clone()
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        conv_stack['s2'] = x.clone()
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        conv_stack['s3'] = x.clone()
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        conv_stack['s4'] = x.clone()
        x = self.pool(x)
        x = self.dropout(x)
        
        # Decoder: bridge and upsample with skip connections.
        x = self.bridge(x)
        
        x = self.up4(x)
        x = torch.cat([x, conv_stack['s4']], dim=1)
        x = self.dropout(x)
        x = self.conv4_dec(x)
        
        x = self.up3(x)
        x = torch.cat([x, conv_stack['s3']], dim=1)
        x = self.dropout(x)
        x = self.conv3_dec(x)
        
        x = self.up2(x)
        x = torch.cat([x, conv_stack['s2']], dim=1)
        x = self.dropout(x)
        x = self.conv2_dec(x)
        
        x = self.up1(x)
        x = torch.cat([x, conv_stack['s1']], dim=1)
        x = self.dropout(x)
        x = self.conv1_dec(x)
        
        x = self.out_conv(x)
        return x

def uresnet(in_channels: int, num_classes: int, base_filters: int = 16, drop_prob: float = 0.1) -> UResNet:
    return UResNet(in_channels, num_classes, base_filters, drop_prob)