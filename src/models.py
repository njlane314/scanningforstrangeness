import torch
import torch.nn as nn
import torch.nn.functional as F

def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def dropout(prob):
    return nn.Dropout(prob)

def reinit_layer(layer, leak=0.0, use_kaiming_normal=True):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        if use_kaiming_normal:
            nn.init.kaiming_normal_(layer.weight, a=leak)
        else:
            nn.init.kaiming_uniform_(layer.weight, a=leak)
        if layer.bias is not None:
            layer.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, k_pad=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.identity = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=1)
        reinit_layer(self.conv1)
        reinit_layer(self.conv2)
    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.relu(x + identity)

class TransposeConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, k_pad=1):
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=k_size, padding=k_pad, output_padding=1, stride=2),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True)
        )
        reinit_layer(self.block[0])
    def forward(self, x):
        return self.block(x)

class Sigmoid(nn.Module):
    def __init__(self, out_range=None):
        super(Sigmoid, self).__init__()
        if out_range is not None:
            self.low, self.high = out_range
            self.range = self.high - self.low
        else:
            self.low = None
            self.high = None
            self.range = None
    def forward(self, x):
        if self.low is not None:
            return torch.sigmoid(x) * self.range + self.low
        else:
            return torch.sigmoid(x)

class UResNetEncoder(nn.Module):
    def __init__(self, in_dim, n_filters=16, drop_prob=0.1):
        super(UResNetEncoder, self).__init__()
        self.ds_conv_1 = ConvBlock(in_dim, n_filters)
        self.ds_conv_2 = ConvBlock(n_filters, 2 * n_filters)
        self.ds_conv_3 = ConvBlock(2 * n_filters, 4 * n_filters)
        self.ds_conv_4 = ConvBlock(4 * n_filters, 8 * n_filters)
        self.ds_maxpool = maxpool()
        self.ds_dropout = dropout(drop_prob)
    def forward(self, x):
        conv_stack = {}
        x = self.ds_conv_1(x)
        conv_stack['s1'] = x.clone()
        x = self.ds_maxpool(x)
        x = self.ds_dropout(x)
        x = self.ds_conv_2(x)
        conv_stack['s2'] = x.clone()
        x = self.ds_maxpool(x)
        x = self.ds_dropout(x)
        x = self.ds_conv_3(x)
        conv_stack['s3'] = x.clone()
        x = self.ds_maxpool(x)
        x = self.ds_dropout(x)
        x = self.ds_conv_4(x)
        conv_stack['s4'] = x.clone()
        x = self.ds_maxpool(x)
        x = self.ds_dropout(x)
        return x, conv_stack

class UResNetDecoder(nn.Module):
    def __init__(self, n_filters=16, drop_prob=0.1, n_classes=1):
        super(UResNetDecoder, self).__init__()
        self.bridge = ConvBlock(8 * n_filters, 16 * n_filters)
        self.us_tconv_4 = TransposeConvBlock(16 * n_filters, 8 * n_filters)
        self.us_conv_4 = ConvBlock(16 * n_filters, 8 * n_filters)
        self.us_tconv_3 = TransposeConvBlock(8 * n_filters, 4 * n_filters)
        self.us_conv_3 = ConvBlock(8 * n_filters, 4 * n_filters)
        self.us_tconv_2 = TransposeConvBlock(4 * n_filters, 2 * n_filters)
        self.us_conv_2 = ConvBlock(4 * n_filters, 2 * n_filters)
        self.us_tconv_1 = TransposeConvBlock(2 * n_filters, n_filters)
        self.us_conv_1 = ConvBlock(2 * n_filters, n_filters)
        self.us_dropout = dropout(drop_prob)
        self.output = nn.Conv2d(n_filters, n_classes, kernel_size=1)
    def forward(self, x, conv_stack):
        x = self.bridge(x)
        x = self.us_tconv_4(x)
        x = torch.cat([x, conv_stack['s4']], dim=1)
        x = self.us_dropout(x)
        x = self.us_conv_4(x)
        x = self.us_tconv_3(x)
        x = torch.cat([x, conv_stack['s3']], dim=1)
        x = self.us_dropout(x)
        x = self.us_conv_3(x)
        x = self.us_tconv_2(x)
        x = torch.cat([x, conv_stack['s2']], dim=1)
        x = self.us_dropout(x)
        x = self.us_conv_2(x)
        x = self.us_tconv_1(x)
        x = torch.cat([x, conv_stack['s1']], dim=1)
        x = self.us_dropout(x)
        x = self.us_conv_1(x)
        x = self.output(x)
        return x

class UResNetFull(nn.Module):
    def __init__(self, in_dim, n_classes, n_filters=16, drop_prob=0.1):
        super(UResNetFull, self).__init__()
        self.encoder = UResNetEncoder(in_dim, n_filters, drop_prob)
        self.decoder = UResNetDecoder(n_filters, drop_prob, n_classes)
    def forward(self, x):
        x, conv_stack = self.encoder(x)
        x = self.decoder(x, conv_stack)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)
    
class SimCLRModel(nn.Module):
    def __init__(self, in_channels, feature_dim, projection_hidden_dim=512, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = UResNetEncoder(in_channels)
        self.projection_head = ProjectionHead(feature_dim, projection_hidden_dim, projection_dim)
    def forward(self, x):
        features, _ = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        projections = self.projection_head(pooled)
        return F.normalize(projections, dim=1)