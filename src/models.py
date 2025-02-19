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

class UResNet(nn.Module):
    def __init__(self, in_dim, n_classes, n_filters=16, drop_prob=0.1, y_range=None):
        super(UResNet, self).__init__()
        self.ds_conv_1 = ConvBlock(in_dim, n_filters)
        self.ds_conv_2 = ConvBlock(n_filters, 2 * n_filters)
        self.ds_conv_3 = ConvBlock(2 * n_filters, 4 * n_filters)
        self.ds_conv_4 = ConvBlock(4 * n_filters, 8 * n_filters)
        self.ds_maxpool_1 = maxpool()
        self.ds_maxpool_2 = maxpool()
        self.ds_maxpool_3 = maxpool()
        self.ds_maxpool_4 = maxpool()
        self.ds_dropout_1 = dropout(drop_prob)
        self.ds_dropout_2 = dropout(drop_prob)
        self.ds_dropout_3 = dropout(drop_prob)
        self.ds_dropout_4 = dropout(drop_prob)
        self.bridge = ConvBlock(8 * n_filters, 16 * n_filters)
        self.us_tconv_4 = TransposeConvBlock(16 * n_filters, 8 * n_filters)
        self.us_tconv_3 = TransposeConvBlock(8 * n_filters, 4 * n_filters)
        self.us_tconv_2 = TransposeConvBlock(4 * n_filters, 2 * n_filters)
        self.us_tconv_1 = TransposeConvBlock(2 * n_filters, n_filters)
        self.us_conv_4 = ConvBlock(16 * n_filters, 8 * n_filters)
        self.us_conv_3 = ConvBlock(8 * n_filters, 4 * n_filters)
        self.us_conv_2 = ConvBlock(4 * n_filters, 2 * n_filters)
        self.us_conv_1 = ConvBlock(2 * n_filters, n_filters)
        self.us_dropout_4 = dropout(drop_prob)
        self.us_dropout_3 = dropout(drop_prob)
        self.us_dropout_2 = dropout(drop_prob)
        self.us_dropout_1 = dropout(drop_prob)
        self.output = nn.Conv2d(n_filters, n_classes, kernel_size=1)
    def forward(self, x):
        res = x
        res = self.ds_conv_1(res)
        conv_stack_1 = res.clone()
        res = self.ds_maxpool_1(res)
        res = self.ds_dropout_1(res)
        res = self.ds_conv_2(res)
        conv_stack_2 = res.clone()
        res = self.ds_maxpool_2(res)
        res = self.ds_dropout_2(res)
        res = self.ds_conv_3(res)
        conv_stack_3 = res.clone()
        res = self.ds_maxpool_3(res)
        res = self.ds_dropout_3(res)
        res = self.ds_conv_4(res)
        conv_stack_4 = res.clone()
        res = self.ds_maxpool_4(res)
        res = self.ds_dropout_4(res)
        res = self.bridge(res)
        res = self.us_tconv_4(res)
        res = torch.cat([res, conv_stack_4], dim=1)
        res = self.us_dropout_4(res)
        res = self.us_conv_4(res)
        res = self.us_tconv_3(res)
        res = torch.cat([res, conv_stack_3], dim=1)
        res = self.us_dropout_3(res)
        res = self.us_conv_3(res)
        res = self.us_tconv_2(res)
        res = torch.cat([res, conv_stack_2], dim=1)
        res = self.us_dropout_2(res)
        res = self.us_conv_2(res)
        res = self.us_tconv_1(res)
        res = torch.cat([res, conv_stack_1], dim=1)
        res = self.us_dropout_1(res)
        res = self.us_conv_1(res)
        output = self.output(res)
        return output

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim, input_size=(512,512), base_filters=64):
        super(ResNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size[0], input_size[1])
            dummy = self.conv1(dummy)
            dummy = self.bn1(dummy)
            dummy = self.relu(dummy)
            dummy = self.maxpool(dummy)
            dummy = self.layer1(dummy)
            dummy = self.layer2(dummy)
            dummy = self.layer3(dummy)
            dummy = self.layer4(dummy)
            dummy = self.avgpool(dummy)
            n_features = dummy.view(1, -1).size(1)
        self.fc = nn.Linear(n_features, feature_dim)
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(self._conv_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._conv_block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def _conv_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class EncodedClassifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_planes, hidden_dim):
        super(EncodedClassifier, self).__init__()
        self.encoder = encoder
        self.num_planes = num_planes
        for param in self.encoder.parameters():
            param.requires_grad = False
        agg_dim = feature_dim * self.num_planes
        self.fc1 = nn.Linear(agg_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)    
    def forward(self, x):
        batch_size, n_planes, C, H, W = x.size()
        assert n_planes == self.num_planes
        x = x.view(batch_size * n_planes, C, H, W)
        features = self.encoder(x)
        features = features.view(batch_size, n_planes, -1)
        agg_features = features.view(batch_size, -1)
        out = F.relu(self.fc1(agg_features))
        out = torch.sigmoid(self.fc2(out))
        return out