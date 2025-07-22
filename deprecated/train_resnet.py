





### scp mpid_epoch_10.pt nlane@uboonegpvm03.fnal.gov:/exp/uboone/app/users/nlane/production/strangeness_mcc9/srcs/ubana/ubana/searchingforstrangeness/.
### scp nlane@uboonegpvm05.fnal.gov:/exp/uboone/app/users/nlane/analysis/rarexsec-analysis/filtered_numu_cc_events.root .



import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(out_channels, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(out_channels, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out

class MPID(nn.Module):
    def __init__(self, num_classes=6, dropout=0.5):
        super(MPID, self).__init__()

        def make_stage(in_channels, out_channels, num_blocks, first_stride):
            layers = []
            layers.append(BasicBlock(in_channels, out_channels, stride=first_stride))
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(out_channels, out_channels, stride=1))
            return nn.Sequential(*layers)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = make_stage(64, 64, 2, 1)   # 128x128, 64 channels
        self.stage2 = make_stage(64, 96, 2, 2)   # 64x64, 96 channels
        self.stage3 = make_stage(96, 128, 2, 2)  # 32x32, 128 channels
        self.stage4 = make_stage(128, 160, 2, 2) # 16x16, 160 channels
        self.stage5 = make_stage(160, 192, 2, 2) # 8x8, 192 channels

        self.features = nn.Sequential(
            self.conv1,
            self.gn1,
            self.relu,
            self.maxpool,
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5
        )

        self.dropout_features_output = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192 * 8 * 8, 192 * 8),
            nn.Dropout(dropout),
            nn.Linear(192 * 8, 192),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout_features_output(x)
        x = self.classifier(x)
        return x



