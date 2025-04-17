class AttentionWeighting(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(AttentionWeighting, self).__init__()
        self.conv = scn.SubmanifoldConvolution(2, in_channels, in_channels // reduction, 3, bias=False)
        self.bn = scn.BatchNormReLU(in_channels // reduction)
        self.out_conv = scn.Convolution(
            dimension=2,
            nIn=in_channels // reduction,
            nOut=1,
            filter_size=1,
            filter_stride=1,
            bias=True
        )
        self.sigmoid = scn.Sigmoid()

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.out_conv(x)
        x = self.sigmoid(x)
        return x

class AttentionWeightedBCELoss(nn.Module):
    def __init__(self, feature_channels):
        super(AttentionWeightedBCELoss, self).__init__()
        self.attention = AttentionWeighting(feature_channels)

    def forward(self, inputs, targets, features):
        attn_weights = self.attention(features)
        attn_weights_dense = attn_weights.dense(shape=[inputs.size(0), 1, inputs.size(2), inputs.size(3)])[:, 0]
        weights = attn_weights_dense.unsqueeze(1).expand(-1, inputs.size(1), -1, -1)
        weights = weights.clamp(0.1, 10.0)
        return F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction='mean')