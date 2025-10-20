import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class wide_basic(nn.Module):
    """WideResNet basic residual block."""
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        # Shortcut when shape changes
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=False))
        out = self.conv2(F.relu(self.bn2(self.dropout(out)), inplace=False))
        out += self.shortcut(x)
        return out


class NetworkBlock(nn.Module):
    """Stack of wide_basic blocks."""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            s = stride if i == 0 else 1
            inp = in_planes if i == 0 else out_planes
            layers.append(block(inp, out_planes, dropout_rate, s))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Wide_ResNet(nn.Module):
    """WideResNet backbone that returns a global-average-pooled feature vector."""
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_channel=3):
        super(Wide_ResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4 for WideResNet."
        n = (depth - 4) // 6
        k = widen_factor

        # Stage widths: [16, 16*k, 32*k, 64*k]
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = nStages[0]

        # Configurable input channel for grayscale inputs (e.g., MNIST)
        self.conv1 = conv3x3(in_channel, nStages[0], stride=1)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], wide_basic, 1, dropout_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], wide_basic, 2, dropout_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], wide_basic, 2, dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])

        # Note: We do not include a classification linear layer here.
        # This module serves as a feature extractor; the caller adds heads.

        self.feature_dim = nStages[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Return L2-un-normalized backbone feature vector (B, C)."""
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn1(out), inplace=False)
        out = F.avg_pool2d(out, out.size(2))  # global average pooling
        out = out.view(out.size(0), -1)
        return out  # Caller may normalize or pass to projection/classifier heads


def build_wideresnet(depth, widen_factor, dropout, num_classes, in_channel=3):
    """Factory for WideResNet backbone. Returns a feature extractor."""
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return Wide_ResNet(depth=depth,
                    widen_factor=widen_factor,
                    dropout_rate=dropout,
                    num_classes=num_classes,
                    in_channel=in_channel)
