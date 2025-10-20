#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-activation ResNet (He et al., 2016). This file provides:
- PreActBlock / PreActBottleneck
- PreActResNet class with configurable `in_channel`
- Thin wrappers (preactresnet18/34/50/101/152) that return a feature-only backbone
All comments are in English as requested.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation BasicBlock. No ReLU after shortcut addition."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes,
                                      kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=False)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation Bottleneck block (1x1, 3x3, 1x1)."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion,
                                      kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=False)
        shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        out = self.conv3(F.relu(self.bn3(out), inplace=False))

        out += shortcut
        return out


class PreActResNet(nn.Module):
    """Pre-activation ResNet backbone. Returns a global-average-pooled feature vector."""
    def __init__(self, block, num_blocks, num_classes=10, widen=None, in_channel=3):
        super(PreActResNet, self).__init__()
        # Keep your original layer widths if provided, else use default [64,128,256,512]
        if widen is None:
            self.layers = np.asarray([64, 128, 256, 512])
        else:
            self.layers = np.asarray(widen)
        assert len(self.layers) == 4, "Expected 4 stages in PreActResNet."

        self.in_planes = self.layers[0]

        # Configurable input channels for grayscale inputs (e.g., MNIST)
        self.conv1 = nn.Conv2d(in_channel, self.layers[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Build 4 stages
        self.layer1 = self._make_layer(block, self.layers[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.layers[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.layers[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.layers[3], num_blocks[3], stride=2)

        # Optional classifier in case someone wants logits here; not used in backbone wrappers.
        self.linear = nn.Linear(self.layers[3] * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create one stage with `num_blocks` blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Return features (B, C) and logits (B, num_classes) if needed."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(2))  # global average pooling
        feat = out.view(out.size(0), -1)
        logits = self.linear(feat)
        return logits, feat


# ---- Thin wrappers that expose a "feature-only backbone" interface ----
class _PreactFeatureBackbone(nn.Module):
    """Wrap PreActResNet to expose features only (without returning logits)."""
    def __init__(self, model: PreActResNet):
        super().__init__()
        self.model = model
        # Infer feature dimension from the linear layer shape
        self.feature_dim = self.model.linear.in_features

    def forward(self, x):
        logits, feat = self.model(x)
        return feat  # caller will add projection/classifier heads


def preactresnet18(**kwargs):
    """Feature-only PreActResNet-18 backbone (returns 512-dim features)."""
    num_classes = kwargs.get("num_classes", 10)
    in_channel = kwargs.get("in_channel", 3)
    return _PreactFeatureBackbone(PreActResNet(PreActBlock, [2, 2, 2, 2],
                                               num_classes=num_classes, in_channel=in_channel))

def preactresnet34(**kwargs):
    """Feature-only PreActResNet-34 backbone (returns 512-dim features)."""
    num_classes = kwargs.get("num_classes", 10)
    in_channel = kwargs.get("in_channel", 3)
    return _PreactFeatureBackbone(PreActResNet(PreActBlock, [3, 4, 6, 3],
                                               num_classes=num_classes, in_channel=in_channel))

def preactresnet50(**kwargs):
    """Feature-only PreActResNet-50 backbone (returns 2048-dim features)."""
    num_classes = kwargs.get("num_classes", 10)
    in_channel = kwargs.get("in_channel", 3)
    return _PreactFeatureBackbone(PreActResNet(PreActBottleneck, [3, 4, 6, 3],
                                               num_classes=num_classes, in_channel=in_channel))

def preactresnet101(**kwargs):
    """Feature-only PreActResNet-101 backbone (returns 2048-dim features)."""
    num_classes = kwargs.get("num_classes", 10)
    in_channel = kwargs.get("in_channel", 3)
    return _PreactFeatureBackbone(PreActResNet(PreActBottleneck, [3, 4, 23, 3],
                                               num_classes=num_classes, in_channel=in_channel))

def preactresnet152(**kwargs):
    """Feature-only PreActResNet-152 backbone (returns 2048-dim features)."""
    num_classes = kwargs.get("num_classes", 10)
    in_channel = kwargs.get("in_channel", 3)
    return _PreactFeatureBackbone(PreActResNet(PreActBottleneck, [3, 8, 36, 3],
                                               num_classes=num_classes, in_channel=in_channel))
# --- end Preact wrappers ---
