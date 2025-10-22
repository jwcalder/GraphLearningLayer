# vgg.py
import torch
import torch.nn as nn

# ------------------------------
# VGG backbone for CIFAR/MNIST inputs
# ------------------------------

cfgs = {
    # VGG-11 (A)
    'A': [64, 'M',
          128, 'M',
          256, 256, 'M',
          512, 512, 'M',
          512, 512, 'M'],
    # VGG-13 (B)
    'B': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 'M',
          512, 512, 'M',
          512, 512, 'M'],
    # VGG-16 (D)
    'D': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 256, 'M',
          512, 512, 512, 'M',
          512, 512, 512, 'M'],
    # VGG-19 (E)
    'E': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """VGG backbone that outputs a pooled feature vector (C=512)."""
    def __init__(self, cfg, in_channel=3, use_bn=True):
        super().__init__()
        self.features = self._make_layers(cfg, in_channel, use_bn=use_bn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Parameter initialization policy (parity with your ResNet init)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, in_channel, use_bn=True):
        """Build VGG feature extractor with optional BatchNorm.
        
        Note:
            We set ceil_mode=True for MaxPool2d so that downsampling on 28x28 inputs
            remains valid across 5 pooling stages (28->14->7->4->2->1). This keeps
            CIFAR-10 (32x32) behavior unchanged (32->16->8->4->2->1).
        """
        layers = []
        in_c = in_channel
        for v in cfg:
            if v == 'M':
                # Use ceil_mode=True to avoid invalid 2x2 pooling on a 1x1 map for 28x28 inputs
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            else:
                conv = nn.Conv2d(in_c, v, kernel_size=3, stride=1, padding=1, bias=False)
                if use_bn:
                    layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_c = v
        return nn.Sequential(*layers)

    def forward(self, x):
        # Return pooled features for the MLP head in buildnet
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # shape: (N, 512)
        return out


# --------
# Builders
# --------
def vgg11(**kwargs):
    """VGG-11 backbone (cfg 'A'). Returns pooled features (dim=512)."""
    return VGG(cfgs['A'], **kwargs)

def vgg13(**kwargs):
    """VGG-13 backbone (cfg 'B'). Returns pooled features (dim=512)."""
    return VGG(cfgs['B'], **kwargs)

def vgg16(**kwargs):
    """VGG-16 backbone (cfg 'D'). Returns pooled features (dim=512)."""
    return VGG(cfgs['D'], **kwargs)

def vgg19(**kwargs):
    """VGG-19 backbone (cfg 'E'). Returns pooled features (dim=512)."""
    return VGG(cfgs['E'], **kwargs)
