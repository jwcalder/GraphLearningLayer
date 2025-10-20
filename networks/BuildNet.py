"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18, resnet34, resnet50, \
    resnet20, resnet32, resnet44, resnet56, resnet110
from .vgg import vgg11, vgg13, vgg16, vgg19
from .wrn import build_wideresnet
# from .cifarcnn import CNN
from .preactresnet import (
    preactresnet18,
    preactresnet34,
    preactresnet50,
    preactresnet101,
    preactresnet152,
)

# ---- Local builder aliases for convenience ----
def wrn_28_2(**kwargs):
    """WideResNet-28x2 feature backbone."""
    return build_wideresnet(28, 2, 0, **kwargs)

def wrn_28_8(**kwargs):
    """WideResNet-28x8 feature backbone."""
    return build_wideresnet(28, 8, 0, **kwargs)

# def cifarcnn(**kwargs):
#     """Simple CIFAR-style CNN backbone."""
#     return CNN(**kwargs)


# ---- Model registry (name -> [builder, feature_dim]) ----
model_dict = {
    # ImageNet-style (4-stage) ResNets
    'resnet18':   [resnet18,  512],
    'resnet34':   [resnet34,  512],
    'resnet50':   [resnet50,  2048],

    # CIFAR-specific (3-stage) ResNets; last stage width = 64
    'resnet20':   [resnet20,   64],
    'resnet32':   [resnet32,   64],
    'resnet44':   [resnet44,   64],
    'resnet56':   [resnet56,   64],
    'resnet110':  [resnet110,  64],

    # VGG family (feature dim = 512)
    'vgg11': [vgg11, 512],
    'vgg13': [vgg13, 512],
    'vgg16': [vgg16, 512],
    'vgg19': [vgg19, 512],

    # Pre-activation ResNets (feature-only wrappers)
    'preactresnet18':  [preactresnet18,  512],
    'preactresnet34':  [preactresnet34,  512],
    'preactresnet50':  [preactresnet50,  2048],
    'preactresnet101': [preactresnet101, 2048],
    'preactresnet152': [preactresnet152, 2048],

    # Wide ResNets and simple CIFAR CNN
    # Note: The feature dim equals the last stage width (64*k).
    'wrn-28-2': [wrn_28_2, 128],
    'wrn-28-8': [wrn_28_8, 512],
    # 'cifarcnn': [cifarcnn, 128],
}


class buildnet(nn.Module):
    """Backbone + projection head (+ optional classifier)."""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, num_classes=10,
                 softmax=True, include_classifier=True, in_channel=3):
        """
        Args:
            name: backbone key in model_dict.
            head: 'linear' | 'mlp' | 'no' projection head after features.
            feat_dim: output feature dimension of the projection head.
            num_classes: number of classes for optional classifier.
            softmax: whether to apply softmax to classifier output (only if classifier is included).
            include_classifier: if True, add a small classifier head; else only (pred=None, features) are returned.
            in_channel: input channel count (e.g., 1 for MNIST, 3 for RGB CIFAR).
        """
        super(buildnet, self).__init__()
        model_fun, dim_in = model_dict[name]

        # Backbones that require num_classes in their constructors
        needs_num_classes = [
            'wrn-28-2', 
            'wrn-28-8', 
            # 'cifarcnn',
            'preactresnet18', 
            'preactresnet34', 
            'preactresnet50',
            'preactresnet101', 
            'preactresnet152'
        ]

        # Backbones that support configurable in_channel (we pass it through)
        supports_in_channel = {
            # ResNets (both 4-stage and 3-stage)
            'resnet18','resnet34','resnet50',
            'resnet20','resnet32','resnet44','resnet56','resnet110',
            # VGG family
            'vgg11','vgg13','vgg16','vgg19',
            # PreAct ResNets (after our change)
            'preactresnet18','preactresnet34','preactresnet50','preactresnet101','preactresnet152',
            # WideResNet (after our change)
            'wrn-28-2','wrn-28-8',
            # If your CNN supports in_channel, you can add 'cifarcnn' here
        }

        # Instantiate encoder with appropriate kwargs
        if name in needs_num_classes and name in supports_in_channel:
            # Requires num_classes and supports in_channel
            self.encoder = model_fun(num_classes=num_classes, in_channel=in_channel)
        elif name in needs_num_classes and name not in supports_in_channel:
            # Requires num_classes only (kept for compatibility)
            self.encoder = model_fun(num_classes=num_classes)
        elif name in supports_in_channel:
            # Supports in_channel only
            self.encoder = model_fun(in_channel=in_channel)
        else:
            # No extra kwargs
            self.encoder = model_fun()

        # Projection head
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'no':
            self.head = nn.Identity()
            feat_dim = dim_in  # keep external contract: output feat dim equals backbone dim
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

        # Optional classifier head (e.g., for finetuning)
        self.include_classifier = include_classifier
        if include_classifier:
            self.linear = nn.Sequential(
                nn.Linear(feat_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_classes)
            )
        else:
            # Identity avoids tracking any extra trainable parameters
            self.linear = nn.Identity()

        # Only apply softmax if classifier is present
        self.softmax = (softmax and include_classifier)
        if self.softmax:
            print("Softmax is added after the MLP classifier.")

    def forward(self, x):
        # 1) backbone features
        feat = self.encoder(x)
        # 2) projection head
        feat = self.head(feat)

        # 3) optional classifier
        if self.include_classifier:
            pred = self.linear(feat)
            if self.softmax:
                pred = F.softmax(pred, dim=1)
        else:
            pred = None

        # Return (pred, normalized_features)
        return pred, F.normalize(feat, dim=1)
