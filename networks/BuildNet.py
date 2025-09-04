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
from .cifarcnn import CNN
from .preactresnet import (
    preactresnet18,
    preactresnet34,
    preactresnet50,
    preactresnet101,
    preactresnet152,
)

def wrn_28_2(**kwargs):
    return build_wideresnet(28,2,0,**kwargs)

def wrn_28_8(**kwargs):
    return build_wideresnet(28,8,0,**kwargs)

def cifarcnn(**kwargs):
    return CNN(**kwargs)


model_dict = {
    # CIFAR-specific 4-stage ResNets
    'resnet18':   [resnet18,  512],
    'resnet34':   [resnet34,  512],
    'resnet50':   [resnet50,  2048],
    # CIFAR-specific 3-stage ResNets
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
    'wrn-28-2': [wrn_28_2, 128],
    'wrn-28-8': [wrn_28_8, 512],
    'cifarcnn': [cifarcnn, 128],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class buildnet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, num_classes=10, softmax=True):
        super(buildnet, self).__init__()
        model_fun, dim_in = model_dict[name]
        needs_num_classes = [
            'wrn-28-2', 'wrn-28-8', 'cifarcnn',
            'preactresnet18', 'preactresnet34', 'preactresnet50',
            'preactresnet101', 'preactresnet152'
        ]
        if name in needs_num_classes:
            # For PreActResNet wrappers, this will reach the wrapper and be forwarded.
            self.encoder = model_fun(num_classes=num_classes)
        else:
            self.encoder = model_fun()
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
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.linear = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        self.softmax = softmax
        if softmax:
            print("Softmax is added after the MLP classifier.")

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        pred = self.linear(feat)
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        return pred, F.normalize(feat, dim=1)



