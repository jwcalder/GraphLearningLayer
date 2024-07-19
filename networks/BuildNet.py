"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet, BasicBlock, Bottleneck
from .wrn import build_wideresnet
from .cifarcnn import CNN

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def wrn_28_2(**kwargs):
    return build_wideresnet(28,2,0,**kwargs)

def wrn_28_8(**kwargs):
    return build_wideresnet(28,8,0,**kwargs)

def cifarcnn(**kwargs):
    return CNN(**kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
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
        if name in ['wrn-28-2', 'wrn-28-8', 'cifarcnn']:
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



