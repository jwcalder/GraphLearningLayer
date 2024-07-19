"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def entropy(predictions):
    return -torch.sum(predictions * torch.log(predictions + 1e-8)) / predictions.shape[0]
    # mask = (predictions > 0.001)&(predictions < 0.999)
    # return -torch.sum(predictions[mask] * torch.log(predictions[mask]))/predictions.shape[0]

def logsumexp(predictions):
    max_pred = torch.max(predictions, dim=1, keepdim=True).values
    sum_exp = torch.sum(torch.exp(predictions - max_pred), dim=1, keepdim=True)
    logsumexp_per_row = max_pred + torch.log(sum_exp)
    return torch.mean(logsumexp_per_row)

def l2(predictions):
    return -torch.sum(predictions ** 2)/predictions.shape[0]
def sym_CE_loss(pred1, pred2):
    # Step 1: Thresholding to get plabels
    with torch.no_grad():
        plabel1 = torch.argmax(pred1, dim=1)
        plabel2 = torch.argmax(pred2, dim=1)

    # Step 2: Compute cross entropy
    loss1 = F.cross_entropy(pred2, plabel1)
    loss2 = F.cross_entropy(pred1, plabel2)

    # Step 3: Sum the losses
    total_loss = loss1 + loss2

    return total_loss

def custom_ce_loss(softmax_logits, targets):
    # the input logits should sum to 1 (each row)
    # Convert targets to one-hot encoding
    batch_size, num_classes = softmax_logits.shape
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).to(softmax_logits.dtype)

    # Calculate CE loss manually
    loss = -torch.sum(one_hot_targets * torch.log(softmax_logits + 1e-8)) / batch_size
    return loss

