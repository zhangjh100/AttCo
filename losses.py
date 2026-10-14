import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def _prepare_target(logits, label, ignore_lb):
    if logits.ndim not in (4, 5):
        raise ValueError('Expected logits [B,C,H,W] or [B,C,D,H,W]')
    if label.ndim == logits.ndim and label.shape[1] == 1:
        label = label[:, 0]
    if label.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError('Target spatial shape must match logits')
    label = label.to(device=logits.device)
    if label.is_floating_point() and not torch.equal(label, label.round()):
        raise ValueError('Target must contain integer class labels')
    label = label.long()
    valid = label != ignore_lb
    if ((label[valid] < 0) | (label[valid] >= logits.shape[1])).any():
        raise ValueError('Target class is outside logits channel range')
    return label, valid


class CE_GeneralizedSoftDiceLoss(nn.Module):
    """Center-slice CE + foreground Dice; accepts raw logits, no wavelet loss."""

    def __init__(self, alpha=0.5, ignore_lb=255):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        self.alpha, self.ignore_lb = alpha, ignore_lb
        self.diceloss = GeneralizedSoftDiceLoss(ignore_lb=ignore_lb)

    def forward(self, logits, label):
        label, valid = _prepare_target(logits, label, self.ignore_lb)
        ce = F.cross_entropy(logits.float(), label, ignore_index=self.ignore_lb, reduction='none')
        ce = (ce * valid).sum() / valid.sum().clamp_min(1)
        return self.alpha * ce + (1 - self.alpha) * self.diceloss(logits, label)


class CE_GeneralizedSoftDiceLoss_v2(nn.Module):
    def __init__(self, alpha=0.5):
        super(CE_GeneralizedSoftDiceLoss_v2, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def diceloss(self, probs, labels):
        N = len(probs)
        smooth = 1.0
        m1 = probs.view(N, -1)
        m2 = labels.view(N, -1)
        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
        return (1 - score).sum() / N

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, D, H, W)
        args: label: tensor of shape(N, D, H, W)
        '''
        # print(logits.shape, label.shape)
        prob = torch.softmax(logits, dim=1)[:, 1:]
        loss = self.diceloss(prob, label) * (1 - self.alpha)

        label = label.squeeze(1).type(torch.LongTensor).to(logits.device)

        loss += self.ce_loss(logits, label) * self.alpha
        return loss


class BCE_SoftDiceLoss_v1(nn.Module):
    def __init__(self, alpha=0.5):
        super(BCE_SoftDiceLoss_v1, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.diceloss = SoftDiceLoss_v1()
        self.alpha = alpha

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, D, H, W)
        args: label: tensor of shape(N, D, H, W)
        '''
        loss = self.bce_loss(logits, label) + self.diceloss(logits, label)
        return loss


class GeneralizedSoftDiceLoss(nn.Module):
    def __init__(self, p=1, smooth=0.0001, reduction='mean', weight=True, ignore_lb=255):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Invalid reduction')
        self.p, self.smooth, self.reduction = p, smooth, reduction
        self.weight, self.ignore_lb = weight, ignore_lb

    def forward(self, logits, label):
        label, valid = _prepare_target(logits, label, self.ignore_lb)
        logits = logits.float()
        safe_label = label.masked_fill(~valid, 0)
        mask = valid.unsqueeze(1)
        one_hot = torch.zeros_like(logits).scatter_(1, safe_label.unsqueeze(1), 1) * mask
        probs = torch.softmax(logits, dim=1) * mask
        dims = tuple(range(2, logits.ndim))
        numer = (probs * one_hot).sum(dims)
        denom = (probs.pow(self.p) + one_hot.pow(self.p)).sum(dims)
        if self.weight:
            # Retain inverse-volume weighting; absent classes have finite weight 1.
            weights = one_hot.sum(dims).clamp_min(1).reciprocal()
            numer, denom = numer * weights, denom * weights
        numer, denom = numer[:, 1:].sum(1), denom[:, 1:].sum(1)
        loss = 1 - (2 * numer + self.smooth) / (denom + self.smooth)
        usable = valid.flatten(1).any(1)
        loss = loss * usable
        if self.reduction == 'mean':
            return loss.sum() / usable.sum().clamp_min(1)
        return loss.sum() if self.reduction == 'sum' else loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, logits, labels):
        N = len(logits)
        logits = nn.Sigmoid()(logits)

        logits_flat = logits.view(N, -1)
        labels_flat = labels.view(N, -1)
        return self.bce_loss(logits_flat, labels_flat)


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        loss = (abs(y_true - y_pred)).sum() / y_true.shape[0]
        return loss


class SoftDiceLoss_v1(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_v1, self).__init__()

    def forward(self, logits, labels):
        N = len(logits)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        m1 = probs.view(N, -1)
        m2 = labels.view(N, -1)
        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
        return (1 - score).sum() / N


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        N = len(inputs)
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(N, -1)
        targets = targets.view(N, -1)

        intersection = (inputs * targets).sum(dim=1)
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss.sum() / N

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


