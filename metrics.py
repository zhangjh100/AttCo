import torch
import torch.nn as nn
from scipy import ndimage
import numpy as np


class DiceMetrics(nn.Module):
    """BraTS labels 0/1/2/3 (raw 4 remapped to 3); returns TC, ED, ET, WT.

    empty_score=0 preserves the original both-empty convention. Slice means
    are training diagnostics and are not equivalent to patient-volume Dice.
    """

    def __init__(self, empty_score=0.0, ignore_lb=255):
        super().__init__()
        self.empty_score, self.ignore_lb = empty_score, ignore_lb

    def dice(self, pred, target):
        pred, target = pred.flatten(1).float(), target.flatten(1).float()
        denom = pred.sum(1) + target.sum(1)
        score = 2 * (pred * target).sum(1) / denom.clamp_min(1)
        return torch.where(denom > 0, score, torch.full_like(score, self.empty_score)).mean()

    @torch.no_grad()
    def forward(self, logits, labels):
        if logits.ndim not in (4, 5) or logits.shape[1] != 4:
            raise ValueError('Expected four-class 2D or 3D logits')
        if labels.ndim == logits.ndim and labels.shape[1] == 1:
            labels = labels[:, 0]
        pred = logits.argmax(1)
        if labels.shape != pred.shape:
            raise ValueError('Center label shape must match predicted class map')
        labels = labels.to(pred.device)
        valid = labels != self.ignore_lb
        if ((labels[valid] < 0) | (labels[valid] > 3)).any():
            raise ValueError('Expected remapped BraTS labels 0,1,2,3')
        regions = (
            ((pred == 1) | (pred == 3), (labels == 1) | (labels == 3)),
            (pred == 2, labels == 2),
            (pred == 3, labels == 3),
            (pred > 0, labels > 0),
        )
        return tuple(self.dice(p & valid, t & valid) for p, t in regions)


class DiceMetric_v2(nn.Module):  ### for BraTS dataset
    def __init__(self):  # weight=None, size_average=True):
        super(DiceMetric_v2, self).__init__()
        self.smooth = 1e-12

    def dice(self, probs, labels):
        N = len(probs)
        m1 = probs.view(N, -1)
        m2 = labels.view(N, -1)

        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection) / (m1.sum(dim=1) + m2.sum(dim=1) + self.smooth)
        return score.mean()

    @torch.no_grad()
    def forward(self, logits, labels):
        pred = logits.argmax(1)
        if labels.ndim == logits.ndim and labels.shape[1] == 1:
            labels = labels[:, 0]
        labels = labels.to(logits.device).long()
        if labels.shape != pred.shape:
            raise ValueError('Label shape mismatch')
        # Original metric includes background (micro Dice equals pixel accuracy).
        return (pred == labels).float().mean()


class DiceAccuracy_v1(nn.Module):
    def __init__(self, prob_mode=False):  # weight=None, size_average=True):
        super(DiceAccuracy_v1, self).__init__()
        self.prob_mode = prob_mode

    def forward(self, logits, labels):
        N = len(logits)
        if self.prob_mode:
            probs = logits
        else:
            probs = torch.sigmoid(logits)
        probs = (probs > 0.5).float()
        smooth = 1e-12

        m1 = probs.view(N, -1)
        m2 = labels.view(N, -1)

        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)

        return score.sum() / N


def numeric_score(logits, labels):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()

    FP = ((probs == 1) & (labels == 0)).sum(dim=1)
    FN = ((probs == 0) & (labels == 1)).sum(dim=1)
    TP = ((probs == 1) & (labels == 1)).sum(dim=1)
    TN = ((probs == 0) & (labels == 0)).sum(dim=1)

    return FP, FN, TP, TN


class PrecisionandRecall(nn.Module):
    def __init__(self):
        super(PrecisionandRecall, self).__init__()

    def forward(self, logits, labels):
        N = len(logits)
        logits = logits.view(N, -1)
        labels = labels.view(N, -1)

        FP, FN, TP, TN = numeric_score(logits, labels)

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        return precision.sum() / N, recall.sum() / N


class Recall(nn.Module):
    def __init__(self):
        super(Recall, self).__init__()

    def forward(self, logits, labels):
        N = len(logits)
        logits = logits.view(N, -1)
        labels = labels.view(N, -1)

        FP, FN, TP, TN = numeric_score(logits, labels)
        recall = TP / (TP + FN + 1e-12)

        return recall.sum() / N


class IouAccuracy_v1(nn.Module):
    def __init__(self):  # weight=None, size_average=True):
        super(IouAccuracy_v1, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        probs = (probs > 0.5).float()
        smooth = 1e-12

        N = len(logits)

        m1 = probs.view(N, -1)
        m2 = labels.view(N, -1)

        intersection = (m1 * m2).sum(dim=1)
        union = m1.sum(dim=1) + m2.sum(dim=1) - (m1 * m2).sum(dim=1)
        score = intersection / (union + smooth)

        return score.sum() / N


def border_map(binary_img, neigh=None):
    """Axial-neighbor surface of a 2D or 3D binary mask."""
    mask = np.asarray(binary_img, dtype=bool)
    if mask.ndim not in (2, 3):
        raise ValueError('Expected a 2D or 3D binary mask')
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)


def border_distance(ref, seg):
    border_ref, border_seg = border_map(ref), border_map(seg)
    if border_ref.shape != border_seg.shape:
        raise ValueError('Masks must have the same shape')
    if not border_ref.any() or not border_seg.any():
        raise ValueError('Use Hausdorff_distance for empty-mask handling')
    distance_ref = ndimage.distance_transform_edt(~border_ref)
    distance_seg = ndimage.distance_transform_edt(~border_seg)
    return border_seg * distance_ref, border_ref * distance_seg


def Hausdorff_distance(ref, seg):
    """Maximum symmetric surface distance in pixels/voxels, not HD95."""
    ref, seg = np.asarray(ref, dtype=bool), np.asarray(seg, dtype=bool)
    if ref.shape != seg.shape:
        raise ValueError('Masks must have the same shape')
    if not ref.any() and not seg.any():
        return 0.0
    if not ref.any() or not seg.any():
        return float('inf')
    a, b = border_distance(ref, seg)
    return float(max(a.max(), b.max()))
