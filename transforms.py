"""2.5D transforms: numpy input [3,H,W], center target [H,W]."""
import numpy as np
import torch
from scipy.ndimage import rotate

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

class Mirroring:
    """Flip H and W independently; never reverse the slice/channel axis."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):
        for axis in (1, 2):
            if np.random.random() < self.p:
                sample['input'] = np.flip(sample['input'], axis=axis).copy()
                if 'target' in sample:
                    sample['target'] = np.flip(sample['target'], axis=axis-1).copy()
        return sample

class RandomRotation:
    """One angle for all slices and the center mask, with nearest mask interpolation."""
    def __init__(self, degrees=15, p=0.5):
        self.degrees, self.p = degrees, p
    def __call__(self, sample):
        if np.random.random() < self.p:
            angle = np.random.uniform(-self.degrees, self.degrees)
            sample['input'] = rotate(sample['input'], angle, axes=(1, 2),
                reshape=False, order=1, mode='constant', cval=0, prefilter=False)
            if 'target' in sample:
                sample['target'] = rotate(sample['target'], angle, axes=(0, 1),
                    reshape=False, order=0, mode='constant', cval=0, prefilter=False)
        return sample

class ToTensor:
    def __call__(self, sample):
        image = np.asarray(sample['input'])
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError('Expected input [3,H,W]')
        sample['input'] = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        if 'target' in sample:
            target = np.asarray(sample['target'])
            if target.shape != image.shape[1:]:
                raise ValueError('Expected center target [H,W]')
            if not np.all(target == np.floor(target)):
                raise ValueError('Target must contain integer class labels')
            sample['target'] = torch.from_numpy(np.ascontiguousarray(target, dtype=np.int64))
        return sample
