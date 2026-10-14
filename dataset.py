import numpy as np
from torch.utils.data import Dataset
import pathlib as plb
import os
import SimpleITK as sitk
import scipy
import torch


class MedDataset(Dataset):
    """BraTS single-modality 2.5D: input [3,H,W], center target [H,W].

    Select modality explicitly (default flair). SimpleITK arrays are [Z,H,W].
    transforms must accept numpy input [3,H,W] and target [H,W], apply the
    SAME geometry to all three slices and the mask, and preserve these shapes.
    Old 3D transforms are not compatible. Difference images belong in model.forward.
    cache_size is the number of volumes cached PER DataLoader worker.
    """

    def __init__(self, path_root, listPatients, transforms=None, mode='train',
                 modality='flair', output_size=(224, 224), normalize=True,
                 cache_size=1):
        from collections import OrderedDict
        if mode not in ('train', 'val', 'test'):
            raise ValueError("mode must be train, val or test")
        if modality not in ('flair', 't1', 't1ce', 't2'):
            raise ValueError("modality must be flair, t1, t1ce or t2")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.path_root = path_root
        self.patients_ID = [str(p) for p in listPatients]
        self.transforms, self.mode, self.modality = transforms, mode, modality
        self.output_size = None if output_size is None else tuple(output_size)
        if self.output_size is not None and (len(self.output_size) != 2 or
                                             any(v <= 0 for v in self.output_size)):
            raise ValueError("output_size must be positive (H,W) or None")
        self.normalize, self.cache_size = normalize, cache_size
        self._cache = OrderedDict()
        self.slice_index = []
        for patient in self.patients_ID:
            reader = sitk.ImageFileReader()
            reader.SetFileName(self._path(patient, self.modality))
            reader.ReadImageInformation()
            size = reader.GetSize()  # SimpleITK header order: X,Y,Z
            if len(size) != 3:
                raise ValueError(f"Expected 3D volume for {patient}, got {size}")
            self.slice_index.extend((patient, z) for z in range(size[2]))

    def _path(self, patient, suffix):
        return os.path.join(self.path_root, patient, f'{patient}_{suffix}.nii.gz')

    def __len__(self):
        return len(self.slice_index)

    def _load_volume(self, patient):
        if patient in self._cache:
            self._cache.move_to_end(patient)
            return self._cache[patient]
        image = self.read_data(self._path(patient, self.modality)).astype(np.float32)
        if image.ndim != 3 or not np.isfinite(image).all():
            raise ValueError(f"Invalid 3D image for {patient}")
        if self.normalize:
            # One set of volume statistics preserves relative adjacent-slice intensity.
            foreground = image != 0
            if foreground.any():
                values = image[foreground]
                image[foreground] = (values - values.mean()) / max(float(values.std()), 1e-6)
        mask = None
        if self.mode != 'test':
            mask = self.read_data(self._path(patient, 'seg'))
            if image.shape != mask.shape:
                raise ValueError(f"Image/mask shape mismatch for {patient}")
            if not np.isin(mask, (0, 1, 2, 4)).all():
                raise ValueError("Expected original BraTS labels 0,1,2,4")
            mask = mask.astype(np.int64)
            mask[mask == 4] = 3
        result = (image, mask)
        if self.cache_size:
            self._cache[patient] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        from scipy.ndimage import zoom
        patient, z = self.slice_index[index]
        image, mask = self._load_volume(patient)
        depth = image.shape[0]
        indices = [max(0, z - 1), z, min(depth - 1, z + 1)]
        stack = image[indices].copy()
        target = None if mask is None else mask[z].copy()
        original_size = stack.shape[-2:]
        if self.output_size is not None and original_size != self.output_size:
            factors = tuple(n / old for n, old in zip(self.output_size, original_size))
            stack = zoom(stack, (1, *factors), order=1, prefilter=False)
            if target is not None:
                target = zoom(target, factors, order=0, prefilter=False)
        sample = {'id': patient, 'slice_index': z, 'input': stack,
                  'original_size': np.asarray(original_size, dtype=np.int64)}
        if target is not None:
            sample['target'] = target
        if self.transforms is not None:
            sample = self.transforms(sample)
        x = sample['input']
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.ascontiguousarray(x))
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError("2.5D transforms must return input [3,H,W]; remove old 3D transforms")
        sample['input'] = x.float().contiguous()
        if target is not None:
            y = sample['target']
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(np.ascontiguousarray(y))
            if y.ndim != 2 or tuple(y.shape) != tuple(x.shape[-2:]):
                raise ValueError("Center target must be [H,W] aligned with input")
            if not bool(((y >= 0) & (y <= 3) & (y == y.round())).all()):
                raise ValueError("Transforms must preserve integer class labels 0,1,2,3")
            sample['target'] = y.long().contiguous()
            sample.update(self._region_masks(sample['target']))
        return sample

    def _region_masks(self, target):
        """BraTS masks after raw label 4 is remapped to class 3."""
        return {
            'wt': (target > 0).unsqueeze(0).float(),
            'tc': ((target == 1) | (target == 3)).unsqueeze(0).float(),
            'et': (target == 3).unsqueeze(0).float(),
        }

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))


import pathlib as plb


def find_studies(path_to_data):
    # find all studies
    dicom_root = plb.Path(path_to_data)
    patient_dirs = list(dicom_root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        # print(sub_dirs)
        study_dirs.extend(sub_dirs)

        # dicom_dirs = dicom_dirs.append(dir.glob('*'))
    return study_dirs


class AutoPETDataset(Dataset):  ## AutoPET Challenge
    """A class for fetching data samples.

    Parameters
    ----------
    paths_to_samples : list
        A list wherein each element is a tuple with two (three) `pathlib.Path` objects for a single patient.
        The first one is the path to the CT image, the second one - to the PET image. If `mode == 'train'`, a path to
        a ground truth mask must be provided for each patient.
    transforms
        Transformations applied to each data sample.
    mode : str
        Must be `train` or `test`. If `train`, a ground truth mask is loaded using a path from `paths_to_samples` and
        added to a sample.
        If `test`, an additional information (an affine array), that describes the position of the image data
        in a reference space, is added to each data sample. Ground truth masks are not loaded in this mode.

    Returns
    -------
    dict
        A dictionary corresponding to a data sample.
        Keys:
            id : A patient's ID.
            input : A numpy array containing CT & PET images stacked along the last (4th) dimension.
            target : A numpy array containing a ground truth mask.
            affine : A numpy array with the position of the image data in a reference space (needed for resampling).
    """

    def __init__(self, path_root, listPatients, transforms=None, mode='train'):
        self.path_root = path_root
        # self.paths_to_samples = listPatients
        self.paths_to_samples = [fn.replace("/mnt/sdc/data/FDG-PET-CT-Lesions/PositiveCases_preprocessed", path_root)
                                 for fn in listPatients]
        self.transforms = transforms
        if mode not in ['train', 'test', 'val']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode
        # if mode == 'train':
        #     self.num_of_seqs = len(paths_to_samples[0]) - 1
        # else:
        #     self.num_of_seqs = len(paths_to_samples[0])

    def __len__(self):
        return len(self.paths_to_samples)

    def __getitem__(self, index):
        sample = dict()

        id_ = self.paths_to_samples[index]  # .parent.stem
        sample['id'] = id_
        if self.mode == "train":
            ct = np.load(os.path.join(self.paths_to_samples[index], 'CTres.npy'))
            pet = np.load(os.path.join(self.paths_to_samples[index], 'SUV.npy'))

            img = np.stack([ct, pet], axis=-1)
            sample['input'] = img

            mask = np.load(os.path.join(self.paths_to_samples[index], 'SEG.npy'))
            mask = np.expand_dims(mask, axis=3)
            assert img.shape[:-1] == mask.shape[:-1], \
                f"Shape mismatch for the image with the shape {img.shape} and the mask with the shape {mask.shape}."

            sample['target'] = mask

        elif self.mode == "val":
            ct_ori = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.paths_to_samples[index], 'CTres.nii.gz')))
            pet_ori = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.paths_to_samples[index], 'SUV.nii.gz')))
            mask_ori = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.paths_to_samples[index], 'SEG.nii.gz')))

            sample['ct_ori'] = np.flip(ct_ori, axis=0).copy()
            sample['pet_ori'] = np.flip(pet_ori, axis=0).copy()
            sample['mask_ori'] = np.flip(mask_ori, axis=0).copy()

            ct = np.load(os.path.join(self.paths_to_samples[index], 'CTres.npy'))
            pet = np.load(os.path.join(self.paths_to_samples[index], 'SUV.npy'))

            img = np.stack([ct, pet], axis=-1)
            sample['input'] = img

            mask = np.load(os.path.join(self.paths_to_samples[index], 'SEG.npy'))
            mask = np.expand_dims(mask, axis=3)
            assert img.shape[:-1] == mask.shape[:-1], \
                f"Shape mismatch for the image with the shape {img.shape} and the mask with the shape {mask.shape}."

            sample['target'] = mask

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        # if return_numpy:
        #     return nib.load(str(path_to_nifti)).get_fdata()
        # return nib.load(str(path_to_nifti))
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))


class ACDCDataset(MedDataset):
    """Official ACDC layout: root/patientXXX/{Info.cfg,patientXXX_frameYY*.nii.gz}.

    Three spatially adjacent short-axis slices from ONE ED/ES volume, not
    three time frames. Labels are unchanged: 0 background, 1 RV, 2 MYO, 3 LV.
    Pass patient IDs explicitly to enforce patient-level dataset splits.
    output_size is a pixel resize, not physical-spacing resampling.
    """

    def __init__(self, path_root, listPatients, transforms=None, mode='train',
                 output_size=(224, 224), normalize=True, cache_size=1,
                 phases=('ED', 'ES'), require_target=None):
        from collections import OrderedDict
        if mode not in ('train', 'val', 'test'):
            raise ValueError('mode must be train, val or test')
        if isinstance(phases, str):
            phases = (phases,)
        if not phases or any(p not in ('ED', 'ES') for p in phases):
            raise ValueError('phases must contain ED and/or ES')
        if cache_size < 0:
            raise ValueError('cache_size must be non-negative')
        self.path_root = str(path_root)
        self.patients_ID = [str(p) for p in listPatients]
        if len(set(self.patients_ID)) != len(self.patients_ID):
            raise ValueError('Duplicate patient IDs')
        self.transforms, self.mode = transforms, mode
        self.output_size = None if output_size is None else tuple(output_size)
        if self.output_size is not None and (len(self.output_size) != 2 or
                                             any(not isinstance(v, (int, np.integer)) or v <= 0 for v in
                                                 self.output_size)):
            raise ValueError('output_size must be positive integer (H,W) or None')
        self.normalize, self.cache_size = normalize, cache_size
        self.require_target = mode != 'test' if require_target is None else require_target
        self._cache, self.volumes, self.slice_index = OrderedDict(), {}, []
        for patient in self.patients_ID:
            folder = plb.Path(path_root) / patient
            info_path = folder / 'Info.cfg'
            info = {}
            for line in info_path.read_text().splitlines():
                key, separator, value = line.partition(':')
                if separator:
                    info[key.strip()] = value.strip()
            for phase in dict.fromkeys(phases):
                if phase not in info:
                    raise ValueError(f'Missing {phase} in {info_path}')
                frame = int(info[phase])
                stem = f'{patient}_frame{frame:02d}'
                image_path = folder / f'{stem}.nii.gz'
                mask_path = folder / f'{stem}_gt.nii.gz'
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if self.require_target and not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(image_path))
                reader.ReadImageInformation()
                size = reader.GetSize()
                if len(size) != 3:
                    raise ValueError(f'Expected 3D phase volume, got {size}: {image_path}')
                self.volumes[stem] = dict(patient_id=patient, phase=phase, frame_index=frame,
                                          image_path=str(image_path), mask_path=str(mask_path),
                                          spacing=np.asarray(reader.GetSpacing(), dtype=np.float64),
                                          origin=np.asarray(reader.GetOrigin(), dtype=np.float64),
                                          direction=np.asarray(reader.GetDirection(), dtype=np.float64),
                                          volume_shape=np.asarray(size[::-1], dtype=np.int64))
                self.slice_index.extend((stem, z) for z in range(size[2]))

    def _load_volume(self, volume_id):
        if volume_id in self._cache:
            self._cache.move_to_end(volume_id)
            return self._cache[volume_id]
        record = self.volumes[volume_id]
        image_itk = sitk.ReadImage(record['image_path'])
        image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
        if image.ndim != 3 or not np.isfinite(image).all():
            raise ValueError(f'Invalid volume: {volume_id}')
        if self.normalize:
            # Shared statistics across the full phase preserve slice differences.
            foreground = image != 0
            if foreground.any():
                values = image[foreground]
                image[foreground] = (values - values.mean()) / max(float(values.std()), 1e-6)
        mask = None
        if self.require_target:
            mask_itk = sitk.ReadImage(record['mask_path'])
            for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
                if not np.allclose(getattr(image_itk, getter)(), getattr(mask_itk, getter)()):
                    raise ValueError(f'Image/mask geometry mismatch: {volume_id}')
            mask = sitk.GetArrayFromImage(mask_itk)
            if not np.isin(mask, (0, 1, 2, 3)).all():
                raise ValueError('ACDC requires labels 0,1,2,3; do not apply BraTS mapping')
            mask = mask.astype(np.int64)
        result = image, mask
        if self.cache_size:
            self._cache[volume_id] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def _region_masks(self, target):
        """ACDC masks; the inherited sampler dispatches directly to this method."""
        return {
            'rv': (target == 1).unsqueeze(0).float(),
            'myo': (target == 2).unsqueeze(0).float(),
            'lv': (target == 3).unsqueeze(0).float(),
        }

    def __getitem__(self, index):
        # Reuse shared 2.5D stacking, resizing, tensor conversion and validation.
        sample = super().__getitem__(index)
        volume_id = self.slice_index[index][0]
        record = self.volumes[volume_id]
        for key in ('patient_id', 'phase', 'frame_index', 'spacing', 'origin',
                    'direction', 'volume_shape'):
            value = record[key]
            sample[key] = value.copy() if isinstance(value, np.ndarray) else value
        sample['volume_id'] = volume_id
        return sample
