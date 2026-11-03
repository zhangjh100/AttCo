"""Official EMIDEC Case*/Images and Contours NIfTI layout; native labels 0..4."""
from collections import OrderedDict
from pathlib import Path
import csv
import random
import numpy as np
import SimpleITK as sitk
from dataset import ACDCDataset

CLASS_NAMES = ('Cavity', 'MYO', 'Infarction', 'NoReflow')

def discover_cases(root):
    root = Path(root).expanduser().resolve()
    cases = {}
    for images in root.rglob('Images'):
        if not images.is_dir():
            continue
        patient = images.parent.name
        files = sorted(p for p in images.iterdir() if p.name.endswith(('.nii', '.nii.gz')))
        if len(files) != 1:
            raise ValueError(f'Expected one NIfTI image in {images}')
        image = files[0]
        masks = [images.parent/'Contours'/image.name]
        masks += [p for p in (images.parent/'Contours').glob('*.nii*') if p not in masks]
        masks = [p for p in masks if p.is_file()]
        if len(masks) > 1 or patient in cases:
            raise ValueError(f'Ambiguous EMIDEC case: {patient}')
        cases[patient] = (image, masks[0] if masks else None)
    if not cases:
        raise FileNotFoundError(f'No EMIDEC Case*/Images/*.nii[.gz] found under {root}')
    return cases

def ensure_folds(root, path):
    path = Path(path)
    if path.is_file():
        return
    cases = discover_cases(root)
    patients = sorted(p for p, (_, mask) in cases.items() if mask is not None)
    if len(patients) < 5:
        raise ValueError('At least five labelled EMIDEC cases are required for five-fold evaluation')
    assignments = {}
    rng = random.Random(100)
    offset = 0
    for group in ('N', 'P', 'other'):
        ids = [p for p in patients if ('N' if p.startswith('Case_N') or p.endswith('N') else 'P' if p.startswith('Case_P') or p.endswith('P') else 'other') == group]
        rng.shuffle(ids)
        for i, patient in enumerate(ids):
            assignments[patient] = (i + offset) % 5
        offset = (offset + len(ids)) % 5
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID'] + [f'Fold_{i}' for i in range(5)])
        writer.writerows([p] + [int(assignments[p] != i) for i in range(5)] for p in patients)

class EMIDECDataset(ACDCDataset):
    num_classes = 5
    def __init__(self, path_root, listPatients, transforms=None, mode='train',
                 output_size=(224,224), normalize=True, cache_size=1, require_target=None):
        if mode not in ('train','val','test') or cache_size < 0:
            raise ValueError('Invalid mode or cache size')
        self.patients_ID = list(map(str, listPatients))
        if len(set(self.patients_ID)) != len(self.patients_ID):
            raise ValueError('Duplicate patient IDs')
        self.path_root = str(Path(path_root).expanduser().resolve())
        self.transforms, self.mode = transforms, mode
        self.output_size = None if output_size is None else tuple(output_size)
        self.normalize, self.cache_size = normalize, cache_size
        self.require_target = mode != 'test' if require_target is None else require_target
        self._cache, self.volumes, self.slice_index = OrderedDict(), {}, []
        cases = discover_cases(path_root)
        for patient in self.patients_ID:
            if patient not in cases:
                raise FileNotFoundError(f'Missing EMIDEC case {patient}')
            image_path, mask_path = cases[patient]
            if self.require_target and mask_path is None:
                raise FileNotFoundError(f'Missing Contours for {patient}; official test cases have no public labels')
            image = sitk.ReadImage(str(image_path))
            if image.GetDimension() != 3:
                raise ValueError(f'Expected 3D EMIDEC image: {patient}')
            record = dict(patient_id=patient, phase='LGE', frame_index=0,
                          image_path=str(image_path), mask_path=str(mask_path) if mask_path else '',
                          spacing=np.asarray(image.GetSpacing()), origin=np.asarray(image.GetOrigin()),
                          direction=np.asarray(image.GetDirection()), volume_shape=np.asarray(image.GetSize()[::-1]))
            self.volumes[patient] = record
            self.slice_index.extend((patient,z) for z in range(image.GetSize()[2]))

    def _region_masks(self, target):
        return {name.lower(): (target == i).unsqueeze(0).float() for i,name in enumerate(CLASS_NAMES,1)}
