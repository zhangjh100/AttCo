"""M&Ms labelled 4D cine series, sampled at the recorded zero-based ED/ES frames."""
import csv
from collections import OrderedDict
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from dataset import ACDCDataset

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = {'ACDC': 'ACDC_Training_5folds.csv', 'MnM': 'MnMs_Training_5folds.csv'}


def read_phase_image(record, mask=False):
    image = sitk.ReadImage(record['mask_path' if mask else 'image_path'])
    if record.get('cine_4d'):
        if image.GetDimension() != 4:
            raise ValueError(f'Expected 4D MnM cine: {image.GetSize()}')
        frame = record['frame_index']
        if not 0 <= frame < image.GetSize()[3]:
            raise ValueError(f'ED/ES frame {frame} outside cine with {image.GetSize()[3]} frames')
        image = sitk.Extract(image, list(image.GetSize()[:3]) + [0], [0, 0, 0, frame])
    if mask and record.get('cine_4d'):
        array = sitk.GetArrayFromImage(image)
        if not np.isin(array, (0, 1, 2, 3)).all():
            raise ValueError('MnM mask must use labels 0,1,2,3')
        # Native MnM 1=LV,2=MYO,3=RV -> shared 1=RV,2=MYO,3=LV.
        remapped = array.copy()
        remapped[array == 1] = 3
        remapped[array == 3] = 1
        result = sitk.GetImageFromArray(remapped)
        result.CopyInformation(image)
        image = result
    return image


class MnMDataset(ACDCDataset):
    def __init__(self, path_root, listPatients, transforms=None, mode='train',
                 output_size=(224,224), normalize=True, cache_size=1,
                 phases=('ED','ES'), require_target=None, metadata_path=None):
        if mode not in ('train','val','test') or cache_size < 0:
            raise ValueError('Invalid dataset mode or cache size')
        self.patients_ID = list(map(str, listPatients))
        if len(set(self.patients_ID)) != len(self.patients_ID):
            raise ValueError('Duplicate patient IDs')
        self.path_root = str(Path(path_root).expanduser().resolve())
        self.transforms, self.mode = transforms, mode
        self.output_size = None if output_size is None else tuple(output_size)
        self.normalize, self.cache_size = normalize, cache_size
        self.require_target = mode != 'test' if require_target is None else require_target
        self._cache, self.volumes, self.slice_index = OrderedDict(), {}, []
        metadata = Path(metadata_path) if metadata_path else PROJECT_ROOT / 'MnMs_metadata.csv'
        with metadata.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        if len({r['patient_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate patient IDs in MnM metadata')
        rows = {r['patient_id']: r for r in rows}
        root = Path(self.path_root)
        roots = (root, root/'Training', root/'Training'/'Labeled', root/'Labeled')
        phases = (phases,) if isinstance(phases, str) else phases
        if not phases or any(p not in ('ED','ES') for p in phases):
            raise ValueError('phases must contain ED and/or ES')
        for patient in self.patients_ID:
            if patient not in rows:
                raise ValueError(f'Missing MnM ED/ES metadata for {patient}: {metadata}')
            row = rows[patient]
            matches = []
            for base in roots:
                for folder in (base/patient, base):
                    for suffix in ('.nii', '.nii.gz'):
                        image = folder / f'{patient}_sa{suffix}'
                        mask = folder / f'{patient}_sa_gt{suffix}'
                        if image.is_file() and (mask.is_file() or not self.require_target):
                            matches.append((image.resolve(), mask.resolve()))
            matches = list(dict.fromkeys(matches))
            if len(matches) != 1:
                raise FileNotFoundError(f'Expected one MnM cine/mask pair for {patient} under {root}; found {len(matches)}')
            image_path, mask_path = matches[0]
            for phase in dict.fromkeys(phases):
                record = dict(patient_id=patient, phase=phase, frame_index=int(row[phase]),
                              image_path=str(image_path), mask_path=str(mask_path), cine_4d=True)
                image = read_phase_image(record)
                if self.require_target:
                    mask = read_phase_image(record, mask=True)
                    for getter in ('GetSize','GetSpacing','GetOrigin','GetDirection'):
                        if not np.allclose(getattr(image,getter)(), getattr(mask,getter)()):
                            raise ValueError(f'MnM image/mask geometry mismatch: {patient}/{phase}')
                record.update(spacing=np.asarray(image.GetSpacing()), origin=np.asarray(image.GetOrigin()),
                              direction=np.asarray(image.GetDirection()), volume_shape=np.asarray(image.GetSize()[::-1]))
                key = f'{patient}_{phase}'
                self.volumes[key] = record
                self.slice_index.extend((key,z) for z in range(image.GetSize()[2]))
