"""Pre-extract labelled MnM ED/ES phases as uncompressed 3D NIfTI."""
import argparse
import csv
from pathlib import Path
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk

from mnm_data import CACHE_METADATA, PROJECT_ROOT, extract_cine_frame, find_cine_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path_image', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--metadata_path', type=Path, default=PROJECT_ROOT / 'MnMs_metadata.csv')
    parser.add_argument('--csv_path', type=Path, default=PROJECT_ROOT / 'MnMs_Training_5folds.csv')
    args = parser.parse_args()
    output = args.output_path.expanduser().resolve()
    if output.exists():
        parser.error(f'Output directory already exists: {output}')
    with args.metadata_path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    metadata = {row['patient_id']: row for row in rows}
    if len(metadata) != len(rows):
        parser.error('Duplicate patient IDs in MnM metadata')
    with args.csv_path.open(newline='', encoding='utf-8-sig') as handle:
        patients = [row['ID'] for row in csv.DictReader(handle)]
    if not patients or len(set(patients)) != len(patients) or not set(patients) <= metadata.keys():
        parser.error('Split CSV must contain unique patient IDs present in the metadata')

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mnm-extract-', dir=output.parent) as temp:
        stage = Path(temp) / 'volumes'
        stage.mkdir()
        for patient in patients:
            source, source_mask = find_cine_pair(args.path_image.expanduser().resolve(), patient)
            cine, cine_mask = sitk.ReadImage(str(source)), sitk.ReadImage(str(source_mask))
            folder = stage / patient
            folder.mkdir()
            for phase in ('ED', 'ES'):
                frame = int(metadata[patient][phase])
                image = extract_cine_frame(cine, frame)
                mask = extract_cine_frame(cine_mask, frame)
                for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
                    if not np.allclose(getattr(image, getter)(), getattr(mask, getter)()):
                        raise ValueError(f'MnM image/mask geometry mismatch: {patient}/{phase}')
                if not np.isin(sitk.GetArrayViewFromImage(mask), (0, 1, 2, 3)).all():
                    raise ValueError(f'Invalid MnM labels: {patient}/{phase}')
                sitk.WriteImage(image, str(folder / f'{patient}_{phase}.nii'))
                sitk.WriteImage(mask, str(folder / f'{patient}_{phase}_gt.nii'))
        shutil.copyfile(args.metadata_path, stage / CACHE_METADATA)
        stage.rename(output)
    print(f'Wrote ED/ES image and mask volumes for {len(patients)} patients to {output}')


if __name__ == '__main__':
    main()
