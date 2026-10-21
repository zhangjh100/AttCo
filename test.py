"""Evaluate registered ACDC models on held-out patients and save original-geometry volumes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import dataset
from comparison_data import build_dataset
import transforms
import metrics
from main import MODEL_REGISTRY, build_model, validate_model_config



def patient_results_table(volume_results):
    """One independent patient per row; retain ED/ES and their arithmetic means.

    Dice values are original-grid 3D scores on [0,1]. Reject missing/duplicate
    phases to avoid silently averaging different numbers of observations.
    """
    keys = ['model', 'fold', 'checkpoint', 'patient_id']
    dice_columns = ['Dice_RV', 'Dice_MYO', 'Dice_LV', 'Dice_mean']
    if volume_results.empty:
        raise ValueError('No volume results to summarize')
    if not volume_results['phase'].isin(['ED', 'ES']).all():
        raise ValueError('Expected ED and ES phases')
    if volume_results.duplicated(keys + ['phase']).any():
        raise ValueError('Duplicate patient-phase result')
    rows = []
    for identifiers, group in volume_results.groupby(keys, sort=True):
        if set(group['phase']) != {'ED', 'ES'}:
            raise ValueError('Missing ED/ES result for patient {}'.format(identifiers[-1]))
        row = dict(zip(keys, identifiers))
        phases = group.set_index('phase')
        for phase in ('ED', 'ES'):
            for column in dice_columns:
                row['{}_{}'.format(column, phase)] = float(phases.loc[phase, column])
        for column in dice_columns:
            row[column] = (row[column + '_ED'] + row[column + '_ES']) / 2
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='ACDC native 2D / 2.5D / 3D evaluation')
    parser.add_argument('--path_image', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--modelname', choices=tuple(MODEL_REGISTRY), default='Wave',
                        help='Must match the architecture used for training')
    parser.add_argument('--csv_path', default='ACDC_Training_5folds.csv')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--volume_size', type=int, nargs=3, default=(32,128,128))
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--output_path', default='./test_predictions')
    args = parser.parse_args()
    try:
        validate_model_config(args.modelname, args.base_channels, args.image_size, args.volume_size)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.image_size) < 64 or any(n % 32 for n in args.image_size):
        parser.error('--image_size must contain multiples of 32, at least 64')
    if args.test_batch_size < 1 or args.num_workers < 0 or args.base_channels < 8:
        parser.error('Invalid batch size, worker count or base channels')
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available()
                          else f'cuda:{args.gpu_id}')
    config_path = Path(args.pretrained).parent / f'config_fold_{args.fold}.json'
    if config_path.is_file():
        saved = json.loads(config_path.read_text())
        if saved['modelname'] != args.modelname:
            parser.error('Checkpoint configuration uses a different modelname')
        size_key = 'volume_size' if MODEL_REGISTRY[args.modelname] == '3d' else 'image_size'
        if tuple(saved[size_key]) != tuple(getattr(args, size_key)):
            parser.error(f'Checkpoint configuration requires --{size_key} ' + ' '.join(map(str, saved[size_key])))
        if args.modelname in ('Wave', 'WaveDe') and saved['base_channels'] != args.base_channels:
            parser.error('Checkpoint configuration uses different base_channels')
    frame = pd.read_csv(args.csv_path)
    patients = frame.loc[frame[f'Fold_{args.fold}'] == 0, 'ID'].tolist()
    if not patients:
        raise ValueError('No held-out patients')
    data = build_dataset(args.modelname, args.path_image, patients, mode='val', volume_size=args.volume_size,
                              transforms=transforms.ToTensor(), output_size=args.image_size)
    loader = DataLoader(data, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = build_model(args.modelname, args.base_channels, args.image_size, args.volume_size)
    state = torch.load(args.pretrained, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}, strict=True)
    model.to(device).eval()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    scores = metrics.ACDCVolumeDice()
    rows = []
    current_id, slices, seen = None, None, set()

    def finish(volume_id, prediction, seen_slices):
        record = data.volumes[volume_id]
        if len(seen_slices) != int(record['volume_shape'][0]):
            raise ValueError(f'Incomplete volume: {volume_id}')
        reference = sitk.ReadImage(record['image_path'])
        target = sitk.GetArrayFromImage(sitk.ReadImage(record['mask_path']))
        scores.update(prediction[None], target[None], [volume_id])
        dice = scores.compute()[-1]
        rows.append(dict(model=args.modelname, fold=args.fold,
                         checkpoint=str(Path(args.pretrained).resolve()),
                         volume_id=volume_id, patient_id=record['patient_id'],
                         phase=record['phase'], Dice_RV=dice[0], Dice_MYO=dice[1],
                         Dice_LV=dice[2], Dice_mean=float(dice.mean())))
        if args.save_pred:
            result = sitk.GetImageFromArray(prediction)
            result.CopyInformation(reference)
            sitk.WriteImage(result, str(output_path / f'{volume_id}_pred.nii.gz'))

    with torch.no_grad():
        for sample in tqdm(loader, desc='ACDC evaluation'):
            logits = model(sample['input'].to(device))
            for i, volume_id in enumerate(sample['volume_id']):
                if MODEL_REGISTRY[args.modelname] == '3d':
                    shape = tuple(int(n) for n in data.volumes[volume_id]['volume_shape'])
                    restored = F.interpolate(logits[i:i+1], size=shape,
                                             mode='trilinear', align_corners=False)
                    prediction = restored.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    finish(volume_id, prediction, set(range(shape[0])))
                    continue
                if volume_id != current_id:
                    if current_id is not None:
                        finish(current_id, slices, seen)
                    current_id = volume_id
                    slices = np.zeros(tuple(data.volumes[volume_id]['volume_shape']), dtype=np.uint8)
                    seen = set()
                z = int(sample['slice_index'][i])
                if z in seen:
                    raise ValueError(f'Duplicate slice: {volume_id}/{z}')
                # Restore continuous class scores before argmax, preserving native H/W.
                restored = F.interpolate(logits[i:i+1], size=slices.shape[1:],
                                         mode='bilinear', align_corners=False)
                slices[z] = restored.argmax(1)[0].cpu().numpy().astype(np.uint8)
                seen.add(z)
    if current_id is not None:
        finish(current_id, slices, seen)
    volume_results = pd.DataFrame(rows)
    patient_results = patient_results_table(volume_results)
    if set(patient_results['patient_id']) != set(patients):
        raise ValueError('Patient results do not match the requested held-out patients')
    volume_csv = output_path / f'test_results_fold_{args.fold}.csv'
    patient_csv = output_path / f'patient_results_fold_{args.fold}.csv'
    volume_results.to_csv(volume_csv, index=False, float_format='%.10f')
    patient_results.to_csv(patient_csv, index=False, float_format='%.10f')
    print('Patient-phase results:', volume_csv)
    print('Patient-level results (one row per patient):', patient_csv)
    print('Patients:', len(patient_results))
    mean_dice = scores.compute().mean(axis=0)
    print('Original-grid patient-phase volume Dice:', dict(zip(('RV', 'MYO', 'LV'), mean_dice)))
    print('Mean Dice:', float(mean_dice.mean()))


if __name__ == '__main__':
    main()
