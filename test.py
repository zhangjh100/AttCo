"""Evaluate registered ACDC models on held-out patients and save original-geometry volumes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import dataset
import transforms
import metrics
from main import MODEL_REGISTRY, build_model, validate_model_config


def main():
    parser = argparse.ArgumentParser(description='ACDC 2.5D center-slice evaluation')
    parser.add_argument('--path_image', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--modelname', choices=tuple(MODEL_REGISTRY), default='Wave',
                        help='Must match the architecture used for training: Wave or WaveDe')
    parser.add_argument('--csv_path', default='ACDC_Training_5folds.csv')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--output_path', default='./test_predictions')
    args = parser.parse_args()
    try:
        validate_model_config(args.modelname, args.base_channels)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.image_size) < 64 or any(n % 32 for n in args.image_size):
        parser.error('--image_size must contain multiples of 32, at least 64')
    if args.test_batch_size < 1 or args.num_workers < 0 or args.base_channels < 8:
        parser.error('Invalid batch size, worker count or base channels')
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available()
                          else f'cuda:{args.gpu_id}')
    frame = pd.read_csv(args.csv_path)
    patients = frame.loc[frame[f'Fold_{args.fold}'] == 0, 'ID'].tolist()
    if not patients:
        raise ValueError('No held-out patients')
    data = dataset.ACDCDataset(args.path_image, patients, mode='val',
                              transforms=transforms.ToTensor(), output_size=args.image_size)
    loader = DataLoader(data, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = build_model(args.modelname, args.base_channels)
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
        rows.append(dict(volume_id=volume_id, patient_id=record['patient_id'],
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
    pd.DataFrame(rows).to_csv(output_path / f'test_results_fold_{args.fold}.csv', index=False)
    mean_dice = scores.compute().mean(axis=0)
    print('Original-grid patient-phase volume Dice:', dict(zip(('RV', 'MYO', 'LV'), mean_dice)))
    print('Mean Dice:', float(mean_dice.mean()))


if __name__ == '__main__':
    main()
