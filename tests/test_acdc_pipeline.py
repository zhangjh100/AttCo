"""Run: python -m unittest discover -s tests -v (requires training dependencies)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader
from dataset import ACDCDataset
from transforms import Compose, Mirroring, ToTensor
from losses import CE_GeneralizedSoftDiceLoss
from metrics import ACDCVolumeDice
from models.Wave_ACDC import Wave


class ACDCPipelineTest(unittest.TestCase):
    def test_adjacent_slices_phases_boundaries_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'patient001'
            folder.mkdir()
            (folder / 'Info.cfg').write_text('ED: 1\nES: 2\n')
            for frame, offset in ((1, 0), (2, 10)):
                image = np.stack([np.full((64, 64), z + offset, np.float32) for z in range(3)])
                mask = np.zeros((3, 64, 64), dtype=np.uint8)
                mask[:, 8:16, 8:16] = 1
                mask[:, 20:28, 20:28] = 2
                mask[:, 32:40, 32:40] = 3
                for suffix, array in (('', image), ('_gt', mask)):
                    sitk.WriteImage(sitk.GetImageFromArray(array),
                                    str(folder / f'patient001_frame{frame:02d}{suffix}.nii.gz'))
            data = ACDCDataset(tmp, ['patient001'], normalize=False, output_size=(64, 64))
            self.assertEqual(len(data), 6)
            for idx, expected in ((0, [0, 0, 1]), (1, [0, 1, 2]),
                                  (2, [1, 2, 2]), (3, [10, 10, 11])):
                self.assertEqual(data[idx]['input'][:, 0, 0].tolist(), expected)
                self.assertEqual(tuple(data[idx]['target'].shape), (64, 64))
            augmented = ACDCDataset(tmp, ['patient001'], normalize=False,
                output_size=(64, 64), transforms=Compose([Mirroring(p=1), ToTensor()]))
            self.assertTrue(torch.equal(augmented[1]['target'], data[1]['target'].flip((0, 1))))
            batch = next(iter(DataLoader(data, batch_size=1)))
            # Full architecture, center-slice loss and backward without BN updates.
            model = Wave(baseChannel=8).eval()
            output = model(batch['input'])
            self.assertEqual(tuple(output.shape), (1, 4, 64, 64))
            loss = CE_GeneralizedSoftDiceLoss()(output, batch['target'])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())

    def test_volume_dice_is_batch_independent(self):
        pred = np.array([[[1, 0]], [[0, 0]], [[3, 2]]])
        target = np.array([[[1, 0]], [[1, 0]], [[3, 2]]])
        a, b = ACDCVolumeDice(), ACDCVolumeDice()
        ids = ['ED', 'ED', 'ES']
        a.update(pred, target, ids)
        for i in range(3):
            b.update(pred[i:i+1], target[i:i+1], ids[i:i+1])
        np.testing.assert_allclose(a.compute(), b.compute())
        np.testing.assert_allclose(a.compute()[0], [2/3, 1, 1])


if __name__ == '__main__':
    unittest.main()
