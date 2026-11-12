"""Check auxiliary inputs and decoder substitutions for WaveDe ablations."""
import unittest

import torch
from torch import nn

from models.baselines import MODEL_REGISTRY, build_model
from models.WaveDe_ACDC import DeformableTransformer2D


class WaveDeAblationsTest(unittest.TestCase):
    def test_new_variants(self):
        for name in ('WaveDe-NoDeform', 'WaveDe-RawNeighbors'):
            for classes in (4, 5):  # ACDC and EMIDEC
                with self.subTest(model=name, classes=classes):
                    self.assertEqual(MODEL_REGISTRY[name], '2.5d')
                    model = build_model(name, base_channels=8,
                                        num_classes=classes).eval()
                    captured = []
                    hook = model.encoder_diff.register_forward_pre_hook(
                        lambda module, args: captured.append(args[0].detach().clone()))
                    x = torch.randn(1, 3, 64, 64)
                    with torch.no_grad():
                        y = model(x)
                    hook.remove()
                    self.assertEqual(tuple(y.shape), (1, classes, 64, 64))
                    expected = (x[:, (0, 2)] if name == 'WaveDe-RawNeighbors'
                                else torch.cat((x[:, 1:2] - x[:, 0:1],
                                                x[:, 2:3] - x[:, 1:2]), dim=1))
                    torch.testing.assert_close(captured[0], expected)
                    count = sum(isinstance(m, DeformableTransformer2D)
                                for m in model.modules())
                    self.assertEqual(count, 0 if name == 'WaveDe-NoDeform' else 5)
                    if name == 'WaveDe-NoDeform':
                        for block in model.decoder:
                            self.assertIsInstance(block.transformer[0], nn.Conv2d)
                        self.assertIsInstance(model.final_transformer[0], nn.Conv2d)


if __name__ == '__main__':
    unittest.main()
