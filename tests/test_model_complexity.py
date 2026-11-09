"""Analytical reference and state restoration checks for complexity measurement."""
import unittest

import torch
from torch import nn

from model_complexity import input_shape, measure_complexity, parse_bool


class ComplexityTest(unittest.TestCase):
    def test_layouts_and_boolean(self):
        self.assertEqual(input_shape('2d', (64, 96)), (1, 1, 64, 96))
        self.assertEqual(input_shape('2.5d', (64, 96)), (1, 3, 64, 96))
        self.assertEqual(input_shape('3d', volume_size=(32, 48, 64)),
                         (1, 1, 32, 48, 64))
        self.assertFalse(parse_bool('False'))
        self.assertTrue(parse_bool('true'))

    def test_conv_reference_and_mixed_modes(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 3, bias=False), nn.Identity())
        model.train()
        model[1].eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = measure_complexity(model, (1, 1, 8, 8))
        self.assertEqual(result['params'], 18)
        self.assertEqual(result['status'], 'ok')
        self.assertAlmostEqual(result['gflops'], 2 * 6 * 6 * 9 / 1e9)
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_partial_never_claims_total(self):
        class Sin(nn.Module):
            def forward(self, x):
                return x.sin()
        result = measure_complexity(Sin(), (1, 1, 8, 8))
        self.assertEqual(result['status'], 'partial')
        self.assertIsNone(result['gflops'])
        self.assertIn('aten::sin', result['unsupported_ops'])

    def test_3d_checkpoint_bypass_and_restore(self):
        class Checkpointed(nn.Module):
            def __init__(self):
                super().__init__()
                self.outside_block_checkpointing = True
                self.inside_block_checkpointing = False
                self.conv = nn.Conv3d(1, 2, 3, bias=False)

            def forward(self, x):
                if self.outside_block_checkpointing:
                    raise RuntimeError('checkpoint path must not be traced')
                return self.conv(x)

        model = Checkpointed().train()
        result = measure_complexity(model, (1, 1, 8, 8, 8))
        self.assertEqual(result['status'], 'ok')
        self.assertAlmostEqual(result['gflops'], 2 * 6 ** 3 * 27 / 1e9)
        self.assertTrue(model.outside_block_checkpointing)
        self.assertFalse(model.inside_block_checkpointing)
        self.assertTrue(model.training)

        model.conv = nn.Conv3d(3, 2, 3)
        result = measure_complexity(model, (1, 1, 8, 8, 8))
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(model.outside_block_checkpointing)
        self.assertTrue(model.training)

    def test_failure_restores_training(self):
        class Broken(nn.Module):
            def forward(self, x):
                raise RuntimeError('test failure')
        model = Broken().train()
        result = measure_complexity(model, (1, 1, 8, 8))
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
