"""WaveFour: WaveDe topology with Wave's Fourier refinement in the difference decoder.

Image decoding retains two-level sym2 DWT/IWT refinement. Each of the four
difference decoder stages and the final full-resolution stage replaces
DeformableTransformer2D with the exact SpectralSCSE implementation from Wave.
Raw encoder skips and transposed-convolution upsampling are unchanged.
Input: [B,3,H,W] or [B,1,3,H,W]. Output: center logits [B,C,H,W].
Unlike Wave, FFT operates on decoder states rather than encoder skip features.
"""
from .WaveDe_ACDC import WaveDe
from .Wave_ACDC import SpectralSCSE


class WaveFour(WaveDe):
    def __init__(self, inChannel=1, outChannel=4, baseChannel=24,
                 wavelet='sym2', kernels=(3,5,7), dilations=(2,2,2)):
        # heads=1 avoids imposing a deformable-attention divisibility constraint
        # on a model whose attention modules are replaced before use.
        super().__init__(inChannel, outChannel, baseChannel, wavelet,
                         kernels, dilations, heads=1)
        for block in self.decoder:
            channels = block.diff_up.in_channels
            block.transformer = SpectralSCSE(channels)
        self.final_transformer = SpectralSCSE(baseChannel)
