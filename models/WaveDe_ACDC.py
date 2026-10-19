"""WaveDe: decoder wavelets + deformable transformer for ACDC 2.5D.

Input [B,3,H,W] (previous, center, next) or [B,1,3,H,W].
Output raw center-slice logits [B,outChannel,H,W]. Encoders are unchanged.
Skips pass raw encoder features directly into elementwise addition.
Decoder upsampling maps to encoder channel counts; skips have no learned operations.
Image decoder: two-level sym2 wavelet refinement -> bilinear upsample -> raw image skip addition.
Difference decoder: single-scale deformable transformer -> ConvTranspose2d -> raw difference skip addition.
The two decoded branches are fused at each scale. No Fourier transform is used.
This is a dense single-scale deformable-attention transformer, not the full
multi-scale object-detection Deformable DETR. Sampling uses PyTorch grid_sample;
no custom CUDA extension or downloaded weights are required.

Run smoke check: python -m models.WaveDe_ACDC
Training constructors: WaveDe(...), or Wave(...) for the original interface.
Original Wave checkpoints are not strictly compatible with this architecture.
"""
import math
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .Wave_ACDC import (ResNet50Encoder, MobileNetV2Encoder, DWT2D, IWT2D,
                            MultiKernelDilated, HighBandMultiKernel,
                            conv_norm_act, param_network)
else:
    from Wave_ACDC import (ResNet50Encoder, MobileNetV2Encoder, DWT2D, IWT2D,
                           MultiKernelDilated, HighBandMultiKernel,
                           conv_norm_act, param_network)


class DecoderWaveletRefinement(nn.Module):
    """Move the original two-level subband processing into decoder states.

    Unlike WaveletSkip, this operates on one fused decoder feature map.
    First LL: multi-kernel dilated processing. Second DWT: four-band residual.
    First-level details: independent ordinary multi-kernel signed refinements.
    IWT reconstructs at the current scale; it does NOT perform decoder upsampling.
    """
    def __init__(self, channels, wavelet, kernels, dilations):
        super().__init__()
        self.dwt, self.iwt = DWT2D(wavelet), IWT2D(wavelet)
        self.low = MultiKernelDilated(channels, channels, kernels, dilations)
        self.second = MultiKernelDilated(4 * channels, 4 * channels, kernels, dilations)
        self.highs = nn.ModuleList([
            HighBandMultiKernel(channels, channels, kernels) for _ in range(3)])
        self.fuse = conv_norm_act(channels, channels, 1)

    def forward(self, x):
        bands = self.dwt(x)
        low = self.low(bands[0])
        packed = torch.cat(self.dwt(low), dim=1)
        low = self.iwt((packed + self.second(packed)).chunk(4, dim=1), low.shape[-2:])
        highs = [block(band) for block, band in zip(self.highs, bands[1:])]
        return x + self.fuse(self.iwt((low, *highs), x.shape[-2:]))


class DeformableAttention2D(nn.Module):
    """Per-query, per-head learned offsets and attention over K sampled values.

    Pixel-center references and offsets in pixel units are converted to [-1,1]
    coordinates for bilinear grid_sample(align_corners=False). Weights normalize
    across sampling points. Heads are processed together, avoiding dense NxN maps.
    """
    def __init__(self, channels, heads=4, points=4):
        super().__init__()
        if heads < 1 or points < 1 or channels % heads:
            raise ValueError('Positive heads/points required; channels must divide heads')
        self.heads, self.points = heads, points
        self.value = nn.Conv2d(channels, channels, 1)
        self.offsets = nn.Conv2d(channels, heads * points * 2, 1)
        self.weights = nn.Conv2d(channels, heads * points, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        angles = torch.arange(heads, dtype=torch.float32) * (2 * math.pi / heads)
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        directions = directions / directions.abs().amax(dim=-1, keepdim=True)
        initial = directions[:, None] * torch.arange(1, points + 1)[None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(initial.flatten())

    def forward(self, x):
        b, c, h, w = x.shape
        m, k = self.heads, self.points
        offsets = self.offsets(x).reshape(b, m, k, 2, h, w)
        offsets = offsets.permute(0, 1, 4, 5, 2, 3)
        attention = self.weights(x).reshape(b, m, k, h, w).softmax(dim=2)
        attention = attention.permute(0, 1, 3, 4, 2).reshape(b*m, 1, h*w, k)
        # Sample in float32 under AMP; preserve float64 for numerical checks.
        value = self.value(x)
        sample_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=sample_dtype),
                                    torch.arange(w, device=x.device, dtype=sample_dtype), indexing='ij')
            reference = torch.stack(((xx + .5) / w, (yy + .5) / h), dim=-1)
            scale = x.new_tensor((w, h), dtype=sample_dtype)
            grid = 2 * (reference[None, None, :, :, None] + offsets.to(sample_dtype) / scale) - 1
            grid = grid.reshape(b*m, h*w, k, 2)
            values = value.to(sample_dtype).reshape(b*m, c//m, h, w)
            sampled = F.grid_sample(values, grid, mode='bilinear', padding_mode='zeros',
                                    align_corners=False)
            result = (sampled * attention.to(sample_dtype)).sum(-1).reshape(b, c, h, w)
        return self.output(result.to(value.dtype))


class DeformableTransformer2D(nn.Module):
    """Spatial position encoding + pre-norm deformable attention + residual FFN."""
    def __init__(self, channels, heads=4, points=4):
        super().__init__()
        self.position = nn.Conv2d(2, channels, 1)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = DeformableAttention2D(channels, heads, points)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 4*channels, 1), nn.GELU(),
                                 nn.Conv2d(4*channels, channels, 1))

    @staticmethod
    def normalize(norm, x):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid((torch.arange(h, device=x.device, dtype=x.dtype)+.5)/h,
                                (torch.arange(w, device=x.device, dtype=x.dtype)+.5)/w,
                                indexing='ij')
        position = self.position(torch.stack((xx, yy), dim=0).unsqueeze(0))
        x = x + self.attention(self.normalize(self.norm1, x) + position)
        return x + self.ffn(self.normalize(self.norm2, x))


class WaveDeDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_skip_channels, diff_skip_channels,
                 wavelet, kernels, dilations, heads, points):
        super().__init__()
        # Project before wavelet processing to bound decoder parameter/memory cost.
        self.image_projection = conv_norm_act(in_channels, out_channels, 1)
        self.wavelet = DecoderWaveletRefinement(out_channels, wavelet, kernels, dilations)
        self.transformer = DeformableTransformer2D(in_channels, heads, points)
        # k=3,p=1,s=2 produces 2n-1 or 2n using output_size, matching ceil-downsampled encoders.
        # All channel adaptation is on the decoder path, never on encoder skips.
        self.image_up_channels = nn.Conv2d(out_channels, image_skip_channels, 1)
        self.diff_up = nn.ConvTranspose2d(in_channels, diff_skip_channels, 3, stride=2, padding=1)
        # Decoder processing AFTER raw feature addition bounds subsequent widths.
        self.image_refine = conv_norm_act(image_skip_channels, out_channels, 1)
        self.diff_refine = conv_norm_act(diff_skip_channels, out_channels, 1)
        self.fuse = conv_norm_act(2*out_channels, out_channels, 1)

    def forward(self, image, diff, image_skip, diff_skip):
        size = image_skip.shape[-2:]
        image = self.wavelet(self.image_projection(image))
        image = F.interpolate(self.image_up_channels(image), size=size,
                              mode='bilinear', align_corners=False)
        diff = self.transformer(diff)
        diff = self.diff_up(diff, output_size=diff_skip.shape)
        if image.shape != image_skip.shape or diff.shape != diff_skip.shape:
            raise ValueError('Direct additive skips require identical B/C/H/W')
        image = image + image_skip
        diff = diff + diff_skip
        image = self.image_refine(image)
        diff = self.diff_refine(diff)
        return self.fuse(torch.cat((image, diff), dim=1)), diff


class WaveDe(nn.Module):
    def __init__(self, inChannel=1, outChannel=4, baseChannel=24,
                 wavelet='sym2', kernels: Sequence[int]=(3, 5, 7),
                 dilations: Sequence[int]=(2, 2, 2), heads=4, points=4):
        super().__init__()
        if inChannel != 1 or outChannel < 2 or baseChannel < 8:
            raise ValueError('Requires inChannel=1, outChannel>=2, baseChannel>=8')
        if heads < 1 or baseChannel % heads or points < 1:
            raise ValueError('baseChannel must divide positive heads; points must be positive')
        self.encoder_slice = ResNet50Encoder(1)
        self.encoder_diff = MobileNetV2Encoder(in_channels=2)
        widths = tuple(baseChannel*k for k in (1, 2, 4, 8, 16))
        channels, diff_channels = ResNet50Encoder.channels, MobileNetV2Encoder.channels
        # Preserve the original joint bottleneck, with an additional difference state.
        self.bottleneck = conv_norm_act(3*channels[-1]+diff_channels[-1], widths[-1], 1)
        self.diff_bottleneck = conv_norm_act(diff_channels[-1], widths[-1], 1)
        self.decoder = nn.ModuleList([WaveDeDecoderBlock(widths[i+1], widths[i], 3*channels[i], diff_channels[i],
            wavelet, kernels, dilations, heads, points) for i in (3, 2, 1, 0)])
        self.final_wavelet = DecoderWaveletRefinement(widths[0], wavelet, kernels, dilations)
        self.final_transformer = DeformableTransformer2D(widths[0], heads, points)
        self.final_diff_up = nn.ConvTranspose2d(widths[0], widths[0], 3, stride=2, padding=1)
        self.full_resolution = conv_norm_act(2*widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], outChannel, 1)

    def forward(self, x):
        if x.ndim == 5 and x.shape[1:3] == (1, 3):
            x = x[:, 0]
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('Expected [B,3,H,W] or [B,1,3,H,W]')
        if not x.is_floating_point():
            raise TypeError('Input must be floating-point MRI intensities')
        if min(x.shape[-2:]) < 32:
            raise ValueError('Input H/W must be at least 32; training batch size 1 needs >=64')
        b, _, h, w = x.shape
        difference = torch.cat((x[:, 1:2]-x[:, 0:1], x[:, 2:3]-x[:, 1:2]), dim=1)
        encoded = self.encoder_slice(x.reshape(3*b, 1, h, w))
        # Preserve previous/center/next channel order at every encoder level.
        images = [f.reshape(b, 3*f.shape[1], *f.shape[-2:]) for f in encoded]
        differences = self.encoder_diff(difference)
        image = self.bottleneck(torch.cat((images[-1], differences[-1]), dim=1))
        diff = self.diff_bottleneck(differences[-1])
        for block, i in zip(self.decoder, (3, 2, 1, 0)):
            image, diff = block(image, diff, images[i], differences[i])
        image = F.interpolate(self.final_wavelet(image), size=(h, w),
                              mode='bilinear', align_corners=False)
        diff = self.final_transformer(diff)
        diff = self.final_diff_up(diff, output_size=(b, diff.shape[1], h, w))
        return self.head(self.full_resolution(torch.cat((image, diff), dim=1)))

    def freeze_encoder_batchnorm(self):
        """Call after model.train() if encoder BatchNorm statistics should be frozen."""
        for encoder in (self.encoder_slice, self.encoder_diff):
            for module in encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


# Preserve the constructor name used by Wave_ACDC clients.
Wave = WaveDe

if __name__ == '__main__':
    torch.manual_seed(0)
    model = WaveDe(baseChannel=8).eval()
    param_network(model)
    with torch.no_grad():
        for size in ((64, 64), (65, 67)):
            output = model(torch.randn(1, 3, *size))
            assert output.shape == (1, 4, *size)
            print('Center logits:', tuple(output.shape))
