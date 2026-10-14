"""ACDC 2.5D segmentation: shared ResNet-50 slice encoder + MobileNetV2 difference encoder.

Input: [B,3,H,W], channels [previous, center, next], or [B,1,3,H,W].
Output: center-slice logits [B,outChannel,H,W] (no softmax).
Requires torch, torchvision and PyWavelets. No weights are downloaded.
Shared image encoder has 1 input channel; difference encoder has 2.
ResNet-50 v1.5 stages: [3,4,6,3], features [64,256,512,1024,2048].
Fixed sym2 DWT is applied ONLY over H,W; coefficients remain differentiable.
Three processed LL bands are fused before a second spatial DWT/IWT.
Center high bands reconstruct skips. All decoder upsampling uses learned IWT coefficients.
Matched periodic boundaries replace the original replicate-boundary analysis.
FFT scSE uses magnitude to generate a phase-preserving symmetric spectral mask.
"""
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
import pywt
from torchvision.models import mobilenet_v2


def param_network(model):
    count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {count:,}")
    return count

class MobileNetV2Encoder(nn.Module):
    # 对应 H/2、H/4、H/8、H/16、H/32
    channels = (16, 24, 32, 96, 320)

    def __init__(self, in_channels=2):
        super().__init__()

        # 默认随机初始化，不下载权重
        backbone = mobilenet_v2(weights=None)

        # 首层输入由 RGB 三通道改为两张差分图
        old_conv = backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(
            new_conv.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        backbone.features[0][0] = new_conv

        # 仅保留编码部分，去掉分类头及最后的 1280 通道扩展层
        self.features = backbone.features[:18]
        self.output_indices = (1, 3, 6, 13, 17)

    def forward(self, x):
        outputs = []

        for index, block in enumerate(self.features):
            x = block(x)

            if index in self.output_indices:
                outputs.append(x)

        return tuple(outputs)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1):
        super().__init__()
        out_channels = planes * 4
        self.conv1 = nn.Conv2d(in_channels, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        return self.relu(self.bn3(self.conv3(y)) + identity)


class ResNet50Encoder(nn.Module):
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        current = 64
        for index, (planes, blocks) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3))):
            stride = 1 if index == 0 else 2
            stage = [Bottleneck(current, planes, stride)]
            current = planes * 4
            stage.extend(Bottleneck(current, planes) for _ in range(blocks - 1))
            setattr(self, f"layer{index + 1}", nn.Sequential(*stage))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s0 = self.relu(self.bn1(self.conv1(x)))
        s1 = self.layer1(self.maxpool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s0, s1, s2, s3, s4

    def load_imagenet_state_dict(self, state):
        """Load torchvision-compatible local ResNet50 weights (without fc).

        RGB conv1 is averaged for grayscale; repeated/scaled for 2-channel input.
        This is optional initialization, not a pretrained guarantee for MRI.
        """
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        weight = state["conv1.weight"]
        n = self.conv1.in_channels
        if weight.shape[1] == 3 and n != 3:
            state["conv1.weight"] = weight.mean(1, keepdim=True).repeat(1, n, 1, 1) / n
        self.load_state_dict(state, strict=True)


def conv_norm_act(in_channels, out_channels, kernel_size=3):
    # GroupNorm keeps decoder/difference fusion usable with small batches.
    groups = min(8, out_channels)
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=kernel_size // 2, bias=False),
        nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True))


def wavelet_filters(wavelet):
    wave = pywt.Wavelet(wavelet)
    if not wave.orthogonal:
        raise ValueError("This matched periodic DWT/IWT requires an orthogonal wavelet")
    lo = torch.tensor(wave.dec_lo[::-1], dtype=torch.float32)
    hi = torch.tensor(wave.dec_hi[::-1], dtype=torch.float32)
    return torch.stack([torch.outer(a, b) for a, b in
                        ((lo, lo), (lo, hi), (hi, lo), (hi, hi))])[:, None]


class DWT2D(nn.Module):
    """Matched periodic orthogonal DWT; odd inputs replicate their last row/column.

    Returns LL,LH,HL,HH, each [B,C,ceil(H/2),ceil(W/2)].
    Filters are fixed; all feature operations support autograd.
    """
    def __init__(self, wavelet="sym2"):
        super().__init__()
        self.register_buffer("filters", wavelet_filters(wavelet))

    def forward(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        h, w = x.shape[-2:]
        pad = (self.filters.shape[-1] - 2) // 2
        iy = torch.arange(-pad, h + pad, device=x.device) % h
        ix = torch.arange(-pad, w + pad, device=x.device) % w
        x = x.index_select(-2, iy).index_select(-1, ix)
        y = F.conv2d(x, self.filters.to(x.dtype).repeat(c, 1, 1, 1),
                     stride=2, groups=c)
        y = y.reshape(b, c, 4, h // 2, w // 2)
        return tuple(y[:, :, i] for i in range(4))


class IWT2D(nn.Module):
    """Inverse matched to DWT2D, with periodic boundary overlap folded back.

    output_size crops only the extra row/column introduced for odd inputs.
    """
    def __init__(self, wavelet="sym2"):
        super().__init__()
        self.register_buffer("filters", wavelet_filters(wavelet))

    def forward(self, bands, output_size=None):
        if len(bands) != 4 or any(t.shape != bands[0].shape for t in bands):
            raise ValueError("IWT requires four equally shaped LL/LH/HL/HH tensors")
        b, c, h, w = bands[0].shape
        x = torch.stack(tuple(bands), dim=2).reshape(b, 4 * c, h, w)
        y = F.conv_transpose2d(x, self.filters.to(x.dtype).repeat(c, 1, 1, 1),
                               stride=2, groups=c)
        oh, ow = 2 * h, 2 * w
        pad = (self.filters.shape[-1] - 2) // 2
        ix = (torch.arange(y.shape[-1], device=y.device) - pad) % ow
        y = y.new_zeros(b, c, y.shape[-2], ow).index_add(-1, ix, y)
        iy = (torch.arange(y.shape[-2], device=y.device) - pad) % oh
        y = y.new_zeros(b, c, oh, ow).index_add(-2, iy, y)
        if output_size is not None:
            th, tw = output_size
            if th not in (oh, oh - 1) or tw not in (ow, ow - 1):
                raise ValueError("IWT target must match the corresponding DWT input size")
            y = y[..., :th, :tw]
        return y

class HighBandMultiKernel(nn.Module):
    """Multi-kernel spatial refinement with a linear residual path."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernels=(3, 5, 7)
    ):
        super().__init__()

        if len(kernels) != 3:
            raise ValueError("Exactly three kernels are required")

        if any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("Kernel sizes must be positive odd integers")

        # 基础分支：线性通道投影，不截断负值
        self.projection = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )

        # 三个不同尺寸的普通空间卷积分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                conv_norm_act(in_channels, out_channels, 1),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    dilation=1,
                    bias=False
                ),
                nn.ReLU(inplace=True)
            )
            for k in kernels
        ])

        # 线性输出：允许预测正、负残差
        self.merge = nn.Conv2d(
            3 * out_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        multi_scale = torch.cat(
            [branch(x) for branch in self.branches],
            dim=1
        )

        residual = self.merge(multi_scale)

        # 相加之后不使用 ReLU
        return self.projection(x) + residual

class MultiKernelDilated(nn.Module):
    """Parallel three kernel sizes, configurable dilation per branch.

    A 1x1 projection before each dense dilated conv limits parameter cost.
    """
    def __init__(self, in_channels, out_channels, kernels=(3, 5, 7), dilations=(2, 2, 2)):
        super().__init__()
        if len(kernels) != 3 or len(dilations) != 3:
            raise ValueError("Exactly three kernels and dilation rates are required")
        if any(k <= 0 or k % 2 == 0 for k in kernels) or any(d <= 0 for d in dilations):
            raise ValueError("Kernel sizes must be positive odd integers; dilation must be positive")
        self.branches = nn.ModuleList([
            nn.Sequential(conv_norm_act(in_channels, out_channels, 1),
                          nn.Conv2d(out_channels, out_channels, k,
                                    dilation=d, padding=d * (k // 2), bias=False),
                          nn.ReLU(inplace=True))
            for k, d in zip(kernels, dilations)])
        self.merge = conv_norm_act(3 * out_channels, out_channels, 1)

    def forward(self, x):
        return self.merge(torch.cat([branch(x) for branch in self.branches], 1))


class WaveletSkip(nn.Module):
    """Fuse three LL bands and reconstruct with refined center details."""

    def __init__(
        self,
        channels,
        out_channels,
        wavelet,
        kernels,
        dilations
    ):
        super().__init__()

        self.dwt = DWT2D(wavelet)
        self.iwt = IWT2D(wavelet)

        # 三张 slice 的一级 LL：分别进行多核空洞卷积
        self.first = nn.ModuleList([
            MultiKernelDilated(
                channels, out_channels, kernels, dilations
            )
            for _ in range(3)
        ])

        # 融合三张 slice 的 LL
        self.ll_fuse = conv_norm_act(
            3 * out_channels, out_channels, 1
        )

        # 二级四子带：联合预测残差
        self.second = MultiKernelDilated(
            4 * out_channels,
            4 * out_channels,
            kernels,
            dilations
        )

        # 目标 slice 的一级 LH、HL、HH：
        # 分别使用独立的普通多核空间卷积模块
        self.high_processors = nn.ModuleList([
            HighBandMultiKernel(
                in_channels=channels,
                out_channels=out_channels,
                kernels=kernels
            )
            for _ in range(3)
        ])

        self.fuse = conv_norm_act(
            out_channels, out_channels, 1
        )

    def forward(self, features):
        # features 顺序：[previous, center, next]
        if len(features) != 3:
            raise ValueError("Expected three slice features")

        if any(x.shape != features[1].shape for x in features):
            raise ValueError("All slice features must have the same shape")

        target_size = features[1].shape[-2:]

        # 1. 三张 slice 分别做一级 DWT
        # 每个元素为 (LL, LH, HL, HH)
        bands = [self.dwt(x) for x in features]

        # 2. 分别处理一级 LL，再拼接融合
        processed_lls = [
            block(sb[0])
            for block, sb in zip(self.first, bands)
        ]

        ll = self.ll_fuse(
            torch.cat(processed_lls, dim=1)
        )

        # 3. 融合 LL 做二级 DWT
        ll_size = ll.shape[-2:]
        second_bands = self.dwt(ll)

        # 4. 联合增强二级四子带
        packed = torch.cat(second_bands, dim=1)
        enhanced = (
            packed + self.second(packed)
        ).chunk(4, dim=1)

        # 5. 二级 IWT：恢复一级 LL 的空间尺寸
        # 内部已有子带残差，不再额外加一次原 LL
        ll = self.iwt(enhanced, output_size=ll_size)

        # 6. 目标 slice 的一级高频子带分别做多核空间卷积
        highs = [
            block(sb)
            for block, sb in zip(
                self.high_processors,
                bands[1][1:]
            )
        ]

        # 7. 增强 LL + 目标 slice 的增强 LH、HL、HH
        reconstructed = self.iwt(
            (ll, *highs),
            output_size=target_size
        )

        return self.fuse(reconstructed)


class WaveletUpsample(nn.Module):
    """Learn four synthesis coefficient maps and upsample by matched IWT.

    Decoder has no saved encoder sub-bands here: coefficients are learned.
    """
    def __init__(self, in_channels, out_channels, wavelet="sym2"):
        super().__init__()
        self.coefficients = nn.Conv2d(in_channels, 4 * out_channels, 1)
        self.iwt = IWT2D(wavelet)

    def forward(self, x, output_size):
        return self.iwt(self.coefficients(x).chunk(4, dim=1), output_size)


class ChannelSE(nn.Module):
    """为每个通道生成一个权重：[B,C,1,1]。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 用绝对值生成权重，仍对原始有符号分量加权
        weight = self.attention(x.abs())
        return x * weight


class SpectralSCSE(nn.Module):
    """空间 FFT -> Re/Im 通道注意力与残差 -> 逆 FFT。"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        # Re、Im 使用独立的通道注意力参数
        self.se_real = ChannelSE(channels, reduction)
        self.se_imag = ChannelSE(channels, reduction)

    def forward(self, x):
        original_dtype = x.dtype

        # FFT 使用 float32，兼容混合精度训练
        with torch.autocast(
            device_type=x.device.type, enabled=False
        ):
            spectrum = torch.fft.fft2(
                x.float(),
                dim=(-2, -1),
                norm="ortho",
            )
            real = spectrum.real
            imag = spectrum.imag

        # 与常规模型参数及 AMP 的计算方式保持一致
        real = real.to(original_dtype)
        imag = imag.to(original_dtype)

        # 分别进行通道注意力，并在频域加入残差
        real_out = real + self.se_real(real)
        imag_out = imag + self.se_imag(imag)

        with torch.autocast(
            device_type=x.device.type, enabled=False
        ):
            enhanced_spectrum = torch.complex(
                real_out.float(),
                imag_out.float(),
            )
            y = torch.fft.ifft2(
                enhanced_spectrum,
                dim=(-2, -1),
                norm="ortho",
            ).real

        return y.to(original_dtype)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, wavelet="sym2"):
        super().__init__()
        self.up = WaveletUpsample(in_channels, out_channels, wavelet)
        self.conv = nn.Sequential(conv_norm_act(out_channels + skip_channels, out_channels),
                                  conv_norm_act(out_channels, out_channels))

    def forward(self, x, skip):
        x = self.up(x, skip.shape[-2:])
        return self.conv(torch.cat((x, skip), dim=1))


class Wave(nn.Module):
    """Only center slice is predicted. inChannel is channels PER slice (must be 1).

    baseChannel controls fusion widths, NOT the ResNet-50 backbone width.
    Default output classes: background, RV, myocardium, LV (check loader mapping).
    No cross-slice loss is included; supervise returned logits with center GT.
    """
    def __init__(self, inChannel=1, outChannel=4, baseChannel=24,
                 wavelet="sym2", kernels: Sequence[int]=(3, 5, 7),
                 dilations: Sequence[int]=(2, 2, 2)):
        super().__init__()
        if inChannel != 1:
            raise ValueError("ACDC single-modality input requires inChannel=1 per slice")
        if baseChannel < 8 or outChannel < 2:
            raise ValueError("baseChannel >= 8 and outChannel >= 2 are required")
        self.encoder_slice = ResNet50Encoder(1)
        self.encoder_diff = MobileNetV2Encoder(in_channels=2)
        widths = tuple(baseChannel * k for k in (1, 2, 4, 8, 16))
        channels = ResNet50Encoder.channels
        diff_channels = MobileNetV2Encoder.channels
        # First four levels are skip connections; deepest level is the bottleneck.
        self.wavelet_skips = nn.ModuleList([
            WaveletSkip(c, w, wavelet, kernels, dilations)
            for c, w in zip(channels[:4], widths[:4])])
        self.fourier_skips = nn.ModuleList([SpectralSCSE(c) for c in diff_channels[:4]])
        self.diff_projections = nn.ModuleList([
            conv_norm_act(c, w, 1) for c, w in zip(diff_channels[:4], widths[:4])])
        self.skip_fusions = nn.ModuleList([conv_norm_act(2 * w, w, 1) for w in widths[:4]])
        self.bottleneck = conv_norm_act(3 * channels[-1] + diff_channels[-1], widths[-1], 1)
        self.decoder = nn.ModuleList([
            DecoderBlock(widths[i + 1], widths[i], widths[i], wavelet) for i in (3, 2, 1, 0)])
        self.final_up = WaveletUpsample(widths[0], widths[0], wavelet)
        self.full_resolution = conv_norm_act(widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], outChannel, 1)

    def forward(self, x):
        if x.ndim == 5 and x.shape[1:3] == (1, 3):
            x = x[:, 0]
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Input must be [B,3,H,W] ordered [previous,center,next], or [B,1,3,H,W]")
        if not x.is_floating_point():
            raise TypeError("Input must be floating-point normalized MRI intensities")
        size = x.shape[-2:]
        prev, center, nxt = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        diff = torch.cat((center - prev, nxt - center), dim=1)
        B, S, H, W = x.shape
        slice_input = x.reshape(B * S, 1, H, W)
        encoded = self.encoder_slice(slice_input)
        encoded = [
            f.reshape(B, S, f.shape[1], f.shape[2], f.shape[3])
            for f in encoded
        ]
        prev_features = tuple(f[:, 0] for f in encoded)
        center_features = tuple(f[:, 1] for f in encoded)
        next_features = tuple(f[:, 2] for f in encoded)
        diff_features = self.encoder_diff(diff)
        features = (
            prev_features,
            center_features,
            next_features,
            diff_features,
        )
        skips = []
        for i in range(4):
            wave = self.wavelet_skips[i]([features[j][i] for j in range(3)])
            freq = self.diff_projections[i](self.fourier_skips[i](features[3][i]))
            skips.append(self.skip_fusions[i](torch.cat((wave, freq), dim=1)))

        y = self.bottleneck(
            torch.cat([feature[-1] for feature in features], dim=1)
        )
        for decoder, skip in zip(self.decoder, reversed(skips)):
            y = decoder(y, skip)
        y = self.final_up(y, size)
        return self.head(self.full_resolution(y))

    def freeze_encoder_batchnorm(self):
        """在 model.train() 之后调用。"""
        for encoder in (self.encoder_slice, self.encoder_diff):
            for module in encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Wave(inChannel=1, outChannel=4, baseChannel=16).eval()
    param_network(model)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    print("Center logits:", tuple(output.shape))
