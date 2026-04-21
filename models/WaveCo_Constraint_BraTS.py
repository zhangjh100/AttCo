import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from einops import rearrange
import pywt


def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('GroupNorm') != -1:
            # init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    # print('Initialize network with %s' % init_type)
    net.apply(init_func)


def param_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    # print(model)
    print("The number of parameters: {}".format(num_params))


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = None
        if downsample or inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, inChannel=2, baseChannel=24):
        super(Encoder, self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = nn.Sequential(
            BasicBlock(inChannel, baseChannel, stride=1, downsample=True),
            BasicBlock(baseChannel, baseChannel, stride=1, downsample=None)
        )
        self.Conv2 = nn.Sequential(
            BasicBlock(baseChannel, baseChannel * 2, stride=1, downsample=True),
            BasicBlock(baseChannel * 2, baseChannel * 2, stride=1, downsample=None)
        )
        self.Conv3 = nn.Sequential(
            BasicBlock(baseChannel * 2, baseChannel * 4, stride=1, downsample=True),
            BasicBlock(baseChannel * 4, baseChannel * 4, stride=1, downsample=None)
        )
        self.Conv4 = nn.Sequential(
            BasicBlock(baseChannel * 4, baseChannel * 8, stride=1, downsample=True),
            BasicBlock(baseChannel * 8, baseChannel * 8, stride=1, downsample=None)
        )
        self.Conv5 = nn.Sequential(
            BasicBlock(baseChannel * 8, baseChannel * 16, stride=1, downsample=True),
            BasicBlock(baseChannel * 16, baseChannel * 16, stride=1, downsample=None)
        )

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        return x1, x2, x3, x4, x5


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, D, H, W = x.size()

        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()

        ctx.save_for_backward(y, var, weight)

        y = weight.view(1, C, 1, 1, 1) * y + bias.view(1, C, 1, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, D, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables

        g = grad_output * weight.view(1, C, 1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)

        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return gx, (grad_output * y).sum(dim=4).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=4).sum(dim=3).sum(
            dim=2).sum(dim=0), None

class LayerNorm3d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm3d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LWN3D(nn.Module):
    def __init__(self, channels, wavelet='sym2', initialize=True):
        super().__init__()
        self.channels = channels
        self.wavelet = wavelet

        wavelet_filter = pywt.Wavelet(wavelet)
        lo = wavelet_filter.dec_lo  # 低通滤波器
        hi = wavelet_filter.dec_hi  # 高通滤波器

        lo = torch.tensor(lo, dtype=torch.float32)
        hi = torch.tensor(hi, dtype=torch.float32)

        # 3D 小波 = 1D 滤波器外积组合，生成 3D 小波 8 种核 (标准3D DWT 8子带)
        kernels = []
        combinations = [
            (lo, lo, lo),  # LLL → 低频 (轮廓)
            (hi, lo, lo),  # HLL → 高频
            (hi, lo, hi),  # HLH → 高频
            (hi, hi, lo),  # HHL → 高频
        ]

        # 逐个生成 3D 卷积核
        for a, b, c in combinations:
            kernel_3d = torch.einsum('i,j,k->ijk', a, b, c)  # 外积生成3D核
            kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # [1,1,Kd,Kh,Kw]
            kernels.append(kernel_3d)

        self.kernel = torch.cat(kernels, dim=0)
        self.kernel = self.kernel.repeat(channels, 1, 1, 1, 1)  # [C*4., 1, Kd, Kh, Kw]
        self.kernel = nn.Parameter(self.kernel, requires_grad=initialize)

    def forward(self, x):
        """
        输入 x: [B, C, D, H, W]
        输出: [B, C*8, D//2, H//2, W//2] → 8个子带拼接在通道维度
        """
        kernel_size = self.kernel.shape[2:]
        padding = tuple((k - 1) // 2 for k in kernel_size)

        # 3D 分组小波卷积 (groups=C → 每个通道独立4个子带小波变换)
        out = F.conv3d(
            x,
            self.kernel,
            stride=2,
            padding=padding,
            groups=self.channels
        )

        # 输出：通道数 ×4，空间尺寸 ÷2 (标准3D DWT结果)
        return out

    def get_filters(self):
        """
        从可学习小波核中提取：低通滤波器(LLL) + 高通滤波器
        直接供小波损失函数使用
        """
        # 分组：每个通道对应4个滤波器 (C*4, 1, K, K, K)
        single_channel_filters = self.kernel[:4]  # 取第一个通道的滤波器代表所有通道
        lo = single_channel_filters[0]  # LLL 低通
        hi = single_channel_filters[1]  # HLL 高通（任选一个高通即可）
        return lo, hi

class WaveletConstraintLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def perfect_reconstruction_loss(self, lo, hi):
        """
        完美重构损失 (Perfect Reconstruction Loss)
        论文核心约束：低通+高通滤波器满足双正交小波重构条件
        """
        # 计算滤波器内积和，目标值 = 2
        sum_lo = torch.sum(lo)
        sum_hi = torch.sum(hi)
        loss = torch.pow(sum_lo + sum_hi - 2.0, 2)
        return loss

    def alias_cancellation_loss(self, lo, hi):
        """
        混叠消除损失 (Alias Cancellation Loss)
        论文核心约束：消除小波变换的频率混叠
        """
        # 生成交替符号掩码 (-1)^k
        k = torch.arange(0, lo.numel(), device=self.device)
        mask = torch.pow(-1.0, k).view(lo.shape)

        # 计算交替加权和，目标值 = 0
        alt_lo = torch.sum(lo * mask)
        alt_hi = torch.sum(hi * mask)
        loss = torch.pow(alt_lo + alt_hi, 2)
        return loss

    def forward(self, lo_filter, hi_filter):
        """
        输入：
            lo_filter: 可学习3D小波**低通滤波器**权重 (LLL)
            hi_filter: 可学习3D小波**高通滤波器**权重
        输出：
            total_wavelet_loss: 论文小波总约束损失
        """
        lo = lo_filter.to(self.device)
        hi = hi_filter.to(self.device)

        # 计算两个子损失
        loss_perfect = self.perfect_reconstruction_loss(lo, hi)
        loss_alias = self.alias_cancellation_loss(lo, hi)

        # 论文总小波损失 = 两者相加
        total_loss = loss_perfect + loss_alias
        return total_loss

class WaveletBlock3D(nn.Module):
    def __init__(self, c, DW_Expand=8, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * 4
        self.wavelet_block1 = LWN3D(c, wavelet='sym2', initialize=True)
        self.norm_after_wavelet = LayerNorm3d(dw_channel)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv3 = nn.Conv3d(
            in_channels=dw_channel, out_channels=c,
            kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        self.sca = nn.Sequential(
            nn.AdaptiveMaxPool3d(1),
            nn.Conv3d(
                in_channels=dw_channel, out_channels=dw_channel,
                kernel_size=1, padding=0, stride=1, groups=1, bias=True
            ),
            nn.Sigmoid()
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(in_channels=c, out_channels=ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv3d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.wavelet_block1(x)  # 输出 8倍通道
        x = self.norm_after_wavelet(x)

        x = self.upsample(x)

        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)

        # SimpleGate 门控
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2

        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class WaveCo_Constraint(nn.Module):
    """
    """

    def __init__(self, inChannel=2, outChannel=4, baseChannel=24):
        super(WaveCo_Constraint, self).__init__()
        self.encoder1 = Encoder(inChannel=inChannel, baseChannel=baseChannel)
        self.encoder2 = Encoder(inChannel=inChannel, baseChannel=baseChannel)

        self.fusion1 = WaveletBlock3D(c=baseChannel * 2, DW_Expand=8, FFN_Expand=2, drop_out_rate=0.)
        self.fusion2 = WaveletBlock3D(c=baseChannel * 4, DW_Expand=8, FFN_Expand=2, drop_out_rate=0.)
        self.fusion3 = WaveletBlock3D(c=baseChannel * 8, DW_Expand=8, FFN_Expand=2, drop_out_rate=0.)
        self.fusion4 = WaveletBlock3D(c=baseChannel * 16, DW_Expand=8, FFN_Expand=2, drop_out_rate=0.)

        # decoder
        self.Up5 = up_conv(ch_in=baseChannel * 8 * 4, ch_out=baseChannel * 8)
        self.Up_conv5 = BasicBlock(baseChannel * 24, baseChannel * 8, stride=1, downsample=True)

        self.Up4 = up_conv(ch_in=baseChannel * 8, ch_out=baseChannel * 4)
        self.Up_conv4 = BasicBlock(baseChannel * 12, baseChannel * 4, stride=1, downsample=True)

        self.Up3 = up_conv(ch_in=baseChannel * 4, ch_out=baseChannel * 2)
        self.Up_conv3 = BasicBlock(baseChannel * 6, baseChannel * 2, stride=1, downsample=True)

        self.Up2 = up_conv(ch_in=baseChannel * 2, ch_out=baseChannel)
        self.Up_conv2 = BasicBlock(baseChannel * 3, baseChannel, stride=1, downsample=True)

        self.Conv_1x1 = nn.Conv3d(baseChannel, outChannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        flair, t1, t1ce, t2 = x[:, :1, :, :, :], x[:, 1:2, :, :, :], x[:, 2:3, :, :, :], x[:, 3:, :, :, :]
        x_in1, x_in2 = torch.cat((t1, t2), dim=1), torch.cat((flair, t1ce), dim=1)
        # x_in1, x_in2 = torch.cat((t1, flair), dim=1), torch.cat((t1ce, t2), dim=1)
        # encoding path
        h_out_1 = self.encoder1(x_in1)
        h_out_2 = self.encoder2(x_in2)

        x1 = torch.cat((h_out_1[0], h_out_2[0]), dim=1)
        x2 = torch.cat((h_out_1[1], h_out_2[1]), dim=1)
        x3 = torch.cat((h_out_1[2], h_out_2[2]), dim=1)
        x4 = torch.cat((h_out_1[3], h_out_2[3]), dim=1)

        x1 = self.fusion1(x1)
        x2 = self.fusion2(x2)
        x3 = self.fusion3(x3)
        x4 = self.fusion4(x4)
        x5 = torch.cat((h_out_1[4], h_out_2[4]), dim=1)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)

        return out


###
if __name__ == "__main__":
    model = WaveCo_Constraint(inChannel=2, outChannel=4)
    x = torch.ones((2, 4, 128, 128, 128))
    output = model(x)
    print(output.shape)
