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
            nn.Upsample(scale_factor=scale_factor),
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
        self.downsample = downsample
        if downsample is not None:
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


class SEAttention(nn.Module):
    def __init__(self, hidden_size, input_size, num_memory_units=64):
        super(SEAttention, self).__init__()
        self.size = input_size
        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Self Attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        # External Attention
        self.memory_key = nn.Linear(hidden_size // self.num_attention_heads, num_memory_units)
        self.memory_value = nn.Linear(num_memory_units, hidden_size // self.num_attention_heads)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # Output combination
        self.conv = nn.Conv3d(hidden_size * 2, hidden_size, kernel_size=1, stride=1, padding=0)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # print(x.shape, new_x_shape, x.size()[:-1])
        x = x.view(*new_x_shape)
        # print(x.shape, x.permute(0, 2, 1, 3).shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        hidden_states = hidden_states.flatten(2)  # (batchsize, hidden_size, d*h*w)
        hidden_states = hidden_states.transpose(-1, -2)  # (batchsize, n_patches, hidden_size)

        mixed_query_layer = self.query(hidden_states)
        # Self Attention
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)

        shape = attention_output.shape
        attention_output = attention_output.permute(0, 2, 1)
        self_attention_output = attention_output.view((shape[0], shape[2]) + self.size)

        # External Attention
        attn = self.memory_key(query_layer)
        attn = attn.softmax(dim=2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        out = self.memory_value(attn)
        out = rearrange(out, 'b h n c -> b n (h c)')
        out = self.proj(out)

        shape = out.shape
        out = out.permute(0, 2, 1)
        external_attention_output = out.view((shape[0], shape[2]) + self.size)

        #
        attention_output = self.conv(torch.cat((self_attention_output, external_attention_output), dim=1))
        return attention_output


class intraInteraction(nn.Module):
    def __init__(self, hidden, input_size=(4, 4, 4)):
        super(intraInteraction, self).__init__()
        self.attention1 = SEAttention(hidden, input_size=input_size)
        self.attention2 = SEAttention(hidden, input_size=input_size)

    def forward(self, x1, x2):
        h1 = self.attention1(x1)
        h2 = self.attention2(x2)
        return h1, h2

class LayerNorm3d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


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
        # 8种组合：L=低通, H=高通
        combinations = [
            (lo, lo, lo),  # LLL → 低频 (轮廓)
            (lo, lo, hi),  # LLH → 高频
            (lo, hi, lo),  # LHL → 高频
            (lo, hi, hi),  # LHH → 高频
            (hi, lo, lo),  # HLL → 高频
            (hi, lo, hi),  # HLH → 高频
            (hi, hi, lo),  # HHL → 高频
            (hi, hi, hi),  # HHH → 高频
        ]

        # 逐个生成 3D 卷积核
        for a, b, c in combinations:
            kernel_3d = torch.einsum('i,j,k->ijk', a, b, c)  # 外积生成3D核
            kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # [1,1,Kd,Kh,Kw]
            kernels.append(kernel_3d)

        self.kernel = torch.cat(kernels, dim=0)
        self.kernel = self.kernel.repeat(channels, 1, 1, 1, 1)  # [C*8, 1, Kd, Kh, Kw]
        self.kernel = nn.Parameter(self.kernel, requires_grad=initialize)

    def forward(self, x):
        """
        输入 x: [B, C, D, H, W]
        输出: [B, C*8, D//2, H//2, W//2] → 8个子带拼接在通道维度
        """
        padding = (self.kernel.shape[2] - 1) // 2

        # 3D 分组小波卷积 (groups=C → 每个通道独立8个子带小波变换)
        out = F.conv3d(
            x,
            self.kernel,
            stride=2,
            padding=padding,
            groups=self.channels
        )

        # 输出：通道数 ×8，空间尺寸 ÷2 (标准3D DWT结果)
        return out


class WaveletBlock3D(nn.Module):
    def __init__(self, c, DW_Expand=8, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.wavelet_block1 = LWN3D(c, wavelet='sym2', initialize=True)

        self.conv3 = nn.Conv3d(
            in_channels=dw_channel, out_channels=c,
            kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
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

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, input_size=(4, 4, 4)):
        super(CrossAttention, self).__init__()
        self.size = input_size
        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value1 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)

        self.out1 = nn.Linear(hidden_size, hidden_size)
        self.out2 = nn.Linear(hidden_size, hidden_size)

        # self.conv1 = single_conv(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
        # self.conv2 = single_conv(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.softmax1 = nn.Softmax(dim=-2)
        self.softmax2 = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # print(x.shape, new_x_shape, x.size()[:-1])
        x = x.view(*new_x_shape)
        # print(x.shape, x.permute(0, 2, 1, 3).shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        hidden_states_1 = x1.flatten(2)  # (batchsize, hidden_size, patch size h * patch size w)
        hidden_states_1 = hidden_states_1.transpose(-1, -2)  # (batchsize, n_patches, hidden_size)

        hidden_states_2 = x2.flatten(2)  # (batchsize, hidden_size, patch size h * patch size w)
        hidden_states_2 = hidden_states_2.transpose(-1, -2)  # (batchsize, n_patches, hidden_size)

        mixed_query_layer = self.query(hidden_states_1)
        mixed_key_layer = self.key(hidden_states_2)
        mixed_value_layer_1 = self.value1(hidden_states_1)
        mixed_value_layer_2 = self.value2(hidden_states_2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer_1 = self.transpose_for_scores(mixed_value_layer_1)
        value_layer_2 = self.transpose_for_scores(mixed_value_layer_2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs_1 = self.softmax1(attention_scores)
        attention_probs_2 = self.softmax2(attention_scores).transpose(-1, -2)
        ###
        context_layer1 = torch.matmul(attention_probs_1, value_layer_1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape)
        attention_output_1 = self.out1(context_layer1)

        shape = attention_output_1.shape
        attention_output_1 = attention_output_1.permute(0, 2, 1)
        attention_output_1 = attention_output_1.view((shape[0], shape[2]) + self.size)
        ###
        context_layer2 = torch.matmul(attention_probs_2, value_layer_2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape)
        attention_output_2 = self.out2(context_layer2)

        shape = attention_output_2.shape
        attention_output_2 = attention_output_2.permute(0, 2, 1)
        attention_output_2 = attention_output_2.view((shape[0], shape[2]) + self.size)
        ####
        attention_output_1 = attention_output_1 + x1
        attention_output_2 = attention_output_2 + x2
        # print(attention_output_1.shape, hidden_states_1.shape)
        return attention_output_1, attention_output_2


class WaveCo(nn.Module):
    """
    """

    def __init__(self, inChannel=2, outChannel=4, baseChannel=24):
        super(WaveCo, self).__init__()
        self.encoder1 = Encoder(inChannel=inChannel, baseChannel=baseChannel)
        self.encoder2 = Encoder(inChannel=inChannel, baseChannel=baseChannel)

        # self.unimodalInteraction = intraInteraction(16 * baseChannel, input_size=(8, 8, 8))
        # self.crossInteraction = CrossAttention(16 * baseChannel, input_size=(8, 8, 8))
        self.fusion = WaveletBlock3D(self, DW_Expand=8, FFN_Expand=2, drop_out_rate=0.)

        # decoder
        self.Up5 = up_conv(ch_in=baseChannel * 16 * 4, ch_out=baseChannel * 8)
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
        x_in1, x_in2 = torch.cat((t1, t1ce), dim=1), torch.cat((flair, t2), dim=1)
        # x_in1, x_in2 = torch.cat((t1, flair), dim=1), torch.cat((t1ce, t2), dim=1)
        # encoding path
        h_out_1 = self.encoder1(x_in1)
        h_out_2 = self.encoder2(x_in2)

        x1 = torch.cat((h_out_1[0], h_out_2[0]), dim=1)
        x2 = torch.cat((h_out_1[1], h_out_2[1]), dim=1)
        x3 = torch.cat((h_out_1[2], h_out_2[2]), dim=1)
        x4 = torch.cat((h_out_1[3], h_out_2[3]), dim=1)

        x1 = self.fusion(x1)
        x2 = self.fusion(x2)
        x3 = self.fusion(x3)
        x4 = self.fusion(x4)
        # x5 = torch.cat((h_out_1[4], h_out_2[4]), dim=1)
        # print(h_out_1[4].shape)
        # h11, h22 = self.unimodalInteraction(h_out_1[4], h_out_2[4])
        # h12, h21 = self.crossInteraction(h_out_1[4], h_out_2[4])
        x5 = torch.cat((h_out_1[3], h_out_2[3]), dim=1)
        # x5 = torch.cat((h11, h12, h21, h22), dim=1)

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
    model = WaveCo(inChannel=2, outChannel=4)
    x = torch.ones((2, 4, 128, 128, 128))
    output = model(x)
    print(output.shape)
