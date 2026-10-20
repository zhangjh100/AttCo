"""Native-dimensional ACDC baselines; official sources are under third_party/."""
import copy
import importlib
import torch
from torch import nn
from torch.nn import functional as F

MODEL_REGISTRY = {name: dim for name, dim in (
    ('Wave', '2.5d'), ('WaveDe', '2.5d'), ('Mamba-UNet', '2d'),
    ('EMCAD', '2d'), ('CASCADE', '2d'), ('EffiDec3D', '3d'),
    ('EfficientMedNeXt', '3d'), ('MT-UNet', '2d'), ('TransUNet', '2d'), ('U-Net', '2d'))}

def validate_model_config(name, base_channels=16, image_size=(224,224), volume_size=(32,128,128)):
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model {name}; choose from {tuple(MODEL_REGISTRY)}')
    if base_channels < 8:
        raise ValueError('base_channels must be at least 8')
    if len(image_size) != 2 or min(image_size)<64 or any(n%32 for n in image_size):
        raise ValueError('image_size must be multiples of 32, at least 64')
    if name in ('MT-UNet', 'TransUNet', 'Mamba-UNet') and tuple(image_size)!=(224,224):
        raise ValueError(f'{name}: this official baseline configuration requires 224x224')
    if len(volume_size)!=3 or min(volume_size)<32 or any(n%16 for n in volume_size):
        raise ValueError('volume_size must be (D,H,W), multiples of 16, at least 32')

class UNet(nn.Module):
    """Ronneberger U-Net: 64/128/256/512/1024, transposed upsampling.

    Same-padding adaptation preserves the input grid for shared ACDC metrics.
    """
    def __init__(self, classes=4):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(inplace=True),
                                 nn.Conv2d(b,b,3,padding=1),nn.ReLU(inplace=True))
        widths=[64,128,256,512,1024]
        self.down=nn.ModuleList([block(a,b) for a,b in zip([1]+widths[:-1],widths)])
        self.up=nn.ModuleList([nn.ConvTranspose2d(a,b,2,2) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.decode=nn.ModuleList([block(a,b) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.head=nn.Conv2d(64,classes,1)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.down):
            x=block(x)
            if i<4: skips.append(x); x=F.max_pool2d(x,2)
        for up,block,skip in zip(self.up,self.decode,reversed(skips)):
            x=block(torch.cat([skip,up(x)],1))
        return self.head(x)

class NativeOutput(nn.Module):
    """Retain all training heads, resize logits, preserve official inference fusion."""
    def __init__(self,network,name):
        super().__init__(); self.network=network; self.name=name
    def forward(self,x):
        if self.name in ('EffiDec3D','EfficientMedNeXt'):
            out=self.network(x,mode='train' if self.training else 'test')
        else:
            out=self.network(x)
        outputs=list(out) if isinstance(out,(tuple,list)) else [out]
        mode='trilinear' if x.ndim==5 else 'bilinear'
        outputs=[F.interpolate(y,size=x.shape[2:],mode=mode,align_corners=False)
                 if y.shape[2:]!=x.shape[2:] else y for y in outputs]
        if self.training and len(outputs)>1: return tuple(outputs)
        if self.name in ('EMCAD','CASCADE'): return torch.stack(outputs).sum(0)
        return outputs[0]

def prediction_logits(output, name=None):
    if isinstance(output,(tuple,list)):
        return torch.stack(output).sum(0) if name in ('EMCAD','CASCADE') else output[0]
    return output

def segmentation_loss(criterion, output, target):
    outputs=output if isinstance(output,(tuple,list)) else (output,)
    return sum(criterion(y,target) for y in outputs)/len(outputs)

def build_model(name,base_channels=16,image_size=(224,224),volume_size=(32,128,128)):
    validate_model_config(name,base_channels,image_size,volume_size)
    if name in ('Wave','WaveDe'):
        module=importlib.import_module(f'models.{name}_ACDC')
        return getattr(module,name)(inChannel=1,outChannel=4,baseChannel=base_channels)
    try:
        if name=='U-Net': net=UNet()
        elif name=='EMCAD':
            from third_party.emcad.lib.networks import EMCADNet
            net=EMCADNet(num_classes=4,encoder='pvt_v2_b2',pretrain=False)
        elif name=='CASCADE':
            from third_party.cascade.lib.networks import PVT_CASCADE
            net=PVT_CASCADE(n_class=4,pretrained_path=None)
        elif name=='MT-UNet':
            from third_party.mt_unet.model.MTUNet import MTUNet
            net=MTUNet(out_ch=4)
        elif name=='TransUNet':
            from third_party.transunet.networks.vit_seg_modeling import VisionTransformer, CONFIGS
            config=copy.deepcopy(CONFIGS['R50-ViT-B_16'])
            config.n_classes=4; config.n_skip=3; config.patches.grid=(14,14)
            net=VisionTransformer(config,img_size=224,num_classes=4)
        elif name=='Mamba-UNet':
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            except (ImportError,OSError) as exc:
                raise RuntimeError('Mamba-UNet needs the official mamba-ssm CUDA selective_scan operator; use a compatible Linux/CUDA environment. No substitute architecture is used.') from exc
            from types import SimpleNamespace as NS
            from third_party.mamba_unet.code.networks.vision_mamba import MambaUnet
            cfg=NS(MODEL=NS(VSSM=NS(PATCH_SIZE=4,IN_CHANS=3,EMBED_DIM=96,DEPTHS=[2,2,2,2]),
                SWIN=NS(MLP_RATIO=4.,PATCH_NORM=True),DROP_RATE=0.,DROP_PATH_RATE=0.2),TRAIN=NS(USE_CHECKPOINT=False))
            net=MambaUnet(cfg,img_size=224,num_classes=4)
        elif name=='EffiDec3D':
            from third_party.effidec3d.networks.MedNeXt.mednextv1.create_mednextv1_effidec3d import create_mednextv1_effidec3d
            net=create_mednextv1_effidec3d(1,4,'M',kernel_size=3,n_channels=32,deep_supervision=True,mode='train')
        elif name=='EfficientMedNeXt':
            from third_party.efficientmednext.networks.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext
            net=create_efficient_mednext(1,4,'T',n_channels=32,deep_supervision=True,mode='train')
    except ImportError as exc:
        raise RuntimeError(f'{name}: missing dependency {exc.name}. Install requirements-baselines.txt') from exc
    return NativeOutput(net,name)
