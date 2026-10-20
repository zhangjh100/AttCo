"""ACDC native 2D slices and true 3D phase-volume inputs."""
import numpy as np
import torch
from scipy.ndimage import zoom, rotate
from dataset import ACDCDataset
from models.baselines import MODEL_REGISTRY

class ACDCSliceDataset(ACDCDataset):
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        # The same center slice, never neighboring slices disguised as RGB.
        sample['input']=sample['input'][1:2].contiguous()
        return sample

class ACDCVolumeDataset(ACDCDataset):
    def __init__(self,*args,volume_size=(32,128,128),augment=False,**kwargs):
        super().__init__(*args,transforms=None,output_size=None,**kwargs)
        self.volume_ids=list(self.volumes)
        self.volume_size=tuple(volume_size); self.augment=augment
    def __len__(self): return len(self.volume_ids)
    def __getitem__(self,index):
        key=self.volume_ids[index]; record=self.volumes[key]
        x,y=self._load_volume(key)
        factors=tuple(n/o for n,o in zip(self.volume_size,x.shape))
        x=zoom(x,factors,order=1,prefilter=False)
        if y is not None: y=zoom(y,factors,order=0,prefilter=False)
        if self.augment:
            for axis in (1,2):
                if np.random.random()<0.5:
                    x=np.flip(x,axis).copy()
                    if y is not None: y=np.flip(y,axis).copy()
            if np.random.random()<0.5:
                angle=np.random.uniform(-15,15)
                x=rotate(x,angle,axes=(1,2),reshape=False,order=1,prefilter=False)
                if y is not None: y=rotate(y,angle,axes=(1,2),reshape=False,order=0,prefilter=False)
        sample={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in record.items()}
        sample.update(input=torch.from_numpy(np.ascontiguousarray(x[None])).float(),volume_id=key,id=key)
        if y is not None: sample['target']=torch.from_numpy(np.ascontiguousarray(y)).long()
        return sample

def build_dataset(name,path,patients,mode='train',output_size=(224,224),volume_size=(32,128,128),transforms=None):
    dim=MODEL_REGISTRY[name]
    if dim=='3d':
        return ACDCVolumeDataset(path,patients,mode=mode,volume_size=volume_size,augment=mode=='train')
    cls=ACDCSliceDataset if dim=='2d' else ACDCDataset
    return cls(path,patients,mode=mode,output_size=output_size,transforms=transforms)
