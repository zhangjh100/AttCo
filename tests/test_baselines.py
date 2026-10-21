"""Regression checks: native inputs, phase boundaries, 3D loss and CLI evaluation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from comparison_data import build_dataset
from dataset import resolve_acdc_root
from transforms import ToTensor

class BaselinePipelineTest(unittest.TestCase):
    def test_acdc_root_layouts(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                resolve_acdc_root(root)
            training = Path(root) / 'training'
            training.mkdir()
            self.make_data(training)
            self.assertEqual(resolve_acdc_root(root), str(training.resolve()))
            self.assertEqual(resolve_acdc_root(training), str(training.resolve()))
            data = build_dataset('U-Net', root, ['patient001'], mode='val',
                                 output_size=(64,64), transforms=ToTensor())
            self.assertEqual(data[0]['input'].shape, (1,64,64))

    def make_data(self,root):
        for patient in ('patient001','patient002'):
            folder=Path(root)/patient; folder.mkdir()
            (folder/'Info.cfg').write_text('ED: 1\nES: 2\n')
            for frame in (1,2):
                x=np.stack([np.full((64,64),10*frame+z,np.float32) for z in range(5)])
                y=np.zeros_like(x,dtype=np.uint8)
                y[:,4:16,4:16]=1; y[:,22:34,22:34]=2; y[:,40:52,40:52]=3
                for suffix,array in (('',x),('_gt',y)):
                    image=sitk.GetImageFromArray(array); image.SetSpacing((1.2,1.3,7.))
                    sitk.WriteImage(image,str(folder/f'{patient}_frame{frame:02d}{suffix}.nii.gz'))
    def test_native_inputs_and_no_phase_mixing(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_data(root)
            a=build_dataset('U-Net',root,['patient001'],mode='val',output_size=(64,64),transforms=ToTensor())
            b=build_dataset('WaveDe',root,['patient001'],mode='val',output_size=(64,64),transforms=ToTensor())
            c=build_dataset('EfficientMedNeXt',root,['patient001'],mode='val',volume_size=(32,32,32))
            self.assertEqual(a[0]['input'].shape,(1,64,64))
            self.assertTrue(torch.equal(a[2]['input'][0],b[2]['input'][1]))
            self.assertEqual(len(c),2); self.assertEqual(c[0]['input'].shape,(1,32,32,32))
            self.assertEqual(c[0]['target'].shape,(32,32,32))
            self.assertEqual(c[0]['phase'],'ED'); self.assertEqual(c[1]['phase'],'ES')
            self.assertEqual(set(c[0]['target'].unique().tolist()),{0,1,2,3})
    def test_train_test_original_grid_2d_and_3d(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_data(root)
            csv=Path(root)/'folds.csv'
            pd.DataFrame({'ID':['patient001','patient002'],'Fold_0':[1,0]}).to_csv(csv,index=False)
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            for name in ('U-Net','EfficientMedNeXt','EMCAD','EffiDec3D'):
                with self.subTest(model=name):
                    common=['--path_image',root,'--csv_path',str(csv),'--modelname',name,
                            '--image_size','64','64','--volume_size','32','32','32']
                    subprocess.run([sys.executable,'main.py',*common,'--train_mode','cpu','--epochs','1',
                                    '--train_batch_size','2','--folds','0','--checkpoint_dir',root+'/ckpt'],
                                   env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    out=Path(root)/name
                    checkpoint=Path(root)/'ckpt'/'ACDC'/name/'best_fold_0.pt'
                    subprocess.run([sys.executable,'test.py',*common,'--cpu','--fold','0','--pretrained',str(checkpoint),
                                    '--save_pred','--output_path',str(out)],env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    table=pd.read_csv(out/'patient_results_fold_0.csv')
                    self.assertEqual(table.patient_id.tolist(),['patient002'])
                    self.assertTrue(np.isfinite(table.Dice_mean).all())
                    for phase in (1,2):
                        image=sitk.ReadImage(str(out/f'patient002_frame{phase:02d}_pred.nii.gz'))
                        self.assertEqual(image.GetSize(),(64,64,5))
                        np.testing.assert_allclose(image.GetSpacing(),(1.2,1.3,7.))
if __name__=='__main__': unittest.main()
