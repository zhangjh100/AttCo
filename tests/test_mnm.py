import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import SimpleITK as sitk
from comparison_data import build_dataset
from mnm_data import read_phase_image


class MnMTest(unittest.TestCase):
    def test_native_cine_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root/'Training'/'Labeled'
            metadata = []
            for patient in ('CASE01','CASE02'):
                folder = training/patient
                folder.mkdir(parents=True)
                x = np.zeros((3,5,64,64), np.float32)
                y = np.zeros(x.shape, np.uint8)
                for t in range(3):
                    x[t] = np.arange(5)[:,None,None] + t*10 + 1
                for t in (0,2):
                    y[t,:,4:16,4:16] = 1
                    y[t,:,22:34,22:34] = 2
                    y[t,:,40:52,40:52] = 3
                for suffix, array in (('',x),('_gt',y)):
                    image = sitk.GetImageFromArray(array, isVector=False)
                    image.SetSpacing((1.2,1.3,7.,0.04))
                    sitk.WriteImage(image,str(folder/f'{patient}_sa{suffix}.nii.gz'))
                metadata.append(dict(patient_id=patient,ED=0,ES=2))
            meta = root/'metadata.csv'
            pd.DataFrame(metadata).to_csv(meta,index=False)
            splits = root/'folds.csv'
            pd.DataFrame(dict(ID=['CASE01','CASE02'],Fold_0=[1,0])).to_csv(splits,index=False)
            options = dict(dataname='MnM',metadata_path=meta,mode='val')
            data = build_dataset('U-Net',root,['CASE01'],output_size=(64,64),**options)
            self.assertEqual(len(data),10)
            self.assertEqual(data[0]['input'].shape,(1,64,64))
            self.assertEqual(data[0]['target'][8,8].item(),3)
            self.assertEqual(data[0]['target'][45,45].item(),1)
            self.assertEqual(data[5]['phase'],'ES')
            ed = sitk.GetArrayFromImage(read_phase_image(data.volumes['CASE01_ED']))
            es = sitk.GetArrayFromImage(read_phase_image(data.volumes['CASE01_ES']))
            np.testing.assert_allclose(es-ed,20)
            adjacent = build_dataset('Wave',training,['CASE01'],output_size=(64,64),**options)
            self.assertEqual(adjacent[0]['input'].shape,(3,64,64))
            volume = build_dataset('EfficientMedNeXt',root,['CASE01'],volume_size=(32,32,32),**options)
            self.assertEqual(len(volume),2)
            self.assertEqual(volume[0]['input'].shape,(1,32,32,32))
            with self.assertRaises(ValueError):
                build_dataset('U-Net',root,['MISSING'],**options)
            env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            for name in ('U-Net','EfficientMedNeXt'):
                common = ['--dataname','MnM','--metadata_path',str(meta),'--csv_path',str(splits),
                          '--path_image',str(root),'--modelname',name,'--image_size','64','64',
                          '--volume_size','32','32','32']
                def run(arguments):
                    result = subprocess.run([sys.executable,*arguments],env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                run(['main.py',*common,'--train_mode','cpu','--epochs','1','--folds','0',
                     '--train_batch_size','2','--checkpoint_dir',str(root/'ckpt')])
                checkpoint = root/'ckpt'/'MnM'/name/'best_fold_0.pt'
                out = root/name
                run(['test.py',*common,'--cpu','--fold','0','--pretrained',str(checkpoint),
                     '--save_pred','--output_path',str(out)])
                table = pd.read_csv(out/'patient_results_fold_0.csv')
                self.assertEqual(table.patient_id.tolist(),['CASE02'])
                self.assertTrue(np.isfinite(table.Dice_mean).all())
                prediction = sitk.ReadImage(str(out/'CASE02_ED_pred.nii.gz'))
                self.assertEqual(prediction.GetSize(),(64,64,5))
                np.testing.assert_allclose(prediction.GetSpacing(),(1.2,1.3,7.))


if __name__ == '__main__':
    unittest.main()
