import unittest
import numpy as np
import pandas as pd
from metrics import Hausdorff_distance
from test import patient_results_table

class HausdorffTest(unittest.TestCase):
    def test_physical_3d_distance(self):
        a=np.zeros((4,5,6),bool);b=a.copy()
        a[1,1,1]=True;b[2,3,3]=True
        self.assertAlmostEqual(Hausdorff_distance(a,b,(5,2,1)),np.sqrt(45))
        self.assertEqual(Hausdorff_distance(a,a,(5,2,1)),0)
        self.assertAlmostEqual(Hausdorff_distance(a,b),3)
    def test_empty(self):
        a=np.zeros((3,3,3),bool);b=a.copy();b[1,1,1]=True
        self.assertEqual(Hausdorff_distance(a,a),0)
        self.assertTrue(np.isinf(Hausdorff_distance(a,b)))
    def test_phase_summary(self):
        rows=[dict(model='Wave',fold=0,checkpoint='x',patient_id='p',phase=p,
                   Dice_RV=1.,Dice_MYO=1.,Dice_LV=1.,Dice_mean=1.,
                   HD_RV_mm=h,HD_MYO_mm=h,HD_LV_mm=h,HD_mean_mm=h)
              for p,h in [('ED',2.),('ES',4.)]]
        result=patient_results_table(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.HD_RV_mm,3.)
        self.assertEqual(result.HD_RV_mm_ED,2.)
    def test_lge_summary(self):
        rows=[dict(model='Wave',fold=0,checkpoint='x',patient_id='p',phase='LGE',volume_id='p',HD_NoReflow_mm=float('inf'))]
        self.assertTrue(np.isinf(patient_results_table(pd.DataFrame(rows)).iloc[0].HD_NoReflow_mm))

if __name__=='__main__': unittest.main()
