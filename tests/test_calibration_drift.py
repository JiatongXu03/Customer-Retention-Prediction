import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from calibration_drift import calibrate, psi

class CalibrationTests(unittest.TestCase):
    cfg = {'sigmoid_C': 1000000.0}

    def test_no_temporal_history_uses_identity(self):
        for method in ['raw','sigmoid','isotonic']:
            np.testing.assert_array_equal(calibrate(method,[],[],[0,.4,1],self.cfg),[0,.4,1])

    def test_extreme_predictions_and_out_of_range_scores(self):
        for method in ['sigmoid','isotonic']:
            result = calibrate(method,[.2,.3,.5,.7],[0,1,0,1],[0,.4,1],self.cfg)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(((result>=0)&(result<=1)).all())

    def test_single_class_calibration_rejected(self):
        with self.assertRaises(ValueError):
            calibrate('sigmoid',[.1,.2],[0,0],[.3],self.cfg)

    def test_psi_detects_mass_outside_constant_reference(self):
        self.assertEqual(psi(np.zeros(30),np.zeros(30)),0)
        self.assertGreater(psi(np.zeros(30),np.ones(30)),0)

if __name__ == '__main__':
    unittest.main()
