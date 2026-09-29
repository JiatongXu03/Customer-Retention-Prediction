import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from advanced_models import temporal_splits

class TemporalBoundaryTests(unittest.TestCase):
    def test_future_labels_rejected(self):
        frame=pd.DataFrame({'cutoff':['2010-06-01','2010-09-01'],
            'label_end_exclusive':['2010-09-02','2010-11-30']})
        with self.assertRaises(AssertionError):
            temporal_splits(frame)

    def test_boundary_equal_is_allowed_and_future_rows_excluded(self):
        frame=pd.DataFrame({'cutoff':['2010-06-01','2010-09-01','2010-12-01'],
            'label_end_exclusive':['2010-09-01','2010-12-01','2011-03-01']})
        splits=temporal_splits(frame)
        self.assertEqual([(a.tolist(),b.tolist()) for a,b in splits],[([0],[1]),([0,1],[2])])

    def test_one_origin_cannot_tune(self):
        frame=pd.DataFrame({'cutoff':['2010-06-01'], 'label_end_exclusive':['2010-08-30']})
        self.assertEqual(temporal_splits(frame),[])

if __name__=='__main__': unittest.main()
