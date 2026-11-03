"""Reconcile saved metrics, frozen validation coverage and calibration boundaries."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from baseline_utils import score
from calibration_drift import calibrate


def validate(root):
    dest = root/'outputs/weeks7_8'
    p = pd.read_csv(dest/'predictions.csv', dtype={'fold':str,'row_id':str})
    metrics = pd.read_csv(dest/'fold_metrics.csv', dtype={'fold':str})
    cp = pd.read_csv(dest/'calibration_predictions.csv', dtype={'fold':str,'row_id':str})
    previous = pd.read_csv(root/'outputs/weeks5_6/predictions.csv', dtype={'fold':str,'row_id':str})
    previous = previous.loc[previous.model.eq('Logistic Regression')]
    checked = 0
    for keys, block in p.groupby(['dataset','fold','base_model','method']):
        dataset, fold, base, method = keys
        expected = previous.loc[previous.dataset.eq(dataset)&previous.fold.eq(fold)].set_index('row_id')
        assert not block.row_id.duplicated().any()
        assert set(block.row_id) == set(expected.index)
        np.testing.assert_array_equal(block.y, expected.loc[block.row_id].y)
        row = metrics.loc[metrics.dataset.eq(dataset)&metrics.fold.eq(fold)&metrics.base_model.eq(base)&metrics.method.eq(method)].iloc[0]
        for key, value in score(block.y,block.p).items():
            np.testing.assert_allclose(row[key],value,atol=1e-10,rtol=1e-9)
        calibration = cp.loc[cp.dataset.eq(dataset)&cp.fold.eq(fold)&cp.base_model.eq(base)]
        assert not set(calibration.row_id) & set(block.row_id)
        assert not calibration.row_id.duplicated().any()
        assert len(calibration) == row.calibration_n
        if method != 'raw':
            raw = p.loc[p.dataset.eq(dataset)&p.fold.eq(fold)&p.base_model.eq(base)&p.method.eq('raw')].set_index('row_id')
            cfg = json.loads((root/'config/weeks7_8_experiment.json').read_text())
            recomputed = calibrate(method, calibration.p, calibration.y, raw.loc[block.row_id].p, cfg)
            np.testing.assert_allclose(block.p, recomputed, atol=1e-9, rtol=1e-8)
        if dataset == 'retail':
            assert all(r.split('@')[-1] < fold for r in calibration.row_id)
            if fold == '2010-09-01' and method != 'raw':
                assert row.status == 'identity_no_history'
        if base == 'Logistic Regression' and method == 'raw':
            np.testing.assert_allclose(block.p, expected.loc[block.row_id].p, atol=1e-6,rtol=1e-5)
        checked += 1
    assert checked == 84 and len(metrics) == 84
    audit = pd.read_csv(dest/'split_checks.csv')
    retail = audit.loc[audit.dataset.eq('retail')]
    assert (pd.to_datetime(retail.train_latest_label_end) <= pd.to_datetime(retail.calibration_cutoff)).all()
    metadata = json.loads((dest/'run_metadata.json').read_text())
    assert metadata['final_test_evaluated'] is False
    assert metadata['final_retail_labels_constructed'] is False
    result = dict(passed=True, metric_rows_reconciled=checked, frozen_validation_coverage=True,
        calibration_outer_disjoint=True, temporal_label_maturity=True, final_test_evaluated=False)
    (dest/'validation_checks.json').write_text(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    print(validate(ROOT))
