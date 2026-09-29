"""Independent reconciliation of saved predictions, boundaries and model comparisons."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from baseline_utils import score


def validate(root):
    out = root/'outputs/weeks5_6'
    pred = pd.read_csv(out/'predictions.csv',dtype={'fold':str,'row_id':str})
    metrics = pd.read_csv(out/'fold_metrics.csv',dtype={'fold':str})
    assert len(metrics) == 42
    assert not pred.duplicated(['dataset','fold','model','row_id']).any()
    assert np.isfinite(pred.p).all() and pred.p.between(0,1).all()
    for (dataset,fold,model), part in pred.groupby(['dataset','fold','model']):
        row = metrics.loc[(metrics.dataset==dataset)&(metrics.fold==fold)&(metrics.model==model)].iloc[0]
        actual = score(part.y,part.p,rankable=model!='Prior / majority')
        for key,value in actual.items():
            assert np.isclose(row[key],value,atol=1e-9,equal_nan=True), (dataset,fold,model,key)
    old = pd.read_csv(root/'outputs/tables/baseline_all_fold_metrics.csv',dtype={'fold':str})
    paired = metrics.merge(old,on=['dataset','fold','model'],suffixes=('_new','_old'))
    assert len(paired)==28
    for metric in ['roc_auc','average_precision','brier','log_loss']:
        assert np.allclose(paired[metric+'_new'],paired[metric+'_old'],atol=1e-8)
    for name in ['ibm','iranian']:
        assignment = pd.read_csv(root/f'outputs/tables/{name}_split_assignments.csv',dtype={'group_id':str})
        dev = assignment.loc[assignment.split.eq('development')]
        held = assignment.loc[assignment.split.eq('reserved_test')]
        assert not set(dev.group_id)&set(held.group_id)
        for _,part in pred.loc[pred.dataset.eq(name)].groupby('model'):
            assert set(part.row_id.astype(int)) == set(dev.row_index)
            assert len(part)==len(dev)
    retail = pd.read_csv(root/'data/processed/retail_development_snapshots.csv.gz')
    dates = sorted(retail.cutoff.unique())
    expected = set((retail.loc[retail.cutoff.ne(dates[0]),'customer_id'].astype(str)+'@'+retail.loc[retail.cutoff.ne(dates[0]),'cutoff']))
    for _,part in pred.loc[pred.dataset.eq('retail')].groupby('model'):
        assert set(part.row_id)==expected
        assert len(part)==len(expected)
    selection = pd.read_csv(out/'split_checks.csv',dtype={'fold':str})
    tuning = pd.read_csv(out/'tuning_scores.csv',dtype={'fold':str})
    for _,row in selection.iterrows():
        scores = tuning.loc[tuning.dataset.eq(row.dataset)&tuning.fold.eq(row.fold)]
        if len(scores):
            assert int(scores.groupby('candidate').average_precision.mean().idxmax()) == row.candidate
        else:
            assert row.dataset=='retail' and row.candidate==0 and row.inner_folds==0
    result = dict(passed=True, reconciled_metric_rows=len(metrics), reproduced_baseline_rows=len(paired),
        final_holdout_predictions=0, prediction_rows=len(pred), candidate_selection_verified=True,
        retail_prediction_coverage_verified=True)
    (out/'validation_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    validate(Path(__file__).resolve().parent)
