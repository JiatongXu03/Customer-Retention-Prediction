"""Weeks 7–8: training-only cross-fit calibration and development retail drift."""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from advanced_models import temporal_splits, make_model
from baseline_utils import telecom_development, baseline_models, score, RETAIL_FEATURES


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    return np.log(p / (1-p)).reshape(-1, 1)


def calibrate(method, calibration_p, calibration_y, p, cfg):
    if method == 'raw' or len(calibration_y) == 0:
        return np.asarray(p, dtype=float)
    if len(np.unique(calibration_y)) < 2:
        raise ValueError('Calibration requires both classes')
    if method == 'sigmoid':
        model = LogisticRegression(C=cfg['sigmoid_C'], solver='lbfgs', max_iter=3000)
        model.fit(logit(calibration_p), calibration_y)
        return model.predict_proba(logit(p))[:, 1]
    if method == 'isotonic':
        return IsotonicRegression(out_of_bounds='clip').fit(calibration_p, calibration_y).predict(p)
    raise ValueError(method)


def psi(reference, current):
    """Reference-quantile PSI; constant reference uses a central point bin."""
    reference, current = np.asarray(reference), np.asarray(current)
    cuts = np.unique(np.quantile(reference, np.arange(.1, 1, .1)))
    if np.min(reference) == np.max(reference):
        value = float(reference[0])
        cuts = np.array([value, np.nextafter(value, np.inf)])
    edges = np.r_[-np.inf, cuts, np.inf]
    a = np.histogram(reference, edges)[0].astype(float)
    b = np.histogram(current, edges)[0].astype(float)
    a = (a + .5) / (a.sum() + .5*len(a))
    b = (b + .5) / (b.sum() + .5*len(b))
    return float(np.sum((b-a)*np.log(b/a)))


def drift_tables(retail):
    rows, cohorts = [], []
    dates = sorted(retail.cutoff.unique())
    for i, date in enumerate(dates):
        current = retail.loc[retail.cutoff.eq(date)]
        cohorts.append(dict(cutoff=date, n=len(current), unique_customers=current.customer_id.nunique(),
            positive_rate=current.churn_proxy_90d.mean(), retention_rate=1-current.churn_proxy_90d.mean()))
        if i == 0:
            continue
        for reference_name, reference in [
            ('first_origin', retail.loc[retail.cutoff.eq(dates[0])]),
            ('expanding_train', retail.loc[retail.cutoff.lt(date)])]:
            for feature in RETAIL_FEATURES:
                a, b = reference[feature], current[feature]
                sd = a.std(ddof=0)
                rows.append(dict(cutoff=date, reference=reference_name, feature=feature,
                    reference_n=len(a), current_n=len(b), reference_mean=a.mean(), current_mean=b.mean(),
                    reference_median=a.median(), current_median=b.median(),
                    standardized_mean_difference=(b.mean()-a.mean())/sd if sd else np.nan,
                    psi=psi(a,b), customer_overlap_fraction=current.customer_id.isin(reference.customer_id).mean()))
    return pd.DataFrame(rows), pd.DataFrame(cohorts)


def run(root):
    cfg = json.loads((root/'config/weeks7_8_experiment.json').read_text())
    xcfg = json.loads((root/'config/weeks5_6_experiment.json').read_text())
    design = json.loads((root/'config/study_design.json').read_text())['retail']
    dest = root/'outputs/weeks7_8'
    dest.mkdir(parents=True, exist_ok=True)
    retail = pd.read_csv(root/'data/processed/retail_development_snapshots.csv.gz')
    assert set(pd.to_datetime(retail.cutoff)) == set(pd.to_datetime(design['rolling_development_cutoffs']))
    assert (pd.to_datetime(retail.label_end_exclusive) <= pd.Timestamp(design['reserved_test_cutoff'])).all()
    assert not retail.duplicated(['customer_id','cutoff']).any()
    datasets = telecom_development(root)
    datasets['retail'] = dict(X=retail[RETAIL_FEATURES], y=retail.churn_proxy_90d.to_numpy(),
        categorical=[], groups=None, row_ids=(retail.customer_id.astype(str)+'@'+retail.cutoff).to_numpy())
    assignments = pd.read_csv(root/'outputs/tables/telecom_development_fold_assignments.csv')
    predictions, metrics, audit, calibration_rows = [], [], [], []
    for name, d in datasets.items():
        X, y, groups = d['X'], d['y'], d['groups']
        if name == 'retail':
            outer = [(str(retail.iloc[v].cutoff.iloc[0])[:10], t, v) for t,v in temporal_splits(retail)]
        else:
            mapping = assignments.loc[assignments.dataset.eq(name)].set_index('row_index').validation_fold
            folds = mapping.loc[d['row_ids']].to_numpy()
            outer = [(str(f), np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in sorted(set(folds))]
            held = pd.read_csv(root/f'outputs/tables/{name}_split_assignments.csv', dtype={'group_id':str})
            reserved = held.loc[held.split.eq('reserved_test')]
            assert not set(d['row_ids']) & set(reserved.row_index)
            assert not set(groups) & set(reserved.group_id)
        for fold, train, val in outer:
            print(f'{name}: {fold}', flush=True)
            Xt, yt = X.iloc[train].reset_index(drop=True), y[train]
            if name == 'retail':
                history = retail.iloc[train].reset_index(drop=True)
                inner = temporal_splits(history)
            else:
                assert not set(groups[train]) & set(groups[val])
                splitter = (StratifiedGroupKFold if name == 'iranian' else StratifiedKFold)(
                    n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['seed'])
                inner = list(splitter.split(Xt, yt, groups[train] if name == 'iranian' else None))
            models = {'Logistic Regression': baseline_models(X, d['categorical'], name == 'retail')['Logistic Regression'],
                'XGBoost fixed candidate 0': make_model(X, d['categorical'], xcfg, cfg['xgboost_candidate'])}
            for model_name, base in models.items():
                cp, cy = [], []
                for inner_fold, (it, iv) in enumerate(inner, 1):
                    assert not set(train[it]) & set(val)
                    assert not set(train[iv]) & set(val)
                    assert not set(it) & set(iv)
                    if name == 'iranian':
                        assert not set(groups[train[it]]) & set(groups[train[iv]])
                    estimator = clone(base).fit(Xt.iloc[it], yt[it])
                    prob = estimator.predict_proba(Xt.iloc[iv])[:,1].astype(float)
                    cp.extend(prob); cy.extend(yt[iv])
                    calibration_rows.extend(dict(dataset=name, fold=fold, base_model=model_name,
                        inner_fold=inner_fold, row_id=str(d['row_ids'][train[j]]), y=int(yt[j]), p=float(p)) for j,p in zip(iv,prob))
                    audit.append(dict(dataset=name, fold=fold, base_model=model_name, inner_fold=inner_fold,
                        train_n=len(it), calibration_n=len(iv), separation_passed=True,
                        train_latest_label_end=history.iloc[it].label_end_exclusive.max() if name == 'retail' else '',
                        calibration_cutoff=history.iloc[iv].cutoff.min() if name == 'retail' else ''))
                fitted = clone(base).fit(X.iloc[train], y[train])
                raw = fitted.predict_proba(X.iloc[val])[:,1].astype(float)
                for method in cfg['calibration_methods']:
                    p = calibrate(method, cp, cy, raw, cfg)
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    status = 'raw' if method == 'raw' else ('fitted' if cy else 'identity_no_history')
                    metrics.append(dict(dataset=name, fold=fold, base_model=model_name, method=method,
                        status=status, calibration_n=len(cy), **score(y[val], p)))
                    predictions.extend(dict(dataset=name, fold=fold, base_model=model_name, method=method,
                        status=status, row_id=str(d['row_ids'][j]), y=int(y[j]), p=float(prob)) for j,prob in zip(val,p))
    frames = dict(predictions=pd.DataFrame(predictions), fold_metrics=pd.DataFrame(metrics),
        calibration_predictions=pd.DataFrame(calibration_rows), split_checks=pd.DataFrame(audit))
    frames['retail_feature_drift'], frames['retail_cohorts'] = drift_tables(retail)
    reliability = []
    for keys, block in frames['predictions'].groupby(['dataset','fold','base_model','method']):
        bins = np.minimum((block.p.to_numpy()*10).astype(int),9)
        for b in sorted(set(bins)):
            part = block.iloc[np.flatnonzero(bins == b)]
            reliability.append(dict(zip(['dataset','fold','base_model','method'],keys), bin=int(b),
                n=len(part), mean_p=part.p.mean(), observed_rate=part.y.mean()))
    frames['reliability'] = pd.DataFrame(reliability)
    fm = frames['fold_metrics']
    fm['absolute_retention_error_pp'] = fm.retention_error_pp.abs()
    measures = ['brier','log_loss','average_precision','roc_auc','retention_error_pp','absolute_retention_error_pp']
    summary = fm.groupby(['dataset','base_model','method'])[measures].mean().reset_index()
    frames['summary'] = summary
    frames['summary_calibratable_origins'] = fm.loc[fm.calibration_n.gt(0)].groupby(
        ['dataset','base_model','method'])[measures].mean().reset_index()
    for name, frame in frames.items():
        frame.to_csv(dest/f'{name}.csv', index=False)
    paths = ['config/study_design.json','config/weeks5_6_experiment.json','config/weeks7_8_experiment.json',
        'src/calibration_drift.py','src/advanced_models.py','src/baseline_utils.py','src/audit_utils.py',
        'run_weeks7_8.py','validate_weeks7_8.py','data/processed/retail_development_snapshots.csv.gz',
        'data/raw/ibm_telco.csv','data/raw/iranian_churn.csv','outputs/tables/telecom_development_fold_assignments.csv',
        'outputs/tables/ibm_split_assignments.csv','outputs/tables/iranian_split_assignments.csv']
    (dest/'run_metadata.json').write_text(json.dumps(dict(config=cfg, python=platform.python_version(),
        sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, xgboost=xgboost.__version__,
        final_test_evaluated=False, final_retail_labels_constructed=False,
        input_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}), indent=2))
    return dest
