"""Development-only nested XGBoost evaluation; run via run_weeks5_6.py."""
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from baseline_utils import telecom_development, preprocessing, baseline_models, score, RETAIL_FEATURES


def temporal_splits(frame):
    dates = sorted(frame.cutoff.unique())
    result = []
    for date in dates[1:]:
        train = np.flatnonzero(frame.cutoff.lt(date))
        val = np.flatnonzero(frame.cutoff.eq(date))
        assert pd.to_datetime(frame.iloc[train].label_end_exclusive).max() <= pd.Timestamp(date)
        result.append((train, val))
    return result


def make_model(X, cats, cfg, candidate):
    return Pipeline([
        ('prepare', preprocessing(X, cats)),
        ('model', xgb.XGBClassifier(**cfg['candidates'][candidate], **cfg['fixed_parameters'],
             random_state=cfg['seed'], n_jobs=1, tree_method='hist', eval_metric='logloss'))])


def run(root):
    cfg = json.loads((root/'config/weeks5_6_experiment.json').read_text())
    destination = root/'outputs/weeks5_6'
    destination.mkdir(parents=True, exist_ok=True)
    metrics, predictions, searches, explanations, boundaries = [], [], [], [], []
    assignments = pd.read_csv(root/'outputs/tables/telecom_development_fold_assignments.csv')
    datasets = telecom_development(root)
    retail = pd.read_csv(root/'data/processed/retail_development_snapshots.csv.gz')
    design = json.loads((root/'config/study_design.json').read_text())
    reserved = pd.Timestamp(design['retail']['reserved_test_cutoff'])
    assert set(pd.to_datetime(retail.cutoff)) == set(pd.to_datetime(design['retail']['rolling_development_cutoffs']))
    assert (pd.to_datetime(retail.label_end_exclusive) <= reserved).all()
    datasets['retail'] = dict(X=retail[RETAIL_FEATURES], y=retail.churn_proxy_90d.to_numpy(),
        categorical=[], groups=None, row_ids=(retail.customer_id.astype(str)+'@'+retail.cutoff).to_numpy())
    for name, d in datasets.items():
        X, y, groups = d['X'], d['y'], d['groups']
        if name == 'retail':
            outer = [(str(retail.iloc[v].cutoff.iloc[0])[:10], t, v) for t,v in temporal_splits(retail)]
        else:
            mapping = assignments.loc[assignments.dataset.eq(name)].set_index('row_index').validation_fold
            folds = mapping.loc[d['row_ids']].to_numpy()
            outer = [(str(f), np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in sorted(set(folds))]
            held = pd.read_csv(root/f'outputs/tables/{name}_split_assignments.csv', dtype={'group_id': str})
            test = held.loc[held.split.eq('reserved_test')]
            assert not set(d['row_ids']) & set(test.row_index)
            assert not set(groups) & set(test.group_id)
        for fold, train, val in outer:
            print(f'{name}: outer {fold}', flush=True)
            Xt, yt = X.iloc[train].reset_index(drop=True), y[train]
            if name == 'retail':
                inner = temporal_splits(retail.iloc[train].reset_index(drop=True))
            else:
                splitter = (StratifiedGroupKFold(cfg['inner_telecom_folds'], shuffle=True, random_state=cfg['seed'])
                    if name == 'iranian' else StratifiedKFold(cfg['inner_telecom_folds'], shuffle=True, random_state=cfg['seed']))
                inner = list(splitter.split(Xt, yt, groups[train] if name == 'iranian' else None))
                assert not set(groups[train]) & set(groups[val])
            for it, iv in inner:
                assert not set(train[it]) & set(val)
                assert not set(train[iv]) & set(val)
                if name == 'iranian':
                    assert not set(groups[train[it]]) & set(groups[train[iv]])
            means = []
            if inner:
                for candidate in range(len(cfg['candidates'])):
                    scores = []
                    for inner_fold, (it, iv) in enumerate(inner, 1):
                        model = make_model(Xt, d['categorical'], cfg, candidate)
                        model.fit(Xt.iloc[it], yt[it])
                        ap = average_precision_score(yt[iv], model.predict_proba(Xt.iloc[iv])[:, 1])
                        scores.append(ap)
                        searches.append(dict(dataset=name, fold=fold, candidate=candidate, inner_fold=inner_fold,
                            train_n=len(it), validation_n=len(iv), average_precision=ap))
                    means.append(float(np.mean(scores)))
                selected = int(np.argmax(means))
            else:
                selected = 0
            boundaries.append(dict(dataset=name, fold=fold, train_n=len(train), validation_n=len(val),
                inner_folds=len(inner), candidate=selected, selection='inner AP' if inner else 'pre-specified fallback',
                separation_passed=True))
            models = baseline_models(X, d['categorical'], log_numeric=name == 'retail')
            models['XGBoost (nested selection)'] = make_model(X, d['categorical'], cfg, selected)
            for model_name, model in models.items():
                model.fit(X.iloc[train], y[train])
                p = model.predict_proba(X.iloc[val])[:, 1].astype(np.float64)
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                metrics.append(dict(dataset=name, fold=fold, model=model_name, **score(y[val], p, rankable=model_name != 'Prior / majority')))
                predictions.extend(dict(dataset=name, fold=fold, model=model_name, row_id=str(d['row_ids'][j]), y=int(y[j]), p=float(prob)) for j,prob in zip(val,p))
                if model_name.startswith('XGBoost'):
                    transformed = model.named_steps['prepare'].transform(X.iloc[val])
                    booster = model.named_steps['model'].get_booster()
                    matrix = xgb.DMatrix(transformed)
                    contributions = booster.predict(matrix, pred_contribs=True)
                    margin = booster.predict(matrix, output_margin=True)
                    assert np.allclose(contributions.sum(axis=1), margin, atol=1e-5)
                    explanations.extend(dict(dataset=name, fold=fold, feature=f, mean_absolute_shap=float(a), mean_signed_shap=float(s))
                        for f,a,s in zip(model.named_steps['prepare'].get_feature_names_out(),
                            np.abs(contributions[:, :-1]).mean(axis=0), contributions[:, :-1].mean(axis=0)))
    frames = dict(fold_metrics=pd.DataFrame(metrics), predictions=pd.DataFrame(predictions),
        tuning_scores=pd.DataFrame(searches), shap_importance=pd.DataFrame(explanations), split_checks=pd.DataFrame(boundaries))
    for name, frame in frames.items():
        frame.to_csv(destination/f'{name}.csv', index=False)
    numeric = ['roc_auc','average_precision','brier','log_loss','precision_top10','recall_top10','lift_top10','retention_error_pp']
    summary = frames['fold_metrics'].groupby(['dataset','model'])[numeric].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(destination/'summary.csv', index=False)
    reliability = []
    for keys, part in frames['predictions'].groupby(['dataset','fold','model']):
        bins = np.minimum((part.p.to_numpy()*10).astype(int),9)
        for b in sorted(set(bins)):
            block = part.iloc[np.flatnonzero(bins == b)]
            reliability.append(dict(dataset=keys[0],fold=keys[1],model=keys[2],bin=int(b),n=len(block),mean_p=block.p.mean(),observed_rate=block.y.mean()))
    pd.DataFrame(reliability).to_csv(destination/'reliability.csv',index=False)
    inputs = ['config/study_design.json','config/weeks5_6_experiment.json','data/raw/ibm_telco.csv','data/raw/iranian_churn.csv',
        'data/processed/retail_development_snapshots.csv.gz','outputs/tables/telecom_development_fold_assignments.csv',
        'outputs/tables/ibm_split_assignments.csv','outputs/tables/iranian_split_assignments.csv', 'src/advanced_models.py','src/baseline_utils.py']
    manifest = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
    (destination/'run_metadata.json').write_text(json.dumps(dict(python=platform.python_version(),sklearn=sklearn.__version__,
        xgboost=xgb.__version__,numpy=np.__version__,pandas=pd.__version__,config=cfg,input_sha256=manifest,
        final_test_evaluated=False,outer_splits=len(boundaries),inner_fits=len(searches),outer_fits=len(metrics)),indent=2))
    return destination
