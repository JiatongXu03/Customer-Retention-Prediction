"""Generate an offline TA report from saved, verified experiment outputs."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import base64
import io
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(root):
    out=root/'outputs/weeks5_6'
    metrics=pd.read_csv(out/'fold_metrics.csv',dtype={'fold':str})
    summary=pd.read_csv(out/'summary.csv')
    checks=json.loads((out/'validation_checks.json').read_text())
    assert checks['passed']
    metadata=json.loads((out/'run_metadata.json').read_text())
    shap=pd.read_csv(out/'shap_importance.csv')
    selected=pd.read_csv(out/'split_checks.csv')
    def table(frame): return frame.to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)
    rows=[]
    for dataset in ['ibm','iranian','retail']:
        part=summary.loc[summary.dataset.eq(dataset)].set_index('model')
        lr=part.loc['Logistic Regression']; boosted=part.loc['XGBoost (nested selection)']
        rows.append(dict(Dataset=dataset,LR_AP=lr.average_precision_mean,XGB_AP=boosted.average_precision_mean,
            AP_change=boosted.average_precision_mean-lr.average_precision_mean,
            LR_AUC=lr.roc_auc_mean,XGB_AUC=boosted.roc_auc_mean,
            LR_Brier=lr.brier_mean,XGB_Brier=boosted.brier_mean))
    comparison=pd.DataFrame(rows)
    comparison.to_csv(out/'model_comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,dataset in zip(axes,['ibm','iranian','retail']):
        part=metrics.loc[metrics.dataset.eq(dataset)]
        for name,color in [('Logistic Regression','#2171b5'),('XGBoost (nested selection)','#db6d28')]:
            p=part.loc[part.model.eq(name)]
            ax.plot(p.fold,p.average_precision,'o-',label=name.split(' (')[0],color=color)
        ax.set_title(dataset.upper()); ax.set_ylabel('Average precision'); ax.set_ylim(0,1)
        ax.tick_params(axis='x',rotation=30); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(out/'fold_average_precision.png',dpi=160)
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=160); plt.close(fig)
    image=base64.b64encode(buf.getvalue()).decode()
    importance=shap.groupby(['dataset','feature']).mean(numeric_only=True).reset_index()
    top=importance.sort_values('mean_absolute_shap',ascending=False).groupby('dataset').head(5).sort_values(['dataset','mean_absolute_shap'],ascending=[True,False])
    retention=metrics.groupby(['dataset','model']).retention_error_pp.agg(mean_error_pp='mean',mae_pp=lambda v:v.abs().mean()).reset_index()
    paragraphs=[]
    for row in rows:
        paragraphs.append(f"{row['Dataset'].upper()}: average precision changed from {row['LR_AP']:.3f} to {row['XGB_AP']:.3f} ({row['AP_change']:+.3f}); Brier score changed from {row['LR_Brier']:.3f} to {row['XGB_Brier']:.3f} (lower is better).")
    paragraphs.append('Development decision: prioritize XGBoost for further Iranian validation; retain Logistic Regression as a competitive IBM option and the retail reference model. Retail average precision is essentially unchanged, while its XGBoost Brier score is slightly worse. Prioritize temporal calibration and drift analysis for retail before expanding model complexity. These are provisional research decisions pending final testing.')
    narrative=' '.join(paragraphs)
    body=f'''<header><p>INFO 5920 · Jiatong Xu (jx429) · Zhiming Zhang (zz939)</p><h1>Customer retention: Weeks 5–6</h1><p class="lead">Nonlinear models, honest validation, and an initial explanation audit</p><p>TA check-in · Prepared September 29, 2026</p></header>
<section><h2>What this milestone adds</h2><p>This is completed work corresponding to the next two-week milestone, not a claim that two calendar weeks were spent. The earlier Weeks 1–4 report established data quality, cohort definitions, and linear baselines. This milestone implements and executes XGBoost across all three datasets, training-only hyperparameter selection, held-out TreeSHAP diagnostics, and calibration/capacity comparisons.</p>
<p><strong>{metadata['inner_fits']} inner search fits and {metadata['outer_fits']} outer model fits completed.</strong> All {checks['reconciled_metric_rows']} metric rows reconcile to saved predictions; all {checks['reproduced_baseline_rows']} prior baseline rows reproduce. Final holdouts remain unused.</p></section>
<section><h2>Research question and results</h2><p>Does nonlinear modeling improve ranking of likely churn/non-repeat customers beyond the existing Logistic Regression baseline?</p><p>{narrative}</p>{table(comparison)}<p>AP = average precision; AUC = ROC-AUC. Values are unweighted means over the same five telecom folds or four retail origins. No cross-industry ranking or significance claim is made.</p><img alt="Average precision by outer validation fold" src="data:image/png;base64,{image}"></section>
<section><h2>Completed work by milestone</h2><ul><li><strong>Week 5 scope:</strong> freeze four XGBoost candidates; implement fold-local preprocessing; reuse the original outer splits; select candidates using three inner telecom folds or earlier retail origins; rerun both baselines in the same environment.</li><li><strong>Week 6 scope:</strong> compare ranking, probability error and top-10% targeting; compute held-out TreeSHAP; export predictions, search results and boundary checks; independently reconcile metrics; prepare this report and the TA discussion notes.</li></ul></section>
<section><h2>How validation stays honest</h2><p>Average precision selects among depths 2/3 and 150/300 trees, with learning rate 0.05, minimum child weight 5 and L2 regularization 5. Preprocessing is fit separately inside every inner training split. Outer outcomes are never used for candidate selection. Iranian profile groups stay together in both inner and outer splits. Retail uses expanding historical origins whose labels close before the next prediction cutoff; recurring customers are allowed. The first retail evaluation has no earlier validation origin and uses candidate 0 without tuning. No early stopping, probability calibration fitting or threshold optimization was performed.</p>{table(selected[['dataset','fold','inner_folds','candidate','selection']])}<p>Outer results are development estimates and can guide the next research stage. Repeated human iteration on these results can still overfit development data; the untouched final tests are required for final confirmation.</p></section>
<section><h2>Probability quality and cohort retention</h2><p>Predicted retention is the cohort mean of 1−p. Positive signed error means predicted retention exceeded observed retention. A high ranking score does not ensure an accurate cohort forecast. Saved reliability bins are diagnostics, not fitted calibration. Retail origins exhibit temporal variation; a pooled calibration plot can conceal this.</p>{table(retention)}<h3>Capacity-constrained targeting</h3>{table(summary[['dataset','model','precision_top10_mean','recall_top10_mean','lift_top10_mean']])}<p>The top 10% is an illustrative contact budget. These metrics measure observed outcome concentration, not whether outreach prevents churn, incremental value, or return on investment. Prior-model capacity metrics are undefined because it cannot rank.</p></section>
<section><h2>Initial explanation audit</h2><p>Native XGBoost TreeSHAP contributions were computed only for each outer validation partition. Their sums were checked against the model’s raw prediction margin. The table shows the five largest average absolute contributions per dataset, equally averaging folds. Contributions are in log-odds units, and one-hot columns are separate predictors. Correlated features may share attribution; these associations do not identify causal intervention targets.</p>{table(top[['dataset','feature','mean_absolute_shap']])}<p>This first audit identifies predictive features. Directional plots, subgroup stability, and domain interpretation remain next-stage work.</p></section>
<section><h2>Limitations and Weeks 7–8</h2><ol><li>Evaluate training-only sigmoid calibration and compare Brier score and cohort retention error using the same protected outer boundaries.</li><li>Inspect SHAP direction and stability, including correlated features and customer subgroups, before proposing business actions.</li><li>Investigate retail drift and horizon sensitivity; do not equate a 90-day non-repeat label with permanent churn.</li><li>Freeze a model-selection and final-evaluation protocol before opening reserved tests. Any intervention benefit requires a separate prospective experiment.</li></ol><p>The Iranian data have no verified customer identity; grouping identical profiles is only a conservative proxy. IBM lacks prediction timestamps. Four dependent retail origins cannot establish robust generalization or confidence intervals. Fold standard deviations describe variation, not statistical significance.</p></section>
<section><h2>TA discussion points</h2><ul><li>Is the nested development comparison sufficient for selecting the final model family?</li><li>Should retail success prioritize risk ranking or cohort-level retention forecasting?</li><li>Should probability calibration and explanation stability take priority over a larger parameter search?</li></ul></section>
<section><h2>Reproduce and inspect</h2><pre>python -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\npython run_weeks5_6.py\npython -m unittest discover -s tests</pre><p>On macOS, XGBoost also requires OpenMP: <code>brew install libomp</code>. Saved evidence is in <code>outputs/weeks5_6/</code>, including predictions, tuning scores, fold metrics, SHAP summaries, reliability bins, SHA-256 input provenance and validation checks. Earlier reports remain historical milestone records.</p></section>'''
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weeks 5–6 Progress Report</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#233044;background:#f6f8fb}header{background:#143753;color:white;padding:32px;border-radius:12px}h1{font-size:36px;margin:8px 0}h2{color:#143753}section{background:white;padding:24px;margin:20px 0;border-radius:12px;overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dde3eb}th{background:#edf2f7}img{max-width:100%}pre{background:#edf2f7;padding:18px}.lead{font-size:21px}@media print{body{background:white;margin:0}section{break-inside:avoid}header{color:black;background:white}}</style>'+body+'</html>'
    (root/'reports/Second_Progress_Report.html').write_text(doc)
    markdown=f'''# Weeks 5–6 progress / TA check-in\n\nPrepared September 29, 2026. Completed milestone scope, not elapsed-time reporting.\n\n## Completed\n\n- Added reproducible XGBoost experiments on IBM, Iranian Churn and retail development data.\n- Executed {metadata['inner_fits']} inner tuning fits and {metadata['outer_fits']} outer fits across 14 protected validation splits.\n- Reproduced all 28 original baseline metric rows and independently reconciled all 42 new metric rows.\n- Added held-out TreeSHAP explanations, reliability bins and top-10% targeting comparisons.\n- Preserved all reserved final holdouts; added temporal boundary regression tests and input hashes.\n\n## Findings\n\n'''+ '\n\n'.join(paragraphs)+'''\n\nThese are development fold means, not final-test results or significance claims. See `Second_Progress_Report.html` for the full tables, limitations and explanation audit.\n\n## Next two weeks\n\n1. Training-only probability calibration, especially retail cohort forecasts.\n2. Directional and subgroup explanation stability.\n3. Freeze the final model protocol, then perform the reserved evaluation at the final stage.\n\n## TA talking points\n\n“We moved from fixed linear baselines to a nested nonlinear comparison on identical outer splits. We can now distinguish better risk ranking from better probability forecasts. Our final tests remain sealed; the next decision is whether calibration and explanation stability matter more than expanding the search.”\n'''
    (root/'reports/Weeks_5_6_Progress.md').write_text(markdown)
    print('Built reports/Second_Progress_Report.html and reports/Weeks_5_6_Progress.md')

if __name__=='__main__': build(Path(__file__).resolve().parent)
