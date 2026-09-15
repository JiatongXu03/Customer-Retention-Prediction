"""Build the self-contained English first-progress report from executed outputs."""
from pathlib import Path
import base64,html,json,re,platform
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
T=ROOT/'outputs/tables';F=ROOT/'outputs/figures'
summary=pd.read_csv(T/'baseline_summary.csv');folds=pd.read_csv(T/'baseline_all_fold_metrics.csv')
logistic=summary.query("model == 'Logistic Regression'").set_index('dataset')
prior=summary.query("model == 'Prior / majority'").set_index('dataset')
retail=folds.query("dataset == 'retail' and model == 'Logistic Regression'").sort_values('fold')
retail_prior=folds.query("dataset == 'retail' and model == 'Prior / majority'").sort_values('fold')
audit=json.loads((ROOT/'outputs/audit_summary.json').read_text())
execution=json.loads((ROOT/'outputs/execution_log.json').read_text())
status=json.loads((ROOT/'outputs/weeks1_4_status.json').read_text())
assert all(x['status']=='passed' for x in execution) and status['final_test_evaluated'] is False
names={'ibm':'IBM telecom','iranian':'Iranian telecom','retail':'Online retail'}
pct=lambda v:f'{v*100:.1f}%'
num=lambda v:f'{v:.3f}'

def image(name):
    return 'data:image/png;base64,'+base64.b64encode((F/name).read_bytes()).decode()

def table(headers,rows,cls=''):
    return '<div class="table-scroll"><table class="'+cls+'"><thead><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(10,4.6));xx=np.arange(len(retail))
ax.plot(xx,retail.observed_retention*100,'o-',color='#17324D',lw=2.5,label='Observed repeat-purchase retention')
ax.plot(xx,retail.predicted_retention*100,'s-',color='#19848A',lw=2,label='Logistic Regression estimate')
ax.plot(xx,retail_prior.predicted_retention*100,'^--',color='#C88D32',lw=1.7,label='Training-prior estimate')
ax.set(xticks=xx,xticklabels=retail.fold,ylim=(25,75),xlabel='Development validation cutoff',ylabel='90-day retained share (%)',title='Ranking improves; cohort forecasts still shift over time')
ax.grid(axis='y',alpha=.16);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=1,frameon=False)
fig.tight_layout();fig.savefig(F/'retail_retention_forecast.png',dpi=170,bbox_inches='tight');plt.close(fig)

eda_specs=[('eda_ibm_tenure.csv','tenure_band','IBM: source churn by tenure','Tenure (months)'),('eda_iranian_complaints.csv','complaint_group','Iranian: source churn by complaints','Complaint status'),('eda_retail_recency.csv','recency_band','Retail: 90-day non-purchase by recency','Purchase recency')]
fig,axs=plt.subplots(1,3,figsize=(13,4.3))
for ax,(file,col,title,xlab) in zip(axs,eda_specs):
    d=pd.read_csv(T/file);xx=np.arange(len(d));ax.bar(xx,d.rate*100,color='#19848A')
    ax.set(xticks=xx,xticklabels=d[col],ylim=(0,100),title=title,ylabel='Outcome share (%)',xlabel=xlab)
    ax.tick_params(axis='x',rotation=25)
    for i,row in d.iterrows():ax.text(i,row.rate*100+2,f'{row.rate:.1%}\nn={int(row.n):,}',ha='center',fontsize=8)
fig.tight_layout();fig.savefig(F/'report_eda_highlights.png',dpi=170,bbox_inches='tight');plt.close(fig)

metric_rows=[]
for ds in names:
    for model in ['Prior / majority','Logistic Regression']:
        r=summary.query('dataset == @ds and model == @model').iloc[0]
        metric_rows.append([names[ds],'<strong>Logistic Regression</strong>' if model.startswith('Logistic') else 'Prior / majority',f'{r.roc_auc_mean:.3f} ± {r.roc_auc_sd:.3f}',num(r.average_precision_mean),pct(r.accuracy_mean),pct(r.recall_05_mean),num(r.f1_05_mean),num(r.brier_mean)])
metrics_table=table(['Dataset','Baseline','ROC-AUC ± SD','AP','Accuracy','Recall','F1','Brier ↓'],metric_rows)
full_rows=[]
for _,r in summary.iterrows():
    full_rows.append([names[r.dataset],r.model,pct(r.precision_05_mean),pct(r.recall_05_mean),num(r.log_loss_mean),f'{r.retention_mae_pp:.2f} pp'])
full_table=table(['Dataset','Model','Precision @ 0.5','Recall @ 0.5','Log loss ↓','Cohort retention MAE ↓'],full_rows)
retention_table=table(['Validation cutoff','Customers','Observed retention','Logistic estimate','Error (predicted − observed)'],[[r.fold,f'{int(r.n):,}',pct(r.observed_retention),pct(r.predicted_retention),f'{r.retention_error_pp:+.2f} pp'] for _,r in retail.iterrows()])
capacity_table=table(['Dataset','Precision in top 10%','Recall in top 10%','Lift over random selection'],[[names[k],pct(r.precision_top10_mean),pct(r.recall_top10_mean),f'{r.lift_top10_mean:.2f}×'] for k,r in logistic.iterrows()])
windows=pd.read_csv(T/'retail_rolling_validation_plan.csv')
window_table=table(['Validation origin','Earlier training origins','Training snapshots','Validation customers','Latest training label end'],[[r.fold,r.train_origins.replace(',', ', '),f'{r.train_n:,}',f'{r.validation_n:,}',r.train_latest_label_end[:10]] for _,r in windows.iterrows()])

sensitivity=pd.read_csv(T/'retail_horizon_sensitivity.csv')
r60=float(sensitivity.query('horizon_days == 60').retained_share.iloc[0]);r90=float(sensitivity.query('horizon_days == 90').retained_share.iloc[0])
ibm_t=pd.read_csv(T/'eda_ibm_tenure.csv');iran_c=pd.read_csv(T/'eda_iranian_complaints.csv');retail_r=pd.read_csv(T/'eda_retail_recency.csv')
complaint=float(iran_c.query("complaint_group == 'Complaint'").rate.iloc[0]);no_complaint=float(iran_c.query("complaint_group == 'No complaint'").rate.iloc[0])
mae=float(logistic.loc['retail','retention_mae_pp']);prior_mae=float(prior.loc['retail','retention_mae_pp'])

css='''
:root{--ink:#17324d;--muted:#5d6e7c;--teal:#167d83;--line:#dbe4e9;--pale:#edf6f5;--gold:#b87816;--bg:#f2f5f7}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:80px}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--teal)}.topbar{position:sticky;top:0;z-index:9;background:#ffffffed;backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}.nav{max-width:1180px;margin:auto;display:flex;align-items:center;gap:20px;padding:13px 24px;font-size:13px;flex-wrap:wrap}.nav .brand{font-weight:750;letter-spacing:.03em;margin-right:auto}.nav a{text-decoration:none;color:var(--muted)}button{font:inherit;border:1px solid var(--line);padding:7px 13px;background:white;border-radius:8px;color:var(--ink);cursor:pointer}main{max-width:1180px;margin:30px auto 60px;padding:0 24px}.hero{background:var(--ink);color:white;padding:44px 44px 36px;border-radius:20px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:750;color:#86d4ce}.hero h1{font-size:clamp(30px,4vw,45px);line-height:1.13;max-width:850px;margin:15px 0 18px;letter-spacing:-.035em}.hero .subtitle{font-size:18px;max-width:900px;color:#dce8ef}.byline{font-size:14px;color:#bfceda;margin-top:22px}.hero .status{display:inline-block;border:1px solid #72929e;border-radius:30px;padding:5px 13px;margin-top:14px;font-size:12px;color:#d7f3ef}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:20px 0}.kpi{border-radius:14px;background:white;border:1px solid var(--line);padding:20px 24px}.kpi b{display:block;font-size:30px;color:var(--teal);line-height:1.2}.kpi span{font-size:13px;color:var(--muted)}section.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:32px 36px;margin:22px 0}h2{font-size:27px;line-height:1.2;letter-spacing:-.025em;margin:0 0 16px}h3{font-size:18px;line-height:1.4;margin:0 0 10px}.section-label{color:var(--teal);font-size:12px;font-weight:750;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px}.lead{font-size:18px;line-height:1.65}.muted,.note{color:var(--muted)}.note{font-size:13px;line-height:1.55}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0 6px}.card{border:1px solid var(--line);border-radius:12px;padding:20px}.card .num{color:var(--teal);font-size:12px;font-weight:750;letter-spacing:.08em}.card p{font-size:14px;margin:10px 0 0}.callout{border-left:4px solid var(--teal);padding:16px 20px;background:var(--pale);border-radius:0 10px 10px 0;margin:20px 0}.callout.amber{border-color:#c38b32;background:#fff7e9}.callout p{margin:0}.table-scroll{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5;font-variant-numeric:tabular-nums}th,td{text-align:left;vertical-align:top;padding:12px 10px;border-bottom:1px solid var(--line)}th{background:#f2f6f8;color:var(--muted);font-size:12px;white-space:nowrap}tbody tr:last-child td{border-bottom:0}figure{margin:24px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:12px;color:var(--muted);margin-top:12px}.split{display:grid;grid-template-columns:1fr 1fr;gap:24px}.compact li{margin:10px 0}.checks{padding-left:20px}.checks li{margin:8px 0}details{border:1px solid var(--line);border-radius:10px;margin:15px 0;padding:15px 18px}summary{cursor:pointer;font-weight:650}details .content{margin-top:18px}.sources{font-size:13px}.sources li{margin-bottom:8px}code{font-size:.86em;padding:2px 5px;border-radius:4px;background:#f0f4f6;color:#29495d}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;background:#f3f6f8;border-radius:8px}.footer{font-size:12px;color:var(--muted);padding:12px 0}.pill{display:inline-block;background:var(--pale);color:var(--teal);border-radius:15px;padding:3px 10px;font-size:12px;white-space:nowrap}.unit{font-size:12px;color:var(--muted)}
@media(max-width:800px){main{padding:0 14px;margin-top:15px}.hero{padding:28px}.cards,.split{grid-template-columns:1fr}.kpis{gap:8px}.kpi{padding:14px}.kpi b{font-size:24px}section.panel{padding:24px}.nav{gap:13px}.nav .brand{width:100%}.nav button{display:none}table{min-width:650px}}
@media print{body{background:white;font-size:10pt}.topbar{display:none}main{max-width:none;margin:0;padding:0}.hero{background:#17324d!important;-webkit-print-color-adjust:exact;print-color-adjust:exact;padding:26px}.hero h1{font-size:28pt}section.panel{border:0;padding:18px 0;margin:10px 0;break-inside:auto}.kpi,.card{break-inside:avoid}.cards{grid-template-columns:repeat(3,1fr)}.table-scroll{overflow:visible}table{min-width:0;font-size:8pt}th,td{padding:7px}h2,h3{break-after:avoid}figure{break-inside:avoid}details{border:0;padding:0}details>summary{display:none}details>.content{display:block!important}a{color:inherit;text-decoration:none}.note{font-size:8pt}}
'''
html_report=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>First Progress Report | Customer Retention | Weeks 1–4</title><style>{css}</style></head><body>
<div class="topbar"><nav class="nav" aria-label="Report sections"><span class="brand">INFO 5920 · FIRST PROGRESS REPORT</span><a href="#project-context">Part I · Context</a><a href="#question">Framework</a><a href="#progress">Part II · Progress</a><a href="#results">Results</a><a href="#next">Next steps</a><button onclick="window.print()">Print / save PDF</button></nav></div>
<main>
<header class="hero"><div class="eyebrow">Project rationale &amp; first progress report</div><h1>From churn prediction<br>to retention decisions</h1><p class="subtitle">A reusable explainable machine learning framework across telecom and online retail.</p><p class="byline">Jiatong Xu (jx429) &nbsp;·&nbsp; Zhiming Zhang (zz939)<br>INFO 5920 · Specialization Project</p><span class="status">Part I · Context and proposal &nbsp; / &nbsp; Part II · Project progress</span></header>
<section class="panel" id="project-context"><div class="section-label">Part I / Research background and proposal</div><h2>What is your context?</h2>
<p class="lead"><strong>A.</strong> Businesses need to anticipate customer loss to plan revenue and decide where to focus retention efforts. Historical records of usage, billing, complaints, and purchasing provide potential warning signals, but their interpretation depends on how the customer relationship works.</p>
<p>In subscription settings, churn can be recorded as a customer leaving a service. In non-subscription retail, departure is not directly observed: a customer who has not purchased recently may have left or may simply buy infrequently. Retention therefore requires an explicit customer group, observation period, and future time horizon.</p>
<p>This creates a methodological problem as well as a business problem. A useful analysis must define an appropriate outcome, evaluate predictions using information available at the decision time, and explain what those predictions can support. Studying several datasets lets us examine which parts of that process can be reused and where the business setting requires a different approach.</p></section>
<section class="panel" id="research-questions"><div class="section-label">I.2 / Main research topic</div><h2>What are you interested in?</h2>
<p class="lead"><strong>A.</strong> Our main research topic is <strong>predicting customer retention with explainable machine learning across different business settings</strong>. We aim to develop and assess a reusable analytical framework using IBM Telco Customer Churn, Iranian Churn, and Online Retail II.</p>
<p>The study addresses three connected questions:</p>
<ol class="checks"><li><strong>Prediction:</strong> How accurately can models identify customers at risk and estimate the retained share of a defined cohort?</li><li><strong>Explanation and comparison:</strong> Which features contribute to predictions, and which patterns remain consistent or differ across datasets?</li><li><strong>Business application:</strong> How should the analysis be adapted to support retention planning and outreach priorities in each setting?</li></ol>
<p class="note">The intended contribution is a reusable analysis process with evidence about its limits. Model explanations describe predictive associations; causal claims about why customers leave or how to prevent departure require additional evidence.</p></section>
<section class="panel" id="question"><div class="section-label">I.3 / Three-layer framework</div><h2>How will we study the problem?</h2><p>We organize the study into three layers, so each modeling choice follows from a defined business question.</p>
<div class="cards"><article class="card"><div class="num">LAYER 1</div><h3>Business problem definition</h3><p>Define who is eligible, what “retained” means, the prediction horizon, and the decision the estimate should support.</p></article><article class="card"><div class="num">LAYER 2</div><h3>ML analysis and validation</h3><p>Construct samples and features, prevent leakage, compare models, evaluate probabilities, and assess which findings transfer.</p></article><article class="card"><div class="num">LAYER 3</div><h3>Business application and feedback</h3><p>Use validated outputs for cohort estimates and outreach priorities, then test interventions and monitor outcomes.</p></article></div>
<p>Within this framework, we will compare Logistic Regression with XGBoost and use SHAP to examine model predictions. The final deliverables will include reproducible notebooks, a cross-dataset model comparison, and a practical guide to adapting the analysis to a new retention problem.</p></section>
<section class="panel" id="progress"><div class="section-label">Part II / Project progress · Weeks 1–4</div><h2>Where the project stands</h2>
<p class="lead">The project has moved from defining the retention problem to testing an initial modeling approach across all three datasets. Data preparation and baseline validation are in place, allowing us to examine both predictive performance and the limits of a shared framework.</p>
<p>The main design decision has been to keep the modeling pipeline consistent while adapting the outcome and validation to each setting. Telecom uses the supplied churn labels; retail uses no repeat purchase within 90 days. IBM uses stratified validation, Iranian data keep identical profiles together, and retail is evaluated forward in time. Final-test evaluation remains reserved.</p>
<details><summary>Development questions and the four-week scope</summary><div class="content">
<h3>What is getting built, and what are you developing?</h3>
<p><strong>A.</strong> The current implementation consists of six executable notebooks with data-loading and cleaning code, retail cohort construction, preprocessing pipelines, and baseline evaluation. The week-by-week deliverables below show how these components fit together.</p>
<h3>What tech things need done in 4wks?</h3>
<p><strong>A.</strong> The first four-week milestone requires the following technical deliverables. These tasks are now completed; the week labels describe the planned work sequence.</p>
<div class="table-scroll"><table><thead><tr><th>Week</th><th>Technical work</th><th>Completed evidence</th></tr></thead><tbody>
<tr><td>1</td><td>Acquire and audit all three datasets; define outcomes and data eligibility.</td><td>Source checks, missingness and duplicate profiles, cleaning rules, and target documentation.</td></tr>
<tr><td>2</td><td>Construct retail cohorts and features; freeze final holdouts and validation rules.</td><td>180-day histories, 90-day labels, telecom split assignments, and a forward retail validation schedule.</td></tr>
<tr><td>3</td><td>Explore development data and implement preprocessing within training folds.</td><td>EDA figures, horizon sensitivity, and imputation, encoding, transformation, and scaling pipelines.</td></tr>
<tr><td>4</td><td>Fit and evaluate initial baselines; reconcile results and prepare the first report.</td><td>Prior/majority and Logistic Regression baselines: 28 model fits, saved predictions and metrics, and six executed notebooks.</td></tr>
</tbody></table></div>
</div></details>
<details><summary>Data definitions and preparation</summary><div class="content" id="data"><h3>Three datasets, with explicit outcome and quality checks</h3>
{table(['Setting','Source size','Outcome / time structure','Key preparation decision'],[
['IBM telecom','7,043 customers<br>21 columns','Source churn label; no customer-level prediction dates','Convert 11 blank TotalCharges values to missing; use as a retrospective benchmark.'],
['Iranian telecom','3,150 records<br>14 columns','Nine months of predictors; state at month twelve','Keep identical candidate profiles in the same split; exclude Status and derived Customer Value from the primary baseline.'],
['Online retail','1,067,371 invoice lines<br>8 source columns','180-day history → purchase in the next 90 days','Construct customer snapshots; identify exclusions and keep gross spending separate from net revenue.']])}
<p>All source files were acquired and checked. The retail positive-purchase view retains <strong>{audit['valid_rows']:,} lines from {audit['valid_customers']:,} customers</strong> after sequential exclusions. Missing customer IDs affect 243,007 source rows; the resulting customer-level analysis does not represent anonymous buyers. Iranian data include 300 extra exact-duplicate rows and no verified customer ID; identical profiles are grouped rather than automatically discarded.</p>
<p class="note">Data sources: <a href="https://github.com/IBM/telco-customer-churn-on-icp4d">IBM</a>, <a href="https://doi.org/10.24432/C5JW3Z">UCI Iranian Churn</a>, and <a href="https://doi.org/10.24432/C5CG6D">UCI Online Retail II</a>. Raw row counts and exclusions are measured from the supplied files. The report covers the assigned work scope, not elapsed calendar time.</p></div></details>
<details><summary>Validation design and execution checks</summary><div class="content" id="design"><h3>How are you building for evaluation?</h3><p><strong>A.</strong> We separate development validation from final testing, fit preprocessing within training partitions, and adapt the split design to each dataset.</p>
<div class="split"><div><h3>Shared baseline pipeline</h3><ol class="checks"><li>Exclude IDs, future outcomes and label-window metadata.</li><li>Fit median imputation, scaling and categorical encoding <strong>inside each training fold</strong>.</li><li>Compare a training-prior/majority baseline with L2 Logistic Regression.</li><li>Report ranking, threshold, probability and cohort-level errors.</li></ol><p class="note">Logistic Regression: C=1, lbfgs, max_iter=3,000; no class weighting or parameter search. Threshold metrics use 0.5. Retail nonnegative numeric features use a fixed log1p transform before scaling.</p></div><div><h3>Dataset-specific validation</h3><ul class="checks"><li><strong>IBM:</strong> 5 stratified development folds; 5,634 development customers; 1,409 reserved.</li><li><strong>Iranian:</strong> 5 stratified grouped folds; 2,521 development records; 629 reserved.</li><li><strong>Retail:</strong> 4 forward validation origins with expanding training history; final 2011-09-01 origin remains reserved.</li></ul><p class="note">Retail customers may recur across time, matching repeated scoring. Training-label windows end before each validation cutoff. Validation observations are not all independent customers.</p></div></div>
<details><summary>View retail temporal validation schedule</summary><div class="content">{window_table}<p class="note">The initial training origin is 2010-06-01. All development-validation labels close before the reserved final cutoff. Neither its labels nor its performance have been computed.</p></div></details>
<p class="note">Methods follow <a href="https://scikit-learn.org/stable/modules/compose.html">Pipeline guidance</a> and <a href="https://scikit-learn.org/stable/modules/cross_validation.html">validation guidance</a>. The current stage inspects probability quality; fitting calibration improvements remains later work.</p><h3>Are all test sets, data, models going to run as expected?</h3>
<p><strong>A.</strong> The implemented development workflow has run successfully: all 33 code cells across six notebooks executed, all 28 baseline fits completed without convergence warnings, and saved metrics were reconciled with predictions. Group-separation and temporal-boundary checks passed. Final-test partitions and the retail final cutoff are defined, but final-test evaluation has deliberately not run and the final retail labels have not been constructed. XGBoost and SHAP are also not yet implemented, so their execution and final performance remain to be verified in later stages.</p>
</div></details>
</section>
<section class="panel" id="results"><div class="section-label">II.1 / Initial findings</div><h2>What the initial models show</h2><p>Logistic Regression improves on the prior baseline in ranking and individual probability quality across all three datasets. This gives us a useful starting point for model comparison, although the different outcomes and validation schemes mean the scores should be interpreted within each dataset.</p>
{metrics_table}
<p class="note"><strong>Reading the table:</strong> values are equal-weight fold means. ROC-AUC shows mean ± fold standard deviation, not a confidence interval. AP = average precision. Lower Brier is better. Precision, recall and F1 use threshold 0.5. Telecom has 5 folds per dataset; retail has 4 temporal folds. Different outcomes and prevalences make cross-dataset ranking inappropriate.</p>
<div class="callout"><p><strong>What improved:</strong> Logistic Regression outperforms the training-prior baseline in ROC-AUC, average precision and Brier score in each dataset. However, recall at threshold 0.5 is only {pct(logistic.loc['ibm','recall_05_mean'])} in IBM and {pct(logistic.loc['iranian','recall_05_mean'])} in Iranian data. A default threshold is not yet a retention policy.</p></div>
<details><summary>Additional metrics and probability diagnostics</summary><div class="content">{full_table}<figure><img src="{image('baseline_reliability.png')}" alt="Untuned logistic regression reliability plots for all three datasets"><figcaption>Raw baseline probabilities from held-out development predictions. Pooled retail predictions span different origins and repeated customers; this is a descriptive check, not proof of future calibration.</figcaption></figure></div></details>
<details><summary>Exploratory patterns behind the next research questions</summary><div class="content" id="eda"><h3>Different measurements suggest different working hypotheses</h3>
<figure><img src="{image('report_eda_highlights.png')}" alt="Development-only outcome rates by tenure in IBM, complaints in Iranian data, and purchase recency in retail"><figcaption>Unadjusted development-only associations. Telecom outcomes are source churn; retail outcome is no purchase within 90 days. Counts are shown above the bars. These panels do not establish a common causal chain.</figcaption></figure>
<div class="cards"><article class="card"><h3>Relationship duration</h3><p>IBM churn is {pct(ibm_t.rate.iloc[0])} for tenure of 0–12 months versus {pct(ibm_t.rate.iloc[-1])} at 49–72 months. Contract groups also differ. Tenure and contract structure are hypotheses to investigate jointly.</p></article><article class="card"><h3>Service friction</h3><p>Iranian churn is {pct(complaint)} among records with a complaint versus {pct(no_complaint)} without one. The large association warrants timing and robustness checks; it is not a treatment-effect estimate.</p></article><article class="card"><h3>Recent engagement</h3><p>Retail non-purchase is {pct(retail_r.rate.iloc[0])} for purchases within 30 days versus {pct(retail_r.rate.iloc[-1])} for recency above 90 days. Activity timing is useful, but retention depends on the chosen horizon.</p></article></div>
<p class="note">At the same first retail development origin, observed retained share changes from {pct(r60)} over 60 days to {pct(r90)} over 90 days. This pre-specified sensitivity confirms that the business definition materially changes the label. The 90-day primary definition was not selected by model score.</p></div></details></section>
<section class="panel" id="business"><div class="section-label">II.2 / An emerging question</div><h2>Does better ranking mean a better retention forecast?</h2>
<p class="lead">Retail ROC-AUC is reasonably consistent across origins ({retail.roc_auc.min():.3f}–{retail.roc_auc.max():.3f}), yet its cohort retention estimates can be materially wrong. The gap shifts our attention from identifying high-risk customers to checking whether the same probabilities support cohort-level planning.</p>
<figure><img src="{image('retail_retention_forecast.png')}" alt="Observed and predicted 90-day retail retention at four forward validation origins"><figcaption>For each eligible validation cohort, estimated retention is mean(1 − predicted non-purchase probability). The observed retained share uses the identical 90-day horizon.</figcaption></figure>
<details><summary>Retention estimates by validation period</summary><div class="content">{retention_table}</div></details>
<div class="callout amber"><p><strong>Forecasting gap:</strong> Logistic Regression’s mean absolute cohort-retention error is <strong>{mae:.2f} percentage points</strong>, versus {prior_mae:.2f} for the training-prior baseline. Better ranking and individual Brier scores have not yet delivered a better aggregate retention forecast. Signed average error would hide errors that cancel across periods.</p></div>
<p>Changing outcome prevalence is visible across origins. Seasonality, cohort composition and temporal drift are plausible explanations to examine next; these experiments do not identify which mechanism caused the shifts. The next stage should evaluate time-aware features and calibration within development data.</p>
<details><summary>Illustrative outreach capacity: top 10% of predicted risk</summary><div class="content">{capacity_table}<p class="note">Equal-weight fold means. “Top 10%” is a capacity scenario, not an optimized campaign recommendation. Lift compares event concentration with random selection within the same validation cohort. It does not estimate customers saved, incremental retention or campaign ROI.</p></div></details>
</section>
<section class="panel" id="next"><div class="section-label">II.3 / Open questions and next steps</div><h2>What we need to understand next</h2>
<p>The central open question is whether the retail forecasting gap reflects changing customer cohorts, time-related behavior, or probability calibration. We will investigate these possibilities within the existing development validation design and review the outcome definition and adjustment codes before drawing broader conclusions.</p>
<p>During Weeks 5–8, we will compare XGBoost with the current Logistic Regression baseline, assess probability calibration, and use SHAP to examine which predictive patterns are stable across settings. The aim is to determine what additional modeling contributes to the original retention questions.</p>
<p>Cross-dataset conclusions remain provisional. IBM lacks prediction timestamps, Iranian customer identities cannot be verified, and retail repeat purchasing is a proxy for retention. These limits will shape the final synthesis and the scope of any business recommendations.</p>
<details><summary>Data and interpretation limits</summary><div class="content" id="limits"><h3>A working baseline is not a deployment claim</h3>
<div class="split"><div><h3>Data and generalization</h3><ul class="compact"><li>IBM lacks prediction timestamps; its validation does not establish prospective lead time.</li><li>Iranian grouping addresses identical observed profiles, not unknown customer identity. Status and derived value semantics remain unresolved.</li><li>Retail excludes anonymous transactions and still includes positive adjustment/nonstandard stock codes flagged for review. Gross spending is not profit.</li></ul></div><div><h3>Inference and action</h3><ul class="compact"><li>Retail repeat purchasing is not contractual retention. Different horizons produce different targets.</li><li>Fold variability is descriptive; repeated customers and temporal origins are dependent.</li><li>Associations and model coefficients are not causal effects. Effective outreach requires intervention evidence.</li></ul></div></div>
<p class="note">The framework has been exercised in three datasets covering two business models. Universal conclusions, cross-domain transfer of fitted weights, and a causal “friction → disengagement → churn” chain have not been established.</p></div></details>
<p><strong>For discussion:</strong> Should the next stage put more emphasis on explaining recurring risk factors or on improving the retail cohort forecast? Is a 90-day repeat-purchase horizon appropriate for the retail comparison?</p>
<p class="note">Final testing and the cross-dataset synthesis are planned for Weeks 9–12, after modeling decisions are fixed.</p></section>
<section class="panel" id="appendix"><div class="section-label">Appendix / Reproducible evidence</div><h2>From report claims back to code</h2>
<details><summary>Notebook and output guide</summary><div class="content">{table(['Notebook','Completed work'],[
['01 · Source and quality audit','Source-byte checks, field profiles, missingness, duplicates and retail cleaning waterfall.'],
['02 · Targets and splits','Frozen telecom holdouts, time-window coverage and first retail cohort.'],
['03 · Cross-dataset framework','Business-concept mapping and the three-layer analysis interface.'],
['04 · Development EDA','Outcome relationships, descriptive figures and the 60/90-day retail sensitivity.'],
['05 · Baseline validation','Training-fold preprocessing, model fits, probabilities, fold metrics and diagnostics.'],
['06 · Evidence checks','Metric reconciliation from predictions and holdout/time-boundary checks.']])}<p>Supporting evidence includes <code>baseline_all_fold_metrics.csv</code>, <code>baseline_validation_predictions.csv</code>, <code>telecom_development_fold_assignments.csv</code>, <code>retail_rolling_validation_plan.csv</code>, and <code>weeks1_4_validation_checks.json</code>.</p><p class="note">Python {platform.python_version()}, pandas 2.2.3, scikit-learn 1.8.0. Standard Python notebook cells were executed sequentially by the included runner, with real stdout and figures saved; no Jupyter kernel was required. The notebooks also remain editable in Jupyter/VS Code. No participant fieldwork or campaign execution is claimed.</p></div></details>
<ol class="sources"><li>IBM. <a href="https://github.com/IBM/telco-customer-churn-on-icp4d">Telco Customer Churn source repository</a>.</li><li>UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5JW3Z">Iranian Churn</a> (2020).</li><li>Daqing Chen. <a href="https://doi.org/10.24432/C5CG6D">Online Retail II</a> (2012). UCI Machine Learning Repository.</li><li>scikit-learn. <a href="https://scikit-learn.org/stable/modules/compose.html">Pipelines and composite estimators</a>; <a href="https://scikit-learn.org/stable/modules/cross_validation.html">cross-validation</a>; <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html">StratifiedGroupKFold</a>; <a href="https://scikit-learn.org/stable/modules/model_evaluation.html">metrics and scoring</a>.</li></ol>
<p class="note">All numerical results in this report are calculated from the accompanying project outputs. Dataset and methodological sources support definitions and design choices, not externally claimed model results.</p></section>
<footer class="footer">INFO 5920 · First Progress Report · Weeks 1–4 scope &nbsp;|&nbsp; Jiatong Xu & Zhiming Zhang<br>Self-contained HTML · Charts embedded · No external scripts or data requests</footer>
</main><script>window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>d.open=true));</script></body></html>'''
(ROOT/'reports/First_Progress_Report.html').write_text(html_report,encoding='utf-8')
# Browser-readable notebook companions retain the actual saved outputs.
def render_md(s):
    lines=[]
    for l in s.splitlines():
        esc=html.escape(l)
        if l.startswith('#'):
            n=min(6,len(l)-len(l.lstrip('#')));lines.append(f'<h{n}>{html.escape(l[n:].strip())}</h{n}>')
        elif l.strip():lines.append('<p>'+esc+'</p>')
    return '\n'.join(lines)
for p in (ROOT/'notebooks').glob('*.ipynb'):
    nb=json.loads(p.read_text());blocks=[]
    for c in nb['cells']:
        s=''.join(c['source'])
        if c['cell_type']=='markdown':blocks.append(render_md(s));continue
        out=[]
        for o in c['outputs']:
            if o['output_type']=='stream':out.append('<pre>'+html.escape(''.join(o['text']))+'</pre>')
            elif 'image/png' in o.get('data',{}):out.append('<img style="max-width:100%" alt="Notebook figure" src="data:image/png;base64,'+o['data']['image/png']+'">')
        blocks.append('<details open><summary>Executed cell '+str(c['execution_count'])+'</summary><div class="content"><pre>'+html.escape(s)+'</pre>'+''.join(out)+'</div></details>')
    p.with_suffix('.html').write_text('<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+p.stem+'</title><style>'+css+'</style><main><section class="panel">'+''.join(blocks)+'</section></main></html>',encoding='utf-8')
print('Report generated:',ROOT/'reports/First_Progress_Report.html')
print('Retail mean absolute retention error:',mae,'prior:',prior_mae)
