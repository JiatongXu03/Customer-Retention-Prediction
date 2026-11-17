# Weeks 9–10: outreach targeting and interactive demo

Completed November 3, 2026. This is a milestone scope, not a claim of elapsed time.

## Completed work

Synced GitHub commit 7cc8a59, which already contains weeks 7–8 calibration and drift analysis. Added reproducible targeting analysis using those saved development predictions. The new analysis covers two fixed base models, three probability methods, five outreach budgets and 14 validation splits: 420 metric rows. No new models were trained and no reserved test sets were evaluated.

At a 10% outreach budget, raw-probability results are:

| Dataset | LR recall | Fixed XGBoost recall | Fixed XGBoost lift |
|---|---:|---:|---:|
| IBM telecom | 27.8% | 28.2% | 2.81× |
| Iranian telecom | 50.4% | 53.2% | 5.26× |
| Retail | 14.7% | 14.9% | 1.49× |

Recall is the share of churn cases captured. Precision is the share of selected customers who churn. Lift compares precision with each fold's churn rate. These are equal-weight fold means, not pooled performance or final-test results. The fixed XGBoost candidate is different from the earlier nested-selected model.

## Evaluation protocol

All budgets and methods are declared in config/weeks9_10_experiment.json. Each fold selects ceil(n × budget) records by descending probability, breaking ties by ascending string row ID. This makes isotonic ties reproducible, but a tied block's result can depend on that order. Report all settings rather than choosing a winner after seeing outer validation. Five telecom folds and four retail origins retain their original boundaries. Retail customers can recur across time, so observed fold ranges are descriptive and do not form confidence intervals. Cross-dataset labels differ and lift values should not rank business value across datasets.

The independent validator reads source predictions, repeats ranking using pandas sorting and checks every selected count, hit count, precision, recall and lift. Eleven regression tests pass across the project, including tie ordering, rounding, invalid scores and zero-positive behavior. The original data, prior outputs and reserved final tests remain unchanged.

## Interactive demo

The local website in sites/retention-lab/dist lets visitors compare datasets, models and calibration, vary outreach budgets, inspect fold results and download the selected table. A break-even calculator separates user-supplied costs and customer values from measured evidence. It assumes a uniform cost per contact and value per saved churner. The project has not measured intervention success or profit. The site exports aggregate results only, with a SHA-256 reference to the prediction source, and does not score new customers.

## Reproduce

```bash
.venv/bin/python run_weeks9_10.py
.venv/bin/python validate_weeks9_10.py
.venv/bin/python -m unittest discover -s tests
python3 -m http.server 8765 --directory sites/retention-lab/dist
```

The HTML demo is self-contained and also opens directly in a browser without a server. Online publication is pending approval to upload these aggregate results to Sites. Final slides are in reports/Weeks_9_10_Check_In.pptx.

## Remaining research

Directional and subgroup explanation stability remains unfinished. Freeze a final training, calibration and outreach protocol before opening final holdouts. A future randomized intervention study is needed to estimate actual retention effects. The present work supports capacity planning and research inspection, not a claim of deployed business impact.
