# Weeks 7–8 methodology: probability calibration and retail time drift

Implemented October 20, 2026. Development evidence only; no final-test performance is computed and no final retail labels are constructed.

## Implementation review

The Weeks 5–6 pipeline correctly reuses the frozen telecom assignments, protects Iranian profile groups, and enforces retail label maturity at every forward split. Its reliability tables do not fit calibrators. Its nested XGBoost selection uses training labels, so reusing that outer-selected hyperparameter to generate calibration OOF scores would allow calibration labels to influence model selection indirectly. This extension avoids that issue with prespecified, fixed base models rather than claiming to calibrate the Weeks 5–6 tuned winner.

The existing telecom loader reads the supplied raw CSVs, then selects only development rows using saved assignments; reserved outcomes never enter fitting, calibration, scoring or method choice. Retail analysis reads only the supplied development snapshots, without opening the raw workbook or generating new cohorts.

## Frozen experiment and calibration

`config/weeks7_8_experiment.json` specifies Logistic Regression with unchanged preprocessing and XGBoost candidate 0 from the Weeks 5–6 grid. This is an explicit change of experiment scope: Iranian fixed XGBoost is weaker than the earlier tuned model, so its results cannot replace the nested-selection comparison. Retail LR retains log1p; XGBoost retains its original preprocessing. No outer-label hyperparameter, calibration-method or threshold optimization is performed.

Each of the original 14 outer splits fits both models. For each outer training set, base-model pipelines are cloned and fitted inside three stratified inner folds (IBM), three stratified profile-group folds (Iranian), or all available earlier expanding-window validation origins (retail). Calibration receives only these inner held-out probabilities and outcomes. All preprocessing is refitted inside the corresponding training subset. Retail inner training labels must close on or before their calibration origin; every outer training label must close on or before outer validation. Repeated customers across retail origins remain intentional.

Sigmoid calibration fits a one-variable logistic model to clipped probability logits (clip 1e-6, C=1e6, no class weighting). Isotonic uses increasing regression with clipped extrapolation. The base pipeline is refitted on the complete outer training history, and its outer probabilities are transformed by the training-only calibrator. This introduces the usual difference between models used to generate calibration scores and the final refit model; results should be interpreted as this specific cross-fit/refit protocol. Both calibrated methods are reported alongside raw probabilities without selecting an outer-validation winner automatically.

The September 2010 retail fold has one training origin and no honest forward calibration history. Both methods use identity fallback there, explicitly labeled `identity_no_history`. December, March and June have respectively one, two and three historical calibration origins. `summary_calibratable_origins.csv` excludes the fallback fold for all methods, including raw, to retain a matched comparison.

## Probability and drift diagnostics

All metrics reuse the existing scoring function, positive class and 0.5 threshold. Brier and log loss assess probability quality; average precision and ROC AUC assess ranking. Isotonic can create ties and change ranking metrics. Fixed ten-bin reliability diagnostics are saved per dataset, fold, base model and method. Fold means are equally weighted. Mean absolute retention error is the average of absolute fold errors, not the absolute signed mean. Retention error is `(observed churn - mean predicted churn) * 100`; positive means retention overprediction.

Retail drift compares each development origin against (a) June 2010 and (b) its complete expanding training history. Saved diagnostics include feature means, medians, reference-standardized mean differences, customer overlap, and population stability index (PSI). PSI uses unique reference decile cuts with infinite tails, half-count smoothing, and fixed reference edges applied to the current cohort. Constant reference values get a central point bin. No PSI alert threshold, p-value or causal interpretation is assigned. Cohort label rates and forecast errors expose outcome/calibration changes alongside feature changes. The expanding reference weights customer-origin observations, not unique people.

Five origins and four validation windows cannot isolate seasonality, acquisition composition, logging changes or conditional concept drift. Outcome-rate changes alone do not establish pure label shift. Overlapping customers and windows invalidate independent-sample significance claims. Historical calibration can become stale; feature drift does not by itself explain or correct calibration error. No recency-weighted calibration, prior-shift correction, model tuning or subgroup explanation-stability experiment has been added after seeing these results.

## Reproduction and verification

Run `python run_weeks7_8.py`, then `python -m unittest discover -s tests`. The runner writes to `outputs/weeks7_8/`, independently validates saved outputs, and generates two figures. It preserves all Weeks 1–6 evidence.

The validator reconciles 84 metric rows from saved predictions, checks exact original validation-row coverage and outcomes, verifies calibration/outer-row separation, reconstructs calibrated probabilities from saved training OOF scores, checks LR raw prediction agreement with Weeks 5–6 within numerical tolerance, and checks retail label maturity. Tests cover invalid/exact temporal boundaries, absent temporal history, single-class calibration, extreme probabilities and constant-reference PSI. Actual package versions and SHA-256 input hashes are recorded in `run_metadata.json`; the execution environment differs from the pinned pandas/scikit-learn versions, so bitwise equivalence to the pinned environment is not claimed.

Further development analysis can cause researcher overfitting. Freeze any production/final protocol separately before the reserved evaluation; these results do not authorize opening that evaluation.
