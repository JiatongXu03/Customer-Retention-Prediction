# Methods for the First Progress Report

> Historical Weeks 1–4 methodology. The cumulative project now includes a Weeks 5–6 extension documented in `weeks5_6_methodology.md` and `Second_Progress_Report.html`; later-stage statements below describe the original baseline milestone.

## Scope and stage

This deliverable covers the work scheduled for Weeks 1–4: acquisition, data documentation, cleaning decisions, development-set exploratory analysis, reproducible preprocessing, and initial baseline validation. These are completed analyses, not simulated results or an assertion that four calendar weeks have elapsed.

The project retains three layers: business problem definition; ML analysis and validation; business application and feedback. Weeks 5–8 will address XGBoost, tuning, explanation and calibration improvements. Final-test evaluation is reserved for the last stage.

## Source data and outcomes

- IBM Telco: 7,043 records and 21 source columns, acquired from IBM's official repository. The classic CSV lacks customer-level prediction timestamps; it is a retrospective classification benchmark. `Churn == Yes` is the positive class.
- Iranian Churn: 3,150 records and 14 source columns. Source documentation describes nine months of predictors and a label at month twelve. The file has no verified customer identifier. `Churn == 1` is the positive class.
- Online Retail II: 1,067,371 invoice lines in two source sheets. Positive-class labels indicate **no valid repeat purchase** in the next 90 days, not verified permanent departure.

Raw source hashes are checked before analysis. Structural quality inspection covers the full source files; predictive exploration and evaluation do not use reserved-test outcomes.

## Cleaning and eligibility

IBM TotalCharges is converted to numeric; 11 blank values become missing and are imputed only within training partitions. They are not automatically set to zero using a full-data rule.

Iranian records are preserved even when profiles repeat. The 300 extra exact-duplicate rows cannot be proven to be duplicate customers. Candidate-profile groups exclude Churn, Status and Customer Value; identical groups stay together in the holdout and cross-validation partitions. Status and Customer Value are excluded from the primary baseline because actionable eligibility and the derived value formula remain uncertain, not because leakage has been proven. Age and Age Group are retained with regularization; correlated variables do not receive causal interpretations.

Retail valid purchases require a customer ID, a timestamp, a non-cancellation invoice, positive quantity and price, and removal of extra exact-duplicate source rows. The resulting positive-purchase view is not net revenue. Adjustment and nonstandard stock codes are flagged for review. Excluding customers without IDs may affect representativeness. Each retail snapshot includes customers with a valid purchase in [cutoff−180 days, cutoff). The label uses [cutoff, cutoff+90 days), with end-exclusive boundaries.

## Baseline preprocessing

All learned preprocessing is inside a scikit-learn Pipeline fitted on the training partition of each validation fold. Numeric predictors use median imputation and missingness indicators, followed by standardization. Categorical predictors use most-frequent imputation and one-hot encoding with unknown-category handling.

IBM numeric variables: tenure, MonthlyCharges, TotalCharges. Other non-ID, non-target fields are categorical. Iranian categorical fields: Complains, Age Group, Tariff Plan. Other retained fields are numeric.

Retail uses seven nonnegative pre-cutoff features: recency, distinct-invoice frequency, gross spending, distinct product count, invoice frequency in the recent and previous 90-day halves, and the span between the first and last observed purchase within the lookback. A fixed log1p transformation precedes scaling. These are pre-specified baseline features; no target-driven selection occurs outside validation.

## Models and validation

The two baseline types are a training-prior DummyClassifier (majority predictions at threshold 0.5) and L2 Logistic Regression with C=1, lbfgs, max_iter=3000, no class weighting, and threshold 0.5. No hyperparameter or threshold search is performed. No fitted probability calibrator is applied; Brier scores and reliability plots are diagnostics of the baseline probabilities.

IBM retains 1,409 customer records for final testing. Its 5,634 development records use five shuffled stratified folds. Iranian retains 629 rows and uses five stratified grouped folds on its 2,521 development rows. Grouping is conservative with respect to identical observed candidate profiles but cannot resolve unknown customer identities.

Retail retains the final 2011-09-01 cutoff. Development origins are 2010-06-01, 2010-09-01, 2010-12-01, 2011-03-01 and 2011-06-01. Each of four evaluations trains on earlier origins and validates on the next one. All training-label windows close before the next validation cutoff; all development-validation labels close before the reserved final cutoff. Customer recurrence is allowed to match repeated scoring of existing customers. Customer IDs never enter the features.

## Reporting and interpretation

ROC-AUC and average precision assess ranking. Accuracy, precision, recall and F1 use the fixed threshold 0.5. Brier score and log loss assess probability quality. Precision/recall/lift in the top 10% are illustrative capacity diagnostics for Logistic Regression; the prior model cannot meaningfully rank customers, so its capacity metrics are left undefined.

Mean ± standard deviation summarizes fold variation, not a confidence interval. Retail origins and repeated customers are dependent. Results across datasets have different outcomes, prevalences and validation schemes and are not an industry ranking. Pooled reliability plots are descriptive, particularly for the retail customer-origin records.

Predicted cohort retention is mean(1−p), compared with the observed retained share for the same validation cohort and horizon. No revenue improvement, incremental retention or treatment effect is inferred from retrospective predictions. EDA, coefficients and future SHAP analyses cannot establish a causal chain.

## References

1. [IBM source repository](https://github.com/IBM/telco-customer-churn-on-icp4d)
2. [Iranian Churn, UCI](https://doi.org/10.24432/C5JW3Z)
3. [Online Retail II, UCI](https://doi.org/10.24432/C5CG6D)
4. [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html)
5. [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
6. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
7. [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)

The executed environment uses scikit-learn 1.8.0, recorded in `outputs/environment.json`; the online documentation may reflect a later release.
