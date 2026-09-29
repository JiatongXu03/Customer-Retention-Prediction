# Weeks 5–6 methodology extension

This milestone implements the next two weeks of planned scope; it does not assert elapsed work time. Original targets, exclusions, split assignments and final holdout boundaries are unchanged.

## Model selection

Four XGBoost candidates are specified in `config/weeks5_6_experiment.json` before running the comparison: max depth 2 or 3, 150 or 300 trees, learning rate 0.05, minimum child weight 5, and L2 regularization 5. Full row/column sampling, a fixed seed, one CPU thread and histogram training make the run reproducible. No early stopping or outer-label threshold selection occurs. The primary selection metric is average precision, reflecting the interest in concentrating positive outcomes near the top of the ranking.

Telecom outer folds exactly reuse the original saved assignments. Each outer training set has three stratified inner folds; Iranian folds also enforce the original candidate-profile groups. Each candidate is scored by mean inner average precision, with first-candidate tie-breaking. The selected pipeline is refit on all outer training data and evaluated once on that outer validation fold. Both fixed original baselines are rerun on the same data.

Retail outer validation uses the original four expanding-window origins. Candidate selection uses only forward validation origins inside the outer training history, with every training label window closing on or before the corresponding validation cutoff. The earliest outer evaluation has one training origin and therefore no valid inner temporal split: it uses the pre-specified first candidate. Later origins have one, two and three inner validations, respectively. Repeated customer appearances across time are intentional; customer IDs never enter the features.

All imputation, scaling and categorical encoding occur inside each training pipeline. XGBoost reuses the existing preprocessing, without retail log1p; the Logistic Regression retail baseline retains its original log1p preprocessing. Feature selection and all dataset exclusions remain unchanged.

## Evaluation and interpretation

Metrics use the existing score implementation. XGBoost probabilities are converted to float64 before aggregating and saving so independent CSV reconciliation matches the in-memory calculation. Threshold-based metrics remain at 0.5. Top-10% results retain seeded tie-breaking and represent a hypothetical capacity constraint, not treatment response. Report means equally weight folds; standard deviations are descriptive and are not confidence intervals. Retail cohorts and customers recur, so independence-based significance claims are inappropriate.

Reliability bins use fixed probability intervals of width 0.1 separately by dataset, outer fold and model. No calibrator has been fitted. Cohort retention error is observed positive rate minus mean predicted positive probability, multiplied by 100; a positive value indicates retention overprediction. Both mean signed error and mean absolute fold error are reported.

Native XGBoost `pred_contribs=True` computes TreeSHAP contributions on transformed outer validation rows. Sum-to-margin checks include the bias term. Reported feature importance excludes that bias and averages absolute contributions within a fold, then equally across folds. These are log-odds contributions under the tree model's native explanation semantics, not causal effects or interventional SHAP. One-hot features are not aggregated to parent fields. Directional and subgroup-stability analysis is deferred.

## Verification and reproducibility

The independent validator recomputes every metric from saved predictions, reproduces the prior 28 baseline rows, verifies complete development-only prediction coverage, confirms telecom holdout group separation, and checks chosen candidates against saved inner scores. Runtime assertions check inner/outer disjointness, Iranian group separation, temporal label maturity, and SHAP additivity. Regression tests deliberately submit an invalid temporal boundary, a valid exact boundary, and a single-origin no-tuning case.

Input SHA-256 values and executed package versions are recorded in `outputs/weeks5_6/run_metadata.json`. The original package manifest is retained as historical evidence. Final holdout performance is not computed and final retail labels are not constructed. Repeated development analysis can still induce researcher overfitting, so model-family and calibration choices should be frozen before the final reserved evaluation.
