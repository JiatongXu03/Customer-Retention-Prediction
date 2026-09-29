# Weeks 5–6 progress / TA check-in

Prepared September 29, 2026. Completed milestone scope, not elapsed-time reporting.

## Completed

- Added reproducible XGBoost experiments on IBM, Iranian Churn and retail development data.
- Executed 144 inner tuning fits and 42 outer fits across 14 protected validation splits.
- Reproduced all 28 original baseline metric rows and independently reconciled all 42 new metric rows.
- Added held-out TreeSHAP explanations, reliability bins and top-10% targeting comparisons.
- Preserved all reserved final holdouts; added temporal boundary regression tests and input hashes.

## Findings

IBM: average precision changed from 0.650 to 0.660 (+0.010); Brier score changed from 0.137 to 0.135 (lower is better).

IRANIAN: average precision changed from 0.755 to 0.875 (+0.121); Brier score changed from 0.071 to 0.049 (lower is better).

RETAIL: average precision changed from 0.708 to 0.709 (+0.000); Brier score changed from 0.213 to 0.214 (lower is better).

Development decision: prioritize XGBoost for further Iranian validation; retain Logistic Regression as a competitive IBM option and the retail reference model. Retail average precision is essentially unchanged, while its XGBoost Brier score is slightly worse. Prioritize temporal calibration and drift analysis for retail before expanding model complexity. These are provisional research decisions pending final testing.

These are development fold means, not final-test results or significance claims. See `Second_Progress_Report.html` for the full tables, limitations and explanation audit.

## Next two weeks

1. Training-only probability calibration, especially retail cohort forecasts.
2. Directional and subgroup explanation stability.
3. Freeze the final model protocol, then perform the reserved evaluation at the final stage.

## TA talking points

“We moved from fixed linear baselines to a nested nonlinear comparison on identical outer splits. We can now distinguish better risk ranking from better probability forecasts. Our final tests remain sealed; the next decision is whether calibration and explanation stability matter more than expanding the search.”
