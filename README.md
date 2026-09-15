# Customer Retention — Weeks 1–4

**INFO 5920 · First Progress Report**  
Jiatong Xu (jx429) · Zhiming Zhang (zz939)

## Start here

Open `reports/First_Progress_Report.html` in a browser. It is a self-contained, English, instructor-facing report with embedded charts, concrete baseline results, methodological caveats, and the Weeks 5–8 plan. It has two parts: project rationale and proposal (research background, main research topic, and three-layer study design), followed by project progress: current stage, initial findings, the emerging forecasting question, and next steps. Development Q&A and technical evidence remain available in expandable sections. No Python or network access is needed to read it.

This package extends the earlier Week 1 foundation. It completes data documentation, development EDA, fold-safe preprocessing, and initial Logistic Regression/prior baselines. XGBoost, tuning, SHAP, and final-test evaluation remain later-stage work.

## Notebooks

| File | Role |
|---|---|
| 01_source_and_quality_audit.ipynb | Raw-source checks and quality audit |
| 02_targets_cohorts_and_split_plan.ipynb | Initial target definitions, fixed holdouts and first retail cohort |
| 03_cross_dataset_framework.ipynb | Initial three-layer framework and concept mapping |
| 04_development_eda.ipynb | Development-only exploratory figures and horizon sensitivity |
| 05_baseline_models_and_validation.ipynb | Preprocessing, five-fold telecom and forward retail baseline evaluation |
| 06_progress_report_evidence.ipynb | Metric reconciliation and final-holdout boundary checks |

Notebooks 01–03 describe the **Week 1 foundation stage**; their stage-specific statements that no models have yet been fitted refer to that point in the sequence. The current cumulative status is `outputs/weeks1_4_status.json`, after notebooks 04–06. The final HTML supersedes the earlier Week 1 report for presenting this milestone.

## Reproduce

Python 3.12 was used. From the extracted project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_notebooks.py
```

On Windows, activate with `.venv\Scripts\activate`. The runner executes standard Python cells and captures real stdout and Matplotlib outputs without a Jupyter kernel. The `.ipynb` files can also be run normally in Jupyter / VS Code (install `jupyterlab ipykernel` separately if needed). The Excel audit may take one to several minutes; allow at least 4 GB free memory.

Raw data are included. `src/download_data.py` can retrieve them again; the manifest verifies supplied bytes. Do not overwrite source files if exact reproducibility is required. `config/study_design.json` documents the frozen design. Primary baseline hyperparameters are deliberately fixed in `src/baseline_utils.py`; change the configuration and implementation together for a new experiment, documenting changes before evaluation.

The report is generated from saved tables by `build_report.py`. Run it after notebook execution to refresh the report and notebook HTML previews. The report does not claim final-test or intervention performance.

## Outputs

- `data/raw/`: original CSV/XLSX and IBM license.
- `data/processed/`: valid retail purchases, development features and temporal snapshots.
- `outputs/tables/`: fold assignments, EDA, per-fold metrics, validation predictions, coefficients and report summaries.
- `outputs/figures/`: report charts.
- `reports/methodology.md`: complete methodological choices and source links.
- `outputs/weeks1_4_validation_checks.json`: reconciled metrics and leakage-boundary checks.

IDs and label metadata are not model inputs. Saved model coefficients are descriptive diagnostics, not causal effects. Final telecom holdouts and the retail 2011-09-01 origin remain unused for final performance evaluation.
