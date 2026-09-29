# Customer Retention — Weeks 1–6

## Latest milestone: Weeks 5–6

Open **[Second Progress Report](reports/Second_Progress_Report.html)** for the TA check-in, or read the concise [progress notes](reports/Weeks_5_6_Progress.md).

The **[second check-in slide deck](reports/Second_Check_In_Report.pptx)** presents current progress, validation methods, editable result charts, model explanations, and the Weeks 7–8 plan in eight slides.

The new milestone adds XGBoost with nested training-only selection, matched baseline reruns, held-out TreeSHAP diagnostics, probability-quality and capacity comparisons, and independent result validation. Final test sets remain reserved. The earlier report and notebooks below are historical Weeks 1–4 artifacts.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS only, if OpenMP is missing: brew install libomp
python run_weeks5_6.py
python -m unittest discover -s tests
```

The new runner uses the supplied development snapshots and frozen fold assignments; it does not require re-executing the six earlier notebooks. The experiment design is in `config/weeks5_6_experiment.json`; all new evidence is in `outputs/weeks5_6/`. `validate_weeks5_6.py` independently checks predictions and metrics; `build_weeks5_6_report.py` regenerates the report from validated saved results. `run_metadata.json` records the executed environment and SHA-256 input hashes. `FILE_MANIFEST.json` remains the original Weeks 1–4 package manifest, not a manifest of this new milestone.

## Historical milestone: Weeks 1–4

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
