# End-to-end runbook (Heart Disease → train → results)

Single data source: **Heart Disease** from OpenML (via `sklearn.datasets.fetch_openml`).  
**No CSV in the repo** — first run downloads the dataset (then it is cached locally or under `~/scikit_learn_data` on Kaggle).

**Target:** binary **heart disease present** (`1`) vs absent (`0`).

---

## 1) Install dependencies

```powershell
pip install -r requirements.txt
```

(Includes **catboost**; use your course’s exact list if different.)

---

## 2) Repository root

```powershell
cd path\to\finalproject
```

---

## 3) Recommended command flow

Run **in order** from the project root:

```powershell
python setup_dirs.py
python experiments/data_discovery.py
python src/preprocess.py

# Classical baselines (ordinal unless noted)
python baselines/train_logistic_regression.py
python baselines/train_logistic_elasticnet.py
python baselines/train_random_forest.py
python baselines/train_xgboost.py
python baselines/train_lightgbm.py
python baselines/train_catboost.py
python baselines/train_hist_gradient_boosting.py

# Neural (one-hot)
python experiments/run_mlp.py
python experiments/run_tabular_transformer.py
python experiments/run_custom_architecture.py

# Optional: multi-model ensemble (requires trained artifacts)
python experiments/ensemble.py

python experiments/fairness_analysis.py
python experiments/compare_all_models.py
python reports/generate_report.py
```

`compare_all_models.py` scans **all** `outputs/<name>/metrics.json` paths it knows about (including every `custom_architecture_*` tag from `run_custom_architecture.py` and `ensemble_full` if you used that folder).

Or use the bundled runner (same steps except RF and `generate_report`):

```powershell
python main_runner.py
```

Then optionally:

```powershell
python reports/generate_report.py
```

---

## 4) Custom model tuning (optional env vars)

Set **before** `python experiments/run_custom_architecture.py` (PowerShell example):

Default **`hybrid`** uses **`balanced_acc`** for both the validation checkpoint and the probability threshold (tuned on val), which favors **both classes** and usually helps **minority-class recall**.

```powershell
$env:OBJECTIVE_MODE="hybrid"
$env:BIAS_MITIGATION="both"
$env:PROTECTED_ATTRS="sex"
$env:CUSTOM_HIDDEN="256"
$env:CUSTOM_BLOCKS="2"
python experiments/run_custom_architecture.py
```

To go back to the **accuracy vs balanced-accuracy blend** on validation instead:

```powershell
$env:VAL_METRIC="composite"
$env:THRESH_METRIC="composite"
$env:COMPOSITE_ALPHA="0.5"
python experiments/run_custom_architecture.py
```

---

## 5) Kaggle (git clone)

1. Turn **Internet** **on** (first OpenML download).
2. Clone repo and `cd` into `finalproject`.
3. Run the same flow as §3 with `!python ...` cells.

No dataset attachment is required; only network for the first OpenML fetch.

---

## 6) Outputs

- Processed arrays: `data/processed/data_ohe.npz`, `data_ord.npz`
- Raw splits for fairness: `data/processed/X_*_raw.csv`
- Metrics: `outputs/<model_name>/metrics.json`
- Checkpoints: `checkpoints/<model_name>/best_model.pt`

Custom runs use tags like `custom_architecture_hybrid_mit_both` depending on env.

---

## 7) Common issues

- **Download fails:** check internet / firewall; retry `python src/preprocess.py`.
- **Wrong metrics after changing code:** delete `data/processed/*` and rerun `preprocess.py`, then retrain.
