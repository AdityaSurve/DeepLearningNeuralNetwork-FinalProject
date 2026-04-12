# Final project report: tabular deep learning on Adult Census Income

## 1. Objective

Compare a custom PyTorch tabular model and standard baselines on predicting **whether income exceeds 50K**, with evaluation on overall metrics and demographic subgroup behavior (**sex**, **race**).

## 2. Dataset

**Adult Census Income** (OpenML version 2): demographic and employment attributes; binary target derived from the `class` column (`>50K` vs `<=50K`). Features are numeric and categorical; preprocessing uses imputation, scaling for numeric columns, and one-hot / ordinal encoding via `src/preprocess.py`.

## 3. Results

Run `python experiments/compare_all_models.py` after training to refresh `outputs/comparisons/leaderboard.csv`, then regenerate this section or use `reports/generate_report.py`.

## 4. Models

- Baselines: logistic regression, gradient-boosted trees (e.g. XGBoost, LightGBM), MLP.
- **Custom model:** residual MLP (`models/custom_architecture.py`), trained with the hybrid objective and optional bias mitigation in `experiments/run_custom_architecture.py`.

## 5. Fairness

Subgroup metrics use `data/processed/X_test_raw.csv` and predictions under `outputs/<model>/predictions.csv`; see `outputs/fairness/` after `experiments/fairness_analysis.py`.

## 6. Conclusion

Pipeline: OpenML fetch → preprocess → train → evaluate → fairness tables, without requiring a local CSV copy of the Adult dataset.
