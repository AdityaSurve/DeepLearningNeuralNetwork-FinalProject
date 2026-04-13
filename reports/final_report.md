# Final project report: Heart Disease Detection (binary classification)

## 1. Project objective
Tabular deep learning vs classical baselines on **Heart Disease** (OpenML): predict **disease presence (target=1)**, with overall metrics and subgroup-style breakdowns by protected/clinical attributes.

## 2. Dataset
Data are loaded via `sklearn.datasets.fetch_openml("heart-disease", version=1)` in `src/preprocess.py` (no bundled CSV). Numeric and categorical columns are imputed, scaled (numeric), and encoded (one-hot / ordinal) before modeling.

## 3. Results & leaderboard
| Model                                    |   roc_auc |   f1_score |   recall |   accuracy |
|:-----------------------------------------|----------:|-----------:|---------:|-----------:|
| lightgbm                                 |  0.899048 |   0.862069 |     1    |   0.826087 |
| catboost                                 |  0.891429 |   0.836364 |     0.92 |   0.804348 |
| ensemble_full                            |  0.878095 |   0.711111 |     0.64 |   0.717391 |
| ensemble                                 |  0.878095 |   0.711111 |     0.64 |   0.717391 |
| logistic_elasticnet                      |  0.859048 |   0.823529 |     0.84 |   0.804348 |
| logistic_regression                      |  0.857143 |   0.823529 |     0.84 |   0.804348 |
| random_forest                            |  0.855238 |   0.77193  |     0.88 |   0.717391 |
| hist_gradient_boosting                   |  0.847619 |   0.8      |     0.88 |   0.76087  |
| xgboost                                  |  0.847619 |   0.8      |     0.88 |   0.76087  |
| custom_architecture_balanced             |  0.79619  |   0.77551  |     0.76 |   0.76087  |
| custom_architecture_balanced_mit_both    |  0.761905 |   0.652174 |     0.6  |   0.652174 |
| custom_architecture_hybrid_mit_both      |  0.742857 |   0.526316 |     0.4  |   0.608696 |
| custom_architecture_accuracy             |  0.714286 |   0.590909 |     0.52 |   0.608696 |
| custom_architecture_balanced_mit_class   |  0.706667 |   0.666667 |     0.64 |   0.652174 |
| custom_architecture_hybrid               |  0.672381 |   0.585366 |     0.48 |   0.630435 |
| custom_architecture_balanced_mit_reweigh |  0.653333 |   0.604651 |     0.52 |   0.630435 |
| custom_architecture_hybrid_mit_class     |  0.571429 |   0.721311 |     0.88 |   0.630435 |
| custom_architecture_accuracy_mit_reweigh |  0.521905 |   0.646154 |     0.84 |   0.5      |
| tabular_transformer                      |  0.457143 |   0.489796 |     0.48 |   0.456522 |
| custom_architecture_accuracy_mit_class   |  0.409524 |   0.704225 |     1    |   0.543478 |
| mlp                                      |  0.361905 |   0.686567 |     0.92 |   0.543478 |
| custom_architecture_accuracy_mit_both    |  0.327619 |   0.666667 |     0.92 |   0.5      |
| custom_architecture_hybrid_mit_reweigh   |  0.314286 |   0.704225 |     1    |   0.543478 |

## 4. Deep learning vs baselines
The custom model is a compact **residual MLP** (`models/custom_architecture.py`), trained with validation-driven thresholding and optional instance reweighing (`experiments/run_custom_architecture.py`). Tree baselines use `data/processed/data_ord.npz`.

**Ensemble** (`experiments/ensemble.py`): soft- or hard-votes multiple models (trees, elastic net, tabular transformer, custom net—whatever has `metrics.json` / checkpoints). Default output `outputs/ensemble/`; use `ENSEMBLE_OUTPUT_DIR=outputs/ensemble_full` for a separate full run. Regenerate **`python experiments/compare_all_models.py`** after training so `leaderboard.csv` lists every model folder.

**Figures:** `outputs/comparisons/model_comparison_auc.png` (all runs), `model_comparison_auc_top15.png` (poster-friendly), `custom_vs_baseline.png` (best classical vs best deep vs best ensemble).

![Best classical vs deep vs ensemble](/outputs/comparisons/custom_vs_baseline.png)

## 5. Fairness analysis
Metrics by **sex** on the held-out test split:
| Attribute   | Model                                    |   Accuracy |       F1 |   Recall |   Precision |   ROC_AUC |   Count |   Group |
|:------------|:-----------------------------------------|-----------:|---------:|---------:|------------:|----------:|--------:|--------:|
| sex         | catboost                                 |   0.835417 | 0.838624 | 0.916667 |    0.776786 |  0.882835 |      23 |     0.5 |
| sex         | custom_architecture_accuracy             |   0.583333 | 0.591667 | 0.522436 |    0.720238 |  0.666489 |      23 |     0.5 |
| sex         | custom_architecture_accuracy_mit_both    |   0.572917 | 0.698276 | 0.916667 |    0.584821 |  0.255698 |      23 |     0.5 |
| sex         | custom_architecture_accuracy_mit_class   |   0.60625  | 0.73399  | 1        |    0.60625  |  0.404202 |      23 |     0.5 |
| sex         | custom_architecture_accuracy_mit_reweigh |   0.558333 | 0.679487 | 0.839744 |    0.608262 |  0.61485  |      23 |     0.5 |
| sex         | custom_architecture_balanced             |   0.6125   | 0.613636 | 0.557692 |    0.694444 |  0.754986 |      23 |     0.5 |
| sex         | custom_architecture_balanced_mit_both    |   0.660417 | 0.638095 | 0.592949 |    0.694444 |  0.73166  |      23 |     0.5 |
| sex         | custom_architecture_balanced_mit_class   |   0.7375   | 0.711538 | 0.711538 |    0.711538 |  0.73896  |      23 |     0.5 |
| sex         | custom_architecture_balanced_mit_reweigh |   0.614583 | 0.597826 | 0.516026 |    0.7125   |  0.594551 |      23 |     0.5 |
| sex         | custom_architecture_hybrid               |   0.483333 | 0.469697 | 0.432692 |    0.513636 |  0.468127 |      23 |     0.5 |
| sex         | custom_architecture_hybrid_mit_both      |   0.627083 | 0.457971 | 0.387821 |    0.616667 |  0.722934 |      23 |     0.5 |
| sex         | custom_architecture_hybrid_mit_class     |   0.670833 | 0.753759 | 0.958333 |    0.644872 |  0.590278 |      23 |     0.5 |
| sex         | custom_architecture_hybrid_mit_reweigh   |   0.60625  | 0.73399  | 1        |    0.60625  |  0.362358 |      23 |     0.5 |
| sex         | ensemble                                 |   0.725    | 0.702381 | 0.634615 |    0.787879 |  0.860755 |      23 |     0.5 |
| sex         | ensemble_full                            |   0.725    | 0.702381 | 0.634615 |    0.787879 |  0.860755 |      23 |     0.5 |
| sex         | hist_gradient_boosting                   |   0.7875   | 0.806366 | 0.878205 |    0.755656 |  0.84099  |      23 |     0.5 |
| sex         | lightgbm                                 |   0.852083 | 0.868578 | 1        |    0.780075 |  0.907229 |      23 |     0.5 |
| sex         | logistic_elasticnet                      |   0.820833 | 0.821538 | 0.836538 |    0.807692 |  0.835114 |      23 |     0.5 |
| sex         | logistic_regression                      |   0.820833 | 0.821538 | 0.836538 |    0.807692 |  0.832799 |      23 |     0.5 |
| sex         | mlp                                      |   0.577083 | 0.707407 | 0.923077 |    0.607143 |  0.385684 |      23 |     0.5 |
| sex         | random_forest                            |   0.739583 | 0.769704 | 0.875    |    0.6875   |  0.838675 |      23 |     0.5 |
| sex         | tabular_transformer                      |   0.452083 | 0.496491 | 0.483974 |    0.611111 |  0.621795 |      23 |     0.5 |
| sex         | xgboost                                  |   0.7875   | 0.806366 | 0.878205 |    0.755656 |  0.846866 |      23 |     0.5 |

## 6. Conclusion
End-to-end pipeline: fetch heart-disease data → preprocess → train baselines and custom PyTorch model → optional tree+DL ensemble → compare → subgroup metrics.
