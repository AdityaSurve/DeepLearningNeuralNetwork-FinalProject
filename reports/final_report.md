# Final project report: Adult Census Income (>50K prediction)

## 1. Project objective
Tabular deep learning vs classical baselines on **Adult Census Income** (OpenML): predict **income >50K**, with overall metrics and fairness-style breakdowns by **sex** and **race**.

## 2. Dataset
Data are loaded via `sklearn.datasets.fetch_openml` in `src/preprocess.py` (no bundled CSV). Numeric and categorical columns are imputed, scaled (numeric), and encoded (one-hot / ordinal) before modeling.

## 3. Results & leaderboard
| Model                               |   roc_auc |   f1_score |   recall |   accuracy |
|:------------------------------------|----------:|-----------:|---------:|-----------:|
| ensemble                            |  0.927259 |   0.717748 | 0.836758 |   0.842465 |
| hist_gradient_boosting              |  0.926978 |   0.709573 | 0.877854 |   0.827982 |
| xgboost                             |  0.926967 |   0.711008 | 0.875571 |   0.829622 |
| lightgbm                            |  0.926905 |   0.708571 | 0.884703 |   0.825796 |
| catboost                            |  0.926061 |   0.70941  | 0.877854 |   0.827845 |
| ensemble_full                       |  0.925963 |   0.709752 | 0.880708 |   0.827572 |
| logistic_elasticnet                 |  0.90202  |   0.673584 | 0.852169 |   0.802295 |
| logistic_regression                 |  0.902019 |   0.672977 | 0.852169 |   0.801749 |
| mlp                                 |  0.900834 |   0.681393 | 0.748288 |   0.832491 |
| custom_architecture_hybrid_mit_both |  0.889759 |   0.664497 | 0.75742  |   0.816915 |
| tabular_transformer                 |  0.792853 |   0.546917 | 0.582192 |   0.769094 |

## 4. Deep learning vs baselines
The custom model is a compact **residual MLP** (`models/custom_architecture.py`), trained with validation-driven thresholding and optional instance reweighing (`experiments/run_custom_architecture.py`). Tree baselines use `data/processed/data_ord.npz`.

**Ensemble** (`experiments/ensemble.py`): soft- or hard-votes multiple models (trees, elastic net, tabular transformer, custom net—whatever has `metrics.json` / checkpoints). Default output `outputs/ensemble/`; use `ENSEMBLE_OUTPUT_DIR=outputs/ensemble_full` for a separate full run. Regenerate **`python experiments/compare_all_models.py`** after training so `leaderboard.csv` lists every model folder.

**Figures:** `outputs/comparisons/model_comparison_auc.png` (all runs), `model_comparison_auc_top15.png` (poster-friendly), `custom_vs_baseline.png` (best classical vs best deep vs best ensemble).

![Best classical vs deep vs ensemble](/outputs/comparisons/custom_vs_baseline.png)

## 5. Fairness analysis
Metrics by **sex** and **race** on the held-out test split:
| Attribute   | Model                               |   Accuracy |       F1 |   Recall |   Precision |   ROC_AUC |   Count |
|:------------|:------------------------------------|-----------:|---------:|---------:|------------:|----------:|--------:|
| race        | catboost                            |   0.858955 | 0.698066 | 0.849818 |    0.593853 |  0.919849 |  1463.8 |
| race        | custom_architecture_hybrid_mit_both |   0.832098 | 0.597217 | 0.669172 |    0.54132  |  0.89149  |  1463.8 |
| race        | ensemble                            |   0.872435 | 0.687849 | 0.748096 |    0.668987 |  0.923197 |  1463.8 |
| race        | ensemble_full                       |   0.857647 | 0.688921 | 0.822256 |    0.59654  |  0.921717 |  1463.8 |
| race        | hist_gradient_boosting              |   0.855326 | 0.685125 | 0.818935 |    0.592701 |  0.922268 |  1463.8 |
| race        | lightgbm                            |   0.8506   | 0.667534 | 0.806171 |    0.573015 |  0.920814 |  1463.8 |
| race        | logistic_elasticnet                 |   0.83859  | 0.638187 | 0.753728 |    0.556004 |  0.900885 |  1463.8 |
| race        | logistic_regression                 |   0.837803 | 0.638625 | 0.756166 |    0.555425 |  0.901224 |  1463.8 |
| race        | mlp                                 |   0.85348  | 0.617018 | 0.623326 |    0.616568 |  0.89289  |  1463.8 |
| race        | tabular_transformer                 |   0.771108 | 0.458542 | 0.519318 |    0.428476 |  0.779785 |  1463.8 |
| race        | xgboost                             |   0.861709 | 0.686327 | 0.796334 |    0.61198  |  0.923649 |  1463.8 |
| sex         | catboost                            |   0.849161 | 0.680011 | 0.82001  |    0.582001 |  0.92395  |  3659.5 |
| sex         | custom_architecture_hybrid_mit_both |   0.838059 | 0.633433 | 0.723586 |    0.563265 |  0.890936 |  3659.5 |
| sex         | ensemble                            |   0.863437 | 0.695584 | 0.781375 |    0.629754 |  0.924906 |  3659.5 |
| sex         | ensemble_full                       |   0.849274 | 0.683115 | 0.826537 |    0.583182 |  0.923507 |  3659.5 |
| sex         | hist_gradient_boosting              |   0.850528 | 0.687138 | 0.824866 |    0.590374 |  0.924304 |  3659.5 |
| sex         | lightgbm                            |   0.848055 | 0.682847 | 0.828877 |    0.581892 |  0.924802 |  3659.5 |
| sex         | logistic_elasticnet                 |   0.826945 | 0.643066 | 0.795256 |    0.540564 |  0.900516 |  3659.5 |
| sex         | logistic_regression                 |   0.826537 | 0.641554 | 0.792018 |    0.540134 |  0.900567 |  3659.5 |
| sex         | mlp                                 |   0.85621  | 0.652783 | 0.669671 |    0.651591 |  0.899378 |  3659.5 |
| sex         | tabular_transformer                 |   0.782685 | 0.481524 | 0.551366 |    0.438303 |  0.773931 |  3659.5 |
| sex         | xgboost                             |   0.851329 | 0.684582 | 0.818673 |    0.589751 |  0.924203 |  3659.5 |

## 6. Conclusion
End-to-end pipeline: fetch Adult → preprocess → train baselines and custom PyTorch model → optional tree+DL ensemble → compare → subgroup metrics.
