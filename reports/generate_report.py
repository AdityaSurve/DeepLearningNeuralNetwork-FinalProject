import os
import pandas as pd

def generate_report():
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/final_report.md'
    
    leaderboard_path = 'outputs/comparisons/leaderboard.csv'
    if os.path.exists(leaderboard_path):
        lb = pd.read_csv(leaderboard_path)
        lb_md = lb.to_markdown(index=False)
    else:
        lb_md = "Leaderboard not found."
        
    fair_path = 'outputs/fairness/fairness_comparison_table.csv'
    attr_summary = "sex"
    if os.path.exists(fair_path):
        fair = pd.read_csv(fair_path)
        if "Attribute" in fair.columns and not fair.empty:
            attrs = sorted(set(fair["Attribute"].astype(str)))
            attr_summary = ", ".join(attrs)
        # Simplify table for display
        fair_md = fair.groupby(['Attribute', 'Model']).mean(numeric_only=True).reset_index().to_markdown(index=False)
    else:
        fair_md = "Fairness analysis not found."
        
    md_content = f"""# Final project report: Heart Disease Detection (binary classification)

## 1. Project objective
Tabular deep learning vs classical baselines on **Heart Disease** (OpenML): predict **disease presence (target=1)**, with overall metrics and subgroup-style breakdowns by protected/clinical attributes.

## 2. Dataset
Data are loaded via `sklearn.datasets.fetch_openml("heart-disease", version=1)` in `src/preprocess.py` (no bundled CSV). Numeric and categorical columns are imputed, scaled (numeric), and encoded (one-hot / ordinal) before modeling.

## 3. Results & leaderboard
{lb_md}

## 4. Deep learning vs baselines
The custom model is a compact **residual MLP** (`models/custom_architecture.py`), trained with validation-driven thresholding and optional instance reweighing (`experiments/run_custom_architecture.py`). Tree baselines use `data/processed/data_ord.npz`.

**Ensemble** (`experiments/ensemble.py`): soft- or hard-votes multiple models (trees, elastic net, tabular transformer, custom net—whatever has `metrics.json` / checkpoints). Default output `outputs/ensemble/`; use `ENSEMBLE_OUTPUT_DIR=outputs/ensemble_full` for a separate full run. Regenerate **`python experiments/compare_all_models.py`** after training so `leaderboard.csv` lists every model folder.

**Figures:** `outputs/comparisons/model_comparison_auc.png` (all runs), `model_comparison_auc_top15.png` (poster-friendly), `custom_vs_baseline.png` (best classical vs best deep vs best ensemble).

![Best classical vs deep vs ensemble](/outputs/comparisons/custom_vs_baseline.png)

## 5. Fairness analysis
Metrics by **{attr_summary}** on the held-out test split:
{fair_md}

## 6. Conclusion
End-to-end pipeline: fetch heart-disease data → preprocess → train baselines and custom PyTorch model → optional tree+DL ensemble → compare → subgroup metrics.
"""
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_content)
    
    print(f"Generated {report_path}")

if __name__ == "__main__":
    generate_report()
