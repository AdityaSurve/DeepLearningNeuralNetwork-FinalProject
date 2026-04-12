# Final deliverables guide (5922 report + poster + 4‑min video)

Use this after your last training run. **Regenerate artifacts** so PDFs/posters match reality:

```powershell
python experiments/fairness_analysis.py
python experiments/compare_all_models.py
python reports/generate_report.py
```

---

## 1. Final report (LaTeX, ≥7 pages + references)

**Course requirement:** PDF from LaTeX (e.g. Overleaf), not the auto-generated `reports/final_report.md` alone—that file is a **scratchpad** for numbers.

### Suggested section flow (maps to rubric)

| Section | What to put (from this repo) |
|--------|------------------------------|
| **Abstract** | Adult >50K; OpenML; classical vs **residual MLP** vs **tabular transformer**; **ensembles**; **subgroup metrics** (sex/race); **main finding**: boosting family ~0.92–0.93 AUROC, custom/transformer weaker on AUROC here, ensembles **trade** accuracy vs balanced accuracy. |
| **Introduction** | Why income prediction matters; risk of bias; gap = “DL always wins” is false on tabular; your contribution = **unified pipeline + systematic comparison + fairness view**. |
| **Related work** | Tabular DL (TabTransformer / FT-Transformer line); gradient boosting (XGB/LGB/CatBoost); fairness metrics & reweighting (Kamiran–Calders-style); 2–4 papers per theme in **prose** (not bullets). |
| **Methods** | `preprocess.py` (OHE vs ordinal); each model family; **CustomTabularNet** (blocks, threshold sweep, optional `BIAS_MITIGATION`); **TabTransformerLite**; **ensemble.py** (soft/hard, `ENSEMBLE_MEMBERS`); metrics in `src/metrics.py`. |
| **Experiments** | **Primary metric:** state one (e.g. ROC AUC). Table from `outputs/comparisons/master_results.csv` (or top‑K). Figures: `model_comparison_auc_top15.png`, `custom_vs_baseline.png`, fairness plots under `outputs/fairness/`. **Trends:** trees strongest AUROC; equal‑weight 7‑model ensemble can **lower** AUROC vs small ensemble; threshold metric changes behavior. **Limitations paragraph:** single split, no multi-seed CI, elastic-net search costly, etc. |
| **Conclusions** | One paragraph takeaway. **Ethics:** historical bias in Adult; models not “fair” because you printed subgroup tables; deployment risks; transparency (linear vs ensemble vs neural). |
| **Bibliography** | `.bib` with dataset + boosting + tabular DL + fairness references. |

### Figures worth exporting for LaTeX

- `outputs/comparisons/model_comparison_auc_top15.png`
- `outputs/comparisons/custom_vs_baseline.png`
- `outputs/comparisons/model_comparison_recall.png` (optional)
- `outputs/fairness/sex_recall_comparison.png`, `race_recall_comparison.png` (if present)
- One **ROC** or **confusion matrix** from your best run (`outputs/xgboost/` or chosen model)

---

## 2. Poster (PDF)

**Audience:** someone who has **not** taken the course (rubric).

**Layout (columns):**

1. **Problem:** Predict income >50K from census features; decisions affect people; bias matters.  
2. **Approach:** Many models + **two neural designs** (residual MLP + transformer-lite) + **ensembles**; same data split and metrics.  
3. **Key result (pick 1–3 numbers from your CSV):** e.g. best AUROC ≈ **0.93** (boosting); custom/transformer **lower** on AUROC; ensemble **mixed**.  
4. **Fairness:** small table or one figure—**gaps** between groups, not a claim of “solved.”  
5. **Takeaway:** **Classical boosting still wins on ranking here**; DL is **worth studying** but not automatically best; **ensembling is not free lunch**.

Use **`model_comparison_auc_top15.png`** and **`custom_vs_baseline.png`** large and legible.

---

## 3. Four-minute video (script ≈ 550–650 words spoken)

**Structure (seconds):**

| 0:00–0:25 | Hook: “Banks and agencies use models like this—who gets a loan or aid. We asked: **do neural nets beat simple strong baselines?**” |
| 0:25–0:55 | Data: Adult, OpenML, binary >50K; many models (trees, boosting, linear, **two neural nets**). |
| 0:55–2:00 | Methods in plain English: “We built a pipeline… residual network for tables… small transformer… combined models into an **ensemble**.” Show **one** diagram or table screenshot. |
| 2:00–3:15 | Results: **show `custom_vs_baseline` or top‑15 bar chart.** “Best models are **gradient boosted trees** around **0.93** AUROC; our custom deep nets were **lower** on this benchmark; **averaging everyone** sometimes **hurt** accuracy but changed **balanced** metrics.” |
| 3:15–3:45 | Fairness: “Performance **varies** by sex and race—we report it.” |
| 3:45–4:00 | Punchline + ethics: “**Takeaway:** for this tabular task, **smart classical models still lead**; deep learning is **not magic**; **always check** who the model fails for.” |

**Delivery:** slow, clear; avoid jargon (“AUROC” once, then “ranking quality”).

---

## 4. Peer evaluation day

Complete all assigned reviews; unrelated to repo quality.

---

## 5. Checklist before submit

- [ ] `compare_all_models.py` run → `leaderboard.csv` includes **hybrid_mit_both** (or whatever you actually trained) and **ensemble** / **ensemble_full** if used  
- [ ] `fairness_analysis.py` run → `outputs/fairness/*.png` exist  
- [ ] Report PDF **≥7 pages** + references  
- [ ] Poster PDF + video URL (YouTube/Vimeo, permissions OK) in Canvas text file  
- [ ] Spell-check names: **CatBoost**, **OpenML**, **AUROC** / **ROC AUC** (pick one spelling)

---

*This guide is project-specific; rubric on Canvas is authoritative if anything conflicts.*
