# End-to-end Runbook (Dataset → Train → Results → GitHub Submission)

This document is a **complete step-by-step** guide to run the project from scratch, either:

- **Locally on Windows**, or
- In a hosted environment (**Kaggle Notebook** recommended; Colab also works),

and then bring the produced results back into your local repo at:

- `c:\Users\Aditya\Documents\Masters\DL NN\DeepLearningNeuralNetwork\finalproject`

so you can commit/push to GitHub for submission.

---

## 0) What this project expects (important)

- **Dataset file**: `dataset/heart_2020_cleaned.csv`
- **Preprocessing script**: `src/preprocess.py`
  - Produces:
    - `data/processed/data_ohe.npz` (used by custom architecture / MLP / LR)
    - `data/processed/data_ord.npz` (used by some other models)
- **Training scripts**:
  - Custom architecture: `experiments/run_custom_architecture.py`
  - MLP baseline: `experiments/run_mlp.py`
  - Baselines: `baselines/train_*.py`
- **Results**:
  - Metrics JSON is written under `outputs/<model_name>/metrics.json`
  - Learning curves PNGs under `outputs/<model_name>/`
  - Model checkpoints under `checkpoints/<model_name>/`

---

## 1) Local (Windows) setup

### 1.1 Create a virtual environment (recommended)

From **PowerShell**:

```powershell
Set-Location "c:\Users\Aditya\Documents\Masters\DL NN\DeepLearningNeuralNetwork\finalproject"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 1.2 Install dependencies

If you already have a working environment, skip this.
Otherwise install the core packages used by the repo:

```powershell
pip install numpy pandas scikit-learn torch matplotlib joblib
```

If you plan to run tree baselines:

```powershell
pip install xgboost lightgbm
```

### 1.3 Put the dataset in the expected place

Ensure this file exists:

- `finalproject/dataset/heart_2020_cleaned.csv`

Full path on your machine:

- `c:\Users\Aditya\Documents\Masters\DL NN\DeepLearningNeuralNetwork\finalproject\dataset\heart_2020_cleaned.csv`

---

## 2) Run the pipeline locally (preprocess → train → evaluate)

### 2.1 Preprocess (creates `data/processed/*.npz`)

```powershell
Set-Location "c:\Users\Aditya\Documents\Masters\DL NN\DeepLearningNeuralNetwork\finalproject"
python src\preprocess.py
```

After it finishes, confirm these exist:

- `data/processed/data_ohe.npz`
- `data/processed/data_ord.npz`

### 2.2 Train the custom architecture

```powershell
python experiments\run_custom_architecture.py
```

Tuning knobs (optional, environment variables):

```powershell
# default: use balanced sampling
$env:BALANCED_SAMPLER="1"
# default: use pos_weight in BCE
$env:USE_POS_WEIGHT="1"
# default: EMA on
$env:USE_EMA="1"

# what validation metric to checkpoint on (recommended):
$env:VAL_METRIC="auprc"

# which threshold metric is used to pick "best threshold" for reporting:
$env:THRESH_METRIC="balanced_acc"

python experiments\run_custom_architecture.py
```

### 2.3 Where to find the local results

Key output files:

- `outputs/custom_architecture/metrics.json`
- `outputs/custom_architecture/loss_curve.png`
- `outputs/custom_architecture/auroc_curve.png`
- (and possibly) `outputs/custom_architecture/val_accuracy_curve.png`

Checkpoint:

- `checkpoints/custom_architecture/best_model.pt`

---

## 3) Kaggle Notebook (recommended hosted option)

This is the best path if you want GPU/compute without dealing with cloud VM setup.

### 3.1 Upload dataset to Kaggle

You have two common options:

- **Option A (fastest)**: upload the CSV as a Kaggle Dataset
  - Kaggle → **Datasets** → **New Dataset**
  - Upload `heart_2020_cleaned.csv`
  - Keep it private or public as needed
- **Option B**: upload directly into the Notebook session
  - Works, but is less reproducible than attaching a Dataset

### 3.2 Create a Kaggle Notebook and attach the dataset

- Kaggle → **Code** → **New Notebook**
- In the right sidebar:
  - **Accelerator**: choose **GPU** (optional but recommended)
  - **Add data**: attach your dataset that contains `heart_2020_cleaned.csv`

### 3.3 Get your code into Kaggle

Pick one:

- **Option A (recommended)**: clone your GitHub repo inside the notebook
  - In a Kaggle cell:

```bash
!git clone <YOUR_GITHUB_REPO_URL>
%cd <YOUR_REPO_FOLDER>/finalproject
```

- **Option B**: upload the whole `finalproject/` folder as a Kaggle Dataset
  - Kaggle → **Datasets** → **New Dataset** → upload your code folder
  - Then attach that “code dataset” to the notebook and `cd` into it

### 3.4 Put the dataset CSV in the expected path

Kaggle mounts datasets under `/kaggle/input/...`.
We need:

- `finalproject/dataset/heart_2020_cleaned.csv`

So in a Kaggle cell (adjust the input path to match your dataset name):

```bash
!mkdir -p dataset
!cp "/kaggle/input/<YOUR_DATASET_FOLDER>/heart_2020_cleaned.csv" "dataset/heart_2020_cleaned.csv"
!ls -lah dataset
```

### 3.5 Install dependencies in Kaggle

Most are preinstalled. If needed:

```bash
!pip -q install numpy pandas scikit-learn torch matplotlib joblib
```

### 3.6 Run preprocessing and training in Kaggle

```bash
!python src/preprocess.py
!python experiments/run_custom_architecture.py
```

Optional env vars (same as local):

```bash
%env BALANCED_SAMPLER=1
%env USE_POS_WEIGHT=1
%env USE_EMA=1
%env VAL_METRIC=auprc
%env THRESH_METRIC=balanced_acc

!python experiments/run_custom_architecture.py
```

### 3.7 Export results from Kaggle back to your computer

Kaggle persists anything written under the notebook working directory.
To download results:

- In the Kaggle UI: **Output** panel → download files

Recommended: zip the results into one file first:

```bash
!zip -r outputs_and_checkpoints.zip outputs checkpoints
```

Then download `outputs_and_checkpoints.zip` from Kaggle.

### 3.8 Put results into your local repo

On your Windows machine:

- Extract the zip
- Copy:
  - `outputs/` → into `finalproject/outputs/`
  - `checkpoints/` → into `finalproject/checkpoints/` (optional for submission unless required)

So you end up with:

- `c:\Users\Aditya\Documents\Masters\DL NN\DeepLearningNeuralNetwork\finalproject\outputs\custom_architecture\metrics.json`

---

## 4) Google Colab (alternative hosted option)

### 4.1 Create a Colab notebook

- Go to Google Colab → New Notebook
- Runtime → Change runtime type → **GPU** (recommended)

### 4.2 Get your code into Colab

In a cell:

```bash
!git clone <YOUR_GITHUB_REPO_URL>
%cd <YOUR_REPO_FOLDER>/finalproject
```

### 4.3 Upload dataset to Colab

Two options:

- **Option A**: upload directly in Colab session (temporary)
- **Option B (recommended)**: put it in Google Drive and mount Drive

Drive workflow:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then copy the CSV into the expected location:

```bash
!mkdir -p dataset
!cp "/content/drive/MyDrive/<PATH_TO>/heart_2020_cleaned.csv" "dataset/heart_2020_cleaned.csv"
```

### 4.4 Install dependencies and run

```bash
!pip -q install numpy pandas scikit-learn torch matplotlib joblib
!python src/preprocess.py
!python experiments/run_custom_architecture.py
```

### 4.5 Download results back to your computer

Zip and download:

```bash
!zip -r outputs_and_checkpoints.zip outputs checkpoints
```

Then use Colab’s file browser to download the zip, and copy results into your local `finalproject/` as described in the Kaggle section.

---

## 5) “Link it”: how notebooks and your repo connect

The simplest robust pattern is:

- Notebook **clones your GitHub repo**
- Notebook **copies the dataset CSV** into `finalproject/dataset/`
- You run the same scripts as locally
- You **zip `outputs/`** and download it back to your machine
- You copy results into the local repo and push to GitHub

This keeps the notebook as a “compute runner”, while your repo stays the source of truth.

---

## 6) Generate results you can submit

After training, confirm you have the key artifacts locally:

- `outputs/custom_architecture/metrics.json`
- (optional but good) plots under `outputs/custom_architecture/`

Then commit/push:

```powershell
Set-Location "c:\Users\Aditya\Documents\Masters\DL NN\DeepLearningNeuralNetwork"
git status
git add finalproject/outputs finalproject/reports finalproject/experiments finalproject/models finalproject/src
git commit -m "Update custom architecture results"
git push
```

Notes:
- If your course does **not** want large files, **do not commit** `checkpoints/` unless required.
- If `outputs/` is large, commit only `metrics.json` and key plots.

---

## 7) Common problems (quick fixes)

### 7.1 Accuracy looks “stuck” around ~0.91

That’s the all-negative baseline because positives are ~9%.
Use AUPRC/AUROC/BAcc/F1 to judge improvements (the code now prints these).

### 7.2 Kaggle/Colab can’t find the dataset

Make sure this file exists before preprocessing:

- `finalproject/dataset/heart_2020_cleaned.csv`

### 7.3 CPU training is slow locally

Use Kaggle GPU or Colab GPU, or reduce `BATCH_SIZE` and/or model size.

