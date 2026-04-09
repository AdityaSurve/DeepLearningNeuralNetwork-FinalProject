import os
import subprocess

def run_script(path):
    print(f"Running {path}...")
    subprocess.run(["python", path], check=True)

def main():
    # 1. Setup and Discovery
    # (Assuming Directories are set up)
    
    # run_script("setup_dirs.py")
    run_script("experiments/data_discovery.py")
    run_script("src/preprocess.py")
    
    # 2. Baselines
    run_script("baselines/train_logistic_regression.py")
    run_script("baselines/train_xgboost.py")
    run_script("baselines/train_lightgbm.py")
    # Random Forest takes long, skipping in generic runner unless desired
    
    # 3. DL Models
    run_script("experiments/run_mlp.py")
    run_script("experiments/run_custom_architecture.py")
    
    # 4. Evaluation
    run_script("experiments/fairness_analysis.py")
    run_script("experiments/compare_all_models.py")
    
    print("Full pipeline completed! Check reports/ and outputs/.")

if __name__ == "__main__":
    main()
