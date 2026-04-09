import os

dirs = [
    'data/raw', 'data/processed', 'data/metadata', 
    'src', 'baselines', 'models', 'experiments', 
    'outputs/logistic_regression', 'outputs/random_forest', 
    'outputs/xgboost', 'outputs/lightgbm', 'outputs/mlp', 
    'outputs/residual_mlp', 'outputs/tab_transformer', 
    'outputs/ft_transformer', 'outputs/tabnet', 
    'outputs/custom_architecture', 'outputs/comparisons', 
    'outputs/fairness', 'outputs/interpretability', 
    'checkpoints/mlp', 'checkpoints/residual_mlp', 
    'checkpoints/tab_transformer', 'checkpoints/ft_transformer', 
    'checkpoints/tabnet', 'checkpoints/custom_architecture', 
    'reports/tables', 'reports/figures/feature_distributions', 
    'reports/logs', 'reports/summary'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    
print("Directories created successfully!")
