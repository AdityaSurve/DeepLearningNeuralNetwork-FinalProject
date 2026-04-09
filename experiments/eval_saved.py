import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train_common import get_device, evaluate_pytorch_model
from models.mlp import SimpleMLP
from models.custom_architecture import CustomTabularNet

def main():
    device = get_device()
    data = np.load('data/processed/data_ohe.npz')
    X_test, y_test = data['X_test'], data['y_test']
    
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    input_dim = X_test.shape[1]

    # Evaluate MLP
    mlp = SimpleMLP(input_dim=input_dim, hidden_layers=[256, 128, 64], dropout=0.3)
    mlp.load_state_dict(torch.load('checkpoints/mlp/best_model.pt', map_location=device))
    mlp.to(device)
    evaluate_pytorch_model(mlp, test_loader, 'outputs/mlp', 'MLP', device=device)

    # Evaluate Custom (match architecture in run_custom_architecture.py)
    custom = CustomTabularNet(
        input_dim=input_dim,
        n_tokens=10,
        d_model=320,
        n_heads=8,
        n_layers=5,
        dim_ff=1280,
        dropout=0.1,
        num_skip_blocks=4,
    )
    custom.load_state_dict(torch.load('checkpoints/custom_architecture/best_model.pt', map_location=device))
    custom.to(device)
    thr = 0.5
    metrics_path = 'outputs/custom_architecture/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding='utf-8') as f:
            thr = float(json.load(f).get('threshold', 0.5))
    evaluate_pytorch_model(
        custom, test_loader, 'outputs/custom_architecture', 'CustomArchitecture', device=device, threshold=thr
    )
    
if __name__ == '__main__':
    main()
