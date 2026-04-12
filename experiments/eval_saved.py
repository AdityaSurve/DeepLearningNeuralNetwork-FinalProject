import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

    # Evaluate Custom (same hyperparameters as run_custom_architecture.py defaults)
    h = int(os.environ.get("CUSTOM_HIDDEN", "256"))
    nb = int(os.environ.get("CUSTOM_BLOCKS", "2"))
    do = float(os.environ.get("CUSTOM_DROPOUT", "0.2"))
    custom = CustomTabularNet(
        input_dim=input_dim, hidden_dim=h, num_blocks=nb, dropout=do
    )
    ckpt = os.environ.get("CUSTOM_CHECKPOINT", "")
    candidates = [
        p
        for p in (
            ckpt,
            "checkpoints/custom_architecture_hybrid/best_model.pt",
            "checkpoints/custom_architecture_hybrid_mit_both/best_model.pt",
            "checkpoints/custom_architecture_hybrid_mit_class/best_model.pt",
            "checkpoints/custom_architecture/best_model.pt",
        )
        if p
    ]
    ckpt_path = next((p for p in candidates if os.path.isfile(p)), None)
    if ckpt_path is None:
        print("Skipping custom eval: no checkpoint found; set CUSTOM_CHECKPOINT.")
    else:
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        custom.load_state_dict(state)
        custom.to(device)
        tag = Path(ckpt_path).parent.name
        out_dir = f"outputs/{tag}"
        thr = 0.5
        metrics_path = os.path.join(out_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                thr = float(json.load(f).get("threshold", 0.5))
        evaluate_pytorch_model(
            custom,
            test_loader,
            out_dir,
            "CustomArchitecture",
            device=device,
            threshold=thr,
        )
    
if __name__ == '__main__':
    main()
