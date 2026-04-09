import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import SimpleMLP
from src.train_common import get_device, train_model, evaluate_pytorch_model, plot_learning_curves

def main():
    print("Loading OHE data for Standard MLP...")
    data = np.load('data/processed/data_ohe.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Calculate pos_weight for imbalanced classes
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    print(f"BCE pos_weight calculated as {pos_weight.item():.2f}")
    
    # Create DataLoaders
    batch_size = 1024
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    model = SimpleMLP(input_dim=input_dim, hidden_layers=[256, 128, 64], dropout=0.3)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    output_dir = 'outputs/mlp'
    os.makedirs('checkpoints/mlp', exist_ok=True)
    checkpoint_path = 'checkpoints/mlp/best_model.pt'
    
    print("Training MLP...")
    best_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=30, patience=5, checkpoint_path=checkpoint_path, device=device
    )
    
    # Evaluation
    metrics = evaluate_pytorch_model(best_model, test_loader, output_dir, "MLP", device=device)
    plot_learning_curves(history, output_dir)
    print(f"Test F1: {metrics['f1_score']:.4f}")
    
if __name__ == "__main__":
    main()
