import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], dropout=0.3):
        super(SimpleMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1)) # Output is logits
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
