import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """
    Custom Dataset for tabular data.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IntrusionDetectionModel(nn.Module):
    """
    MLP architecture for the model of intrusion detection.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_layers: list, dropout_rate: float):
        super(IntrusionDetectionModel, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)