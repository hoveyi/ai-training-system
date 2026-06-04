import torch
import torch.nn as nn


class TitanicMLP(nn.Module):
    """MLP-深度型：多层+BN+Dropout，适合特征交互复杂的表格数据"""
    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TitanicMLP_Wide(nn.Module):
    """MLP-宽型：更宽的隐藏层，依靠宽度而非深度来拟合"""
    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def get_model(model_name='mlp', input_dim=9):
    if model_name == 'mlp':
        return TitanicMLP(input_dim)
    elif model_name == 'mlp_wide':
        return TitanicMLP_Wide(input_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
