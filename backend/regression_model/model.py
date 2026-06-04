import torch
import torch.nn as nn


class RegressionMLP(nn.Module):
    """MLP回归器：适合多维非线性映射"""
    def __init__(self, input_dim=3, output_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class RegressionLSTM(nn.Module):
    """LSTM回归器：适合时序非线性动态系统预测"""
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def get_model(model_name='mlp', input_dim=3, output_dim=1):
    if model_name == 'mlp':
        return RegressionMLP(input_dim, output_dim)
    elif model_name == 'lstm':
        return RegressionLSTM(input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
