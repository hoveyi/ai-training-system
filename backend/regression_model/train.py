import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from model import get_model


class Config:
    n_samples = 5000
    epochs = 200
    batch_size = 64
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = './checkpoints/best_model.pth'
    early_stop_patience = 40
    seq_len = 10


def generate_nonlinear_data(n_samples=5000):
    """生成多输入多输出非线性系统数据：Lorenz-like混沌映射"""
    np.random.seed(42)
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    x3 = np.random.uniform(-3, 3, n_samples)

    # 复杂非线性函数，包含正弦、指数和高阶项
    y1 = (np.sin(x1 * 1.5) * np.cos(x2 * 0.8) +
          x3 ** 2 * 0.3 +
          np.exp(-np.abs(x2)) * 0.5 +
          np.sin(x1 * x2 * 0.3) * 0.4)
    y2 = (np.cos(x3 * 1.2) * np.sin(x1 * 0.7) +
          x2 ** 3 * 0.1 +
          np.tanh(x1 + x2 + x3) * 0.6)
    y3 = (np.sin(x1 + x2) * np.cos(x3) * 0.8 +
          np.abs(x1 * x3) * 0.15 +
          x2 ** 2 * 0.2 +
          np.cos(x1 * x2 * x3 * 0.1) * 0.3)

    noise = 0.05
    y1 += np.random.normal(0, noise, n_samples)
    y2 += np.random.normal(0, noise, n_samples)
    y3 += np.random.normal(0, noise, n_samples)

    X = np.stack([x1, x2, x3], axis=1).astype(np.float32)
    Y = np.stack([y1, y2, y3], axis=1).astype(np.float32)
    return X, Y


def generate_sequence_data(n_samples=5000, seq_len=10):
    """生成时序非线性数据：基于Lorenz-like序列"""
    n_total = n_samples + seq_len
    t = np.linspace(0, 50, n_total)
    x1 = np.sin(t * 0.7) + 0.5 * np.sin(t * 1.5) + np.random.normal(0, 0.05, n_total)
    x2 = np.cos(t * 0.6) + 0.3 * np.sin(t * 2.0) + np.random.normal(0, 0.05, n_total)
    x3 = np.sin(t * 0.9 + 1.0) * np.cos(t * 0.3) + np.random.normal(0, 0.05, n_total)

    target = (np.sin(t * 0.8) * np.cos(t * 0.5) +
              np.exp(-0.1 * t) * np.sin(t * 1.2) +
              0.2 * np.sin(t * 2.1) * np.cos(t * 0.4))

    X_seq = np.stack([x1, x2, x3], axis=1).astype(np.float32)
    Y = target.astype(np.float32)

    # 构建滑动窗口
    X_windows = []
    Y_windows = []
    for i in range(n_samples):
        X_windows.append(X_seq[i:i + seq_len])
        Y_windows.append(Y[i + seq_len])

    return np.array(X_windows), np.array(Y_windows)


def train_mlp():
    print("Generating nonlinear data...")
    X, Y = generate_nonlinear_data(Config.n_samples)

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X = scaler_X.fit_transform(X)
    Y = scaler_Y.fit_transform(Y)

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model = get_model('mlp', input_dim=3, output_dim=3).to(Config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    early_stop = 0

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(Config.device), y.to(Config.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.device), y.to(Config.device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= Config.early_stop_patience:
                print(f"[MLP] Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # 测试 R²
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(Config.device)
            preds.append(model(x).cpu().numpy())
            truths.append(y.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    r2 = r2_score(truths, preds, multioutput='uniform_average')
    print(f"[MLP] Test R²: {r2:.4f}")

    return model, r2, scaler_X, scaler_Y


def train_lstm():
    print("Generating sequence data...")
    X, Y = generate_sequence_data(Config.n_samples, Config.seq_len)

    # 标准化（per feature across all samples and timesteps）
    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    scaler_X = StandardScaler().fit(X_flat)
    X = scaler_X.transform(X_flat).reshape(n_samples, seq_len, n_features)
    scaler_Y = StandardScaler().fit(Y.reshape(-1, 1))
    Y = scaler_Y.transform(Y.reshape(-1, 1)).flatten()

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model = get_model('lstm', input_dim=3, output_dim=1).to(Config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    early_stop = 0

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(Config.device), y.to(Config.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.device), y.to(Config.device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= Config.early_stop_patience:
                print(f"[LSTM] Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(Config.device)
            preds.append(model(x).cpu().numpy())
            truths.append(y.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    r2 = r2_score(truths, preds)
    print(f"[LSTM] Test R²: {r2:.4f}")

    return model, r2, scaler_X, scaler_Y


def main():
    os.makedirs('./checkpoints', exist_ok=True)
    print(f"Device: {Config.device}")

    print("\n=== Training MLP Regressor ===")
    mlp_model, mlp_r2, scaler_X_mlp, scaler_Y_mlp = train_mlp()

    print("\n=== Training LSTM Regressor ===")
    lstm_model, lstm_r2, scaler_X_lstm, scaler_Y_lstm = train_lstm()

    print(f"\nMLP R²: {mlp_r2:.4f}, LSTM R²: {lstm_r2:.4f}")

    torch.save({
        'mlp_state_dict': {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()},
        'lstm_state_dict': {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()},
        'r2_mlp': mlp_r2,
        'r2_lstm': lstm_r2,
        'scaler_X_mlp': scaler_X_mlp,
        'scaler_Y_mlp': scaler_Y_mlp,
        'scaler_X_lstm': scaler_X_lstm,
        'scaler_Y_lstm': scaler_Y_lstm,
    }, Config.model_save_path)
    print(f"Saved both models to {Config.model_save_path}")


if __name__ == '__main__':
    main()
