import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import get_model


class Config:
    epochs = 200
    batch_size = 32
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = './checkpoints/best_model.pth'
    early_stop_patience = 30


def load_titanic_data():
    """加载并预处理Titanic数据集"""
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
    except Exception:
        url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
        df = pd.read_csv(url)
        df.columns = [c.lower() for c in df.columns]

    # 选择特征
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'

    df = df[features + [target]].copy()

    # 编码
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # 填充缺失值
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['embarked'] = df['embarked'].fillna(2)

    # 添加衍生特征
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size', 'is_alone']

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    input_dim = X.shape[1]
    return train_loader, val_loader, test_loader, input_dim, scaler, feature_cols


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return running_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return running_loss / len(loader), correct / total


def train_model(model, train_loader, val_loader, config, model_name):
    model = model.to(config.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_acc = 0
    early_stop = 0

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"[{model_name}] Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= config.early_stop_patience:
                print(f"[{model_name}] Early stop at epoch {epoch+1}, best val acc={best_val_acc:.4f}")
                break

    return best_val_acc


def main():
    os.makedirs('./checkpoints', exist_ok=True)
    print(f"Device: {Config.device}")
    print("Loading Titanic dataset...")
    train_loader, val_loader, test_loader, input_dim, scaler, feature_cols = load_titanic_data()
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    results = {}
    state_dicts = {}
    for model_name in ['mlp', 'mlp_wide']:
        print(f"\n--- Training {model_name} ---")
        model = get_model(model_name, input_dim)
        val_acc = train_model(model, train_loader, val_loader, Config, model_name)
        state_dicts[model_name] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        test_loss, test_acc = evaluate(model, test_loader, nn.BCELoss(), Config.device)
        results[model_name] = test_acc
        print(f"[{model_name}] Test Accuracy: {test_acc:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest: {best_name} ({results[best_name]:.4f})")

    # 获取scaler参数
    _, _, _, _, scaler, _ = load_titanic_data()

    torch.save({
        'mlp_state_dict': state_dicts['mlp'],
        'mlp_wide_state_dict': state_dicts['mlp_wide'],
        'input_dim': input_dim,
        'feature_cols': feature_cols,
        'test_accuracy_mlp': results['mlp'],
        'test_accuracy_mlp_wide': results['mlp_wide'],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }, Config.model_save_path)
    print(f"Saved both models to {Config.model_save_path}")


if __name__ == '__main__':
    main()
