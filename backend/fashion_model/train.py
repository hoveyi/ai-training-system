import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import struct
import numpy as np
from model import get_model


_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))  # 向上两级到项目根目录

class Config:
    data_dir = os.path.join(_project_root, 'dataset', 'cloths')
    num_classes = 10
    batch_size = 128
    epochs = 30
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = os.path.join(_script_dir, 'checkpoints', 'best_model.pth')


def read_idx(filename):
    """读取IDX格式文件"""
    with open(filename, 'rb') as f:
        zero, dtype, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data


class TransformDataset(torch.utils.data.Dataset):
    """对已有Dataset包装标准化和数据增强"""
    def __init__(self, dataset, mean, std, augment=False):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, [2])
        x = (x - self.mean) / self.std
        return x, y


def get_loaders():
    """从本地IDX文件加载Fashion-MNIST数据"""
    data_dir = Config.data_dir

    # 读取原始数据（文件嵌套在同名文件夹内）
    train_images = read_idx(os.path.join(data_dir, 'train-images-idx3-ubyte', 'train-images-idx3-ubyte'))
    train_labels = read_idx(os.path.join(data_dir, 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte'))
    test_images = read_idx(os.path.join(data_dir, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte'))
    test_labels = read_idx(os.path.join(data_dir, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte'))

    # 转为float并归一化到[0,1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # 标准化参数（Fashion-MNIST的经验值）
    mean = 0.2860
    std = 0.3530

    # 训练集增强：在numpy层面做简易增强
    # 不做numpy增强，转为在Dataset中加transform
    train_images_t = torch.tensor(train_images).unsqueeze(1)  # (N,1,28,28)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)

    # 划分验证集（前10%）
    val_size = int(0.1 * len(train_images_t))
    indices = torch.randperm(len(train_images_t), generator=torch.Generator().manual_seed(42))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = TensorDataset(train_images_t[train_indices], train_labels_t[train_indices])
    val_dataset = TensorDataset(train_images_t[val_indices], train_labels_t[val_indices])

    test_images_t = torch.tensor(test_images).unsqueeze(1)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)
    test_dataset = TensorDataset(test_images_t, test_labels_t)

    train_dataset = TransformDataset(train_dataset, mean, std, augment=True)
    val_dataset = TransformDataset(val_dataset, mean, std, augment=False)
    test_dataset = TransformDataset(test_dataset, mean, std, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


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
        correct += outputs.argmax(1).eq(y).sum().item()
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
        correct += outputs.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return running_loss / len(loader), correct / total


def train_model(model, train_loader, val_loader, config, model_name):
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_acc = 0
    best_state = None

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)
        scheduler.step()

        print(f"[{model_name}] Epoch {epoch+1}/{config.epochs}: "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_val_acc


def main():
    os.makedirs('./checkpoints', exist_ok=True)
    print(f"Device: {Config.device}")
    print("Loading Fashion-MNIST...")
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
                   '凉鞋', '衬衫', '运动鞋', '包', '踝靴']

    results = {}
    state_dicts = {}
    for model_name in ['cnn', 'resnet18']:
        print(f"\n--- Training {model_name} ---")
        model = get_model(model_name, Config.num_classes)
        val_acc = train_model(model, train_loader, val_loader, Config, model_name)
        state_dicts[model_name] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), Config.device)
        results[model_name] = test_acc
        print(f"[{model_name}] Test Accuracy: {test_acc:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest: {best_name} ({results[best_name]:.4f})")

    torch.save({
        'cnn_state_dict': state_dicts['cnn'],
        'resnet18_state_dict': state_dicts['resnet18'],
        'num_classes': Config.num_classes,
        'class_names': class_names,
        'test_accuracy_cnn': results['cnn'],
        'test_accuracy_resnet18': results['resnet18'],
    }, Config.model_save_path)
    print(f"Saved both models to {Config.model_save_path}")


if __name__ == '__main__':
    main()
