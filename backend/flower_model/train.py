import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import get_model, SimpleCNN


# 配置
class Config:
    data_dir = '../../dataset/flowers'  # 数据集路径
    num_classes = 5
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = './checkpoints/best_model.pth'
    early_stop_patience = 10


def get_data_transforms():
    """数据预处理和增强"""

    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证集和测试集只做基本变换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class TransformedDataset(torch.utils.data.Dataset):
    """在已有Dataset基础上应用独立transform，各子集互不干扰"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)  # 确保是普通列表
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_data(config):
    """加载数据集"""
    train_transform, val_transform = get_data_transforms()

    # 加载完整数据集（不带 transform，由各子集独立应用）
    full_dataset = datasets.ImageFolder(root=config.data_dir)

    # 划分训练集、验证集、测试集 (70% train, 15% val, 15% test)
    total = len(full_dataset)
    indices = list(range(total))
    # 用sklearn打乱并划分
    from sklearn.model_selection import train_test_split
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=full_dataset.targets)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                         stratify=[full_dataset.targets[i] for i in temp_idx])

    # 各子集独立使用自己的 transform，互不干扰
    train_dataset = TransformedDataset(full_dataset, train_idx, train_transform)
    val_dataset = TransformedDataset(full_dataset, val_idx, val_transform)
    test_dataset = TransformedDataset(full_dataset, test_idx, val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0)

    class_names = full_dataset.classes
    return train_loader, val_loader, test_loader, class_names


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def train(config):
    """主训练函数"""
    print(f"Using device: {config.device}")

    # 加载数据
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = load_data(config)
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # 创建模型
    model = get_model('resnet50', config.num_classes)
    model = model.to(config.device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # 训练记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    early_stop_counter = 0

    # 创建保存目录
    os.makedirs('./checkpoints', exist_ok=True)

    print("Starting training...")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, config.device)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, config.model_save_path)
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 早停
        if early_stop_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 测试最佳模型
    print("\nTesting best model...")
    checkpoint = torch.load(config.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, config.device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    return model, class_names


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')

    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('./checkpoints/training_curves.png')
    plt.show()


if __name__ == '__main__':
    config = Config()

    # 训练 ResNet50
    print("=" * 50)
    print("Training ResNet50...")
    print("=" * 50)
    model_resnet, class_names = train(config)

    # 修改配置以训练 SimpleCNN（更多epoch，较小学习率）
    config.learning_rate = 0.001
    config.epochs = 80
    config.early_stop_patience = 15
    # 临时修改model_save_path
    original_path = config.model_save_path

    # 训练 SimpleCNN
    print("\n" + "=" * 50)
    print("Training SimpleCNN...")
    print("=" * 50)

    # 手动train SimpleCNN
    train_loader, val_loader, test_loader, class_names = load_data(config)
    cnn_model = SimpleCNN(num_classes=config.num_classes)
    cnn_model = cnn_model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_acc = 0
    best_cnn_state = None
    early_stop_counter = 0

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(cnn_model, train_loader, criterion, optimizer, config.device)
        val_loss, val_acc = validate(cnn_model, val_loader, criterion, config.device)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[SimpleCNN] Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cnn_state = {k: v.cpu().clone() for k, v in cnn_model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stop_patience:
                print(f"[SimpleCNN] Early stop at epoch {epoch+1}")
                break

    cnn_model.load_state_dict(best_cnn_state)
    test_loss, test_acc = validate(cnn_model, test_loader, criterion, config.device)
    print(f"[SimpleCNN] Test Accuracy: {test_acc:.2f}%")

    # 加载ResNet的best checkpoint
    resnet_ckpt = torch.load(original_path)
    resnet_acc = resnet_ckpt['val_acc']

    # 保存合并的checkpoint
    torch.save({
        'resnet50_state_dict': resnet_ckpt['model_state_dict'],
        'simple_cnn_state_dict': best_cnn_state,
        'class_names': class_names,
        'test_accuracy_resnet50': resnet_acc,
        'test_accuracy_simple_cnn': test_acc,
    }, original_path)
    print(f"\nSaved both models to {original_path}")
    print(f"ResNet50 Acc: {resnet_acc:.2f}%, SimpleCNN Acc: {test_acc:.2f}%")