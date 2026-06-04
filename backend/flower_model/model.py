import torch
import torch.nn as nn
import torchvision.models as models


class FlowerClassifier(nn.Module):
    """花卉分类模型 - 基于ResNet50的迁移学习"""

    def __init__(self, num_classes=5, pretrained=True):
        super(FlowerClassifier, self).__init__()

        # 使用预训练的ResNet50作为backbone
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)

        # 获取特征提取层
        in_features = self.backbone.fc.in_features

        # 替换最后的全连接层，适应5分类任务
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # 添加特征金字塔用于多尺度特征提取
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        # 提取浅层特征（边缘、纹理等）
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # 提取中层特征（形状、部分等）
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        # 提取深层特征（语义信息）
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # 全局平均池化
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # 分类
        x = self.backbone.fc(x)

        return x


class SimpleCNN(nn.Module):
    """简化的CNN模型，适合从零训练"""

    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积：提取浅层特征（边缘、颜色）
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 第二层卷积：提取中层特征（纹理、形状）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 第三层卷积：提取深层特征（语义信息）
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 第四层卷积：更深层特征
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name='resnet50', num_classes=5):
    """获取模型"""
    if model_name == 'resnet50':
        return FlowerClassifier(num_classes=num_classes, pretrained=True)
    elif model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")