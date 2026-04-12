"""
BearingResNet18 — 基于预训练 ResNet18 的轴承故障诊断模型
适配单通道 64×64 STFT 灰度图输入
"""
import torch
import torch.nn as nn
from torchvision import models


class BearingResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(BearingResNet18, self).__init__()

        # 加载预训练 ResNet18
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = models.resnet18(weights=weights)
            except AttributeError:
                self.backbone = models.resnet18(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=False)

        # 修改第一层卷积：3通道 → 1通道
        # 保留预训练权重：取 3 通道权重的均值作为单通道初始化
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )

        # 替换最后全连接层
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    model = BearingResNet18(num_classes=4)
    print(model)
    # 测试输入：单通道 64×64
    input_tensor = torch.randn(2, 1, 64, 64)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
