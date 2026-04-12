import torch
import torch.nn as nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """时空注意力模块"""
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力：关注哪些频率带/隐藏通道最含有特征
        out = x * self.ca(x)
        # 空间注意力：关注 STFT 图像上哪些特定时间-频率块（亮点）是共振冲击
        out = out * self.sa(out)
        return out


class BearingResNet18_CBAM(nn.Module):
    """
    集成了 CBAM 时空注意力机制的 ResNet18
    接收 64x64 单通道时频图输入
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(BearingResNet18_CBAM, self).__init__()

        # 加载基础 ResNet18
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                base_model = models.resnet18(weights=weights)
            except AttributeError:
                base_model = models.resnet18(pretrained=True)
        else:
            base_model = models.resnet18(pretrained=False)

        # 1. 修改第一层适配单通道（灰度时频图）
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(base_model.conv1.weight.mean(dim=1, keepdim=True))
        
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # 2. 提取特征层并在每个大阶段后插入 CBAM 模块
        self.layer1 = base_model.layer1
        self.cbam1 = CBAM(64)
        
        self.layer2 = base_model.layer2
        self.cbam2 = CBAM(128)
        
        self.layer3 = base_model.layer3
        self.cbam3 = CBAM(256)
        
        self.layer4 = base_model.layer4
        self.cbam4 = CBAM(512)

        # 3. 分类头
        self.avgpool = base_model.avgpool
        in_features = base_model.fc.in_features  # 512
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers + CBAM Attention
        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = BearingResNet18_CBAM(num_classes=4)
    # 测试输入：单通道 64×64
    input_tensor = torch.randn(2, 1, 64, 64)
    output = model(input_tensor)
    print("模型测试成功！")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
