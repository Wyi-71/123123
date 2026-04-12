import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """
    轻量级 2D CNN，用于提取 STFT 切片（长条图）的空间频谱特征
    输入尺寸: (Batch, 1, 64, 8) -> 高度=频率=64, 宽度=时间步长=8
    """
    def __init__(self, output_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # (16, 32, 4)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # (32, 16, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))  # (64, 8, 1)

        # 展平后的维度: 64 * 8 * 1 = 512
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class STFT_CNN_LSTM(nn.Module):
    """
    时频特征提取专用的 CNN-LSTM 架构 (2D CNN + 时序网络)
    将 64x64 的完整 STFT 图像沿时间轴切片，通过 CNN 提取特征后送入 LSTM
    """
    def __init__(self, num_classes=4, num_slices=8, cnn_out_dim=128, lstm_hidden=128, lstm_layers=2):
        super(STFT_CNN_LSTM, self).__init__()
        
        self.num_slices = num_slices
        # 64 x 64 切成 num_slices 份，每份宽度为 64 // num_slices
        self.slice_width = 64 // num_slices 
        
        # 1. 空间特征提取器 (CNN)
        self.cnn_extractor = CNNFeatureExtractor(output_dim=cnn_out_dim)
        
        # 2. 时序记忆抽取器 (LSTM)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 1, 64, 64) -> B=Batch Size, 1=Channel, H=Freq, W=Time
        B = x.size(0)
        
        # 1. 沿时间轴 (宽度) 进行切片
        # 将 64 宽度的图切分为 num_slices 个宽为 slice_width 的小长条
        # x_slices 是一个张量元组，包含 num_slices 个 (B, 1, 64, 8)
        x_slices = torch.split(x, self.slice_width, dim=3)
        
        # 2. 拼接成批次一起送入 CNN 进行特征提取
        # 拼接后 shape: (B * num_slices, 1, 64, slice_width)
        x_cat = torch.cat(x_slices, dim=0) 
        
        # CNN 提特征: (B * num_slices, cnn_out_dim)
        cnn_features = self.cnn_extractor(x_cat)
        
        # 3. 变形回序列形式供 LSTM 使用
        # shape: (num_slices, B, cnn_out_dim) -> (B, num_slices, cnn_out_dim)
        cnn_features = cnn_features.view(self.num_slices, B, -1).transpose(0, 1)
        
        # 4. LSTM 时序建模
        # lstm_out shape: (B, num_slices, lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # 取 LSTM 最后一个时间步的输出作为整段信号的压缩表征
        # final_feat shape: (B, lstm_hidden)
        final_feat = lstm_out[:, -1, :]
        
        # 5. 分类输出
        out = self.classifier(final_feat)
        return out


if __name__ == '__main__':
    model = STFT_CNN_LSTM(num_classes=4, num_slices=8)
    print("CNN-LSTM 模型初始化成功！")
    
    # 测试输入参数：(Batch_Size, Channels, Freq_Bins, Time_Frames)
    input_tensor = torch.randn(4, 1, 64, 64)
    output = model(input_tensor)
    
    print(f"输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")

    # 可以看出网络如何剥洋葱一样处理时空特征
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数量: {total_params:,}")
