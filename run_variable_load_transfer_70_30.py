import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, '变负载实验_mat')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'results_70_30')
LOADS = ['0HP', '1HP', '2HP', '3HP']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 动态添加路径以导入各模块
sys.path.append(os.path.join(BASE_DIR, 'cwru_data'))
sys.path.append(os.path.join(BASE_DIR, '时空注意力机制加强版'))
sys.path.append(os.path.join(BASE_DIR, '双分支 CNN-LSTM 时频特征提取架构'))

from dataset import CWRUDataset
from model import SimpleCNN
from resnet_model import BearingResNet18
from cbam_model import BearingResNet18_CBAM
from cnn_lstm_model import STFT_CNN_LSTM

# 训练配置
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS_SOURCE = 20  # 0HP 训练轮数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / (total + 1e-10)

def train_model(model, train_loader, test_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        test_acc = evaluate(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
        
    return model, best_acc

def main():
    results = []
    
    # 1. 加载 0HP 数据 (Source Domain)
    print("🚀 正在加载 0HP 原域数据进行基准训练(70%训练集, 30%测试集)...")
    source_path = os.path.join(DATA_ROOT, '0HP')
    source_dataset = CWRUDataset(source_path)
    
    train_size = int(0.7 * len(source_dataset))
    test_size = len(source_dataset) - train_size
    s_train, s_test = random_split(source_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    s_train_loader = DataLoader(s_train, batch_size=BATCH_SIZE, shuffle=True)
    s_test_loader = DataLoader(s_test, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 定位定义的四款模型
    models_config = {
        "SimpleCNN": lambda: SimpleCNN(num_classes=4),
        "ResNet18": lambda: BearingResNet18(num_classes=4, pretrained=False),
        "CBAM-ResNet18": lambda: BearingResNet18_CBAM(num_classes=4, pretrained=False),
        "CNN-LSTM": lambda: STFT_CNN_LSTM(num_classes=4)
    }
    
    expert_weights = {}
    
    # 3. 训练 & 在 0HP 上打分
    for name, model_fn in models_config.items():
        print(f"\n--- [阶段 1] 训练模型: {name} ---")
        model = model_fn().to(device)
        model, best_acc = train_model(model, s_train_loader, s_test_loader, EPOCHS_SOURCE)
        expert_weights[name] = model.state_dict()
        print(f"✅ {name} 在 0HP 上训练完成! 最佳准确率: {best_acc:.2f}%")
        results.append({'Model': name, 'Load': '0HP', 'Accuracy': best_acc})

    # 4. 在 1HP, 2HP, 3HP 上进行零样本 (Zero-Shot) 直接测试
    for load in ['1HP', '2HP', '3HP']:
        print(f"\n--- [阶段 2] 零样本测试负载: {load} ---")
        target_path = os.path.join(DATA_ROOT, load)
        target_dataset = CWRUDataset(target_path)
        target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        for name in models_config.keys():
            # 重新实例化并加载 0HP 专家权重
            model = models_config[name]().to(device)
            model.load_state_dict(expert_weights[name])
            
            # 直接在目标负载上测试 (不进行任何学习)
            acc = evaluate(model, target_loader)
            print(f"  » {name} -> {load} 零样本准确率: {acc:.2f}%")
            results.append({'Model': name, 'Load': load, 'Accuracy': acc})

    # 5. 汇总保存
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'variable_load_zero_shot_results.csv'), index=False)
    print(f"\n📊 结果已保存至 {OUTPUT_DIR}\\variable_load_zero_shot_results.csv")
    
    # 6. 绘制折线图对比
    plot_line_comparison(df)

def plot_line_comparison(df):
    plt.figure(figsize=(10, 6))
    
    loads = ['0HP', '1HP', '2HP', '3HP']
    models = df['Model'].unique()
    
    # 配色方案
    colors = ['#5e81ac', '#88c0d0', '#bf616a', '#d08770']
    markers = ['o', 's', '^', 'D']
    
    min_acc = df['Accuracy'].min()
    
    for i, model_name in enumerate(models):
        accs = []
        for load in loads:
            acc = df[(df['Model'] == model_name) & (df['Load'] == load)]['Accuracy'].values[0]
            accs.append(acc)
        
        plt.plot(loads, accs, label=model_name, marker=markers[i], color=colors[i], linewidth=2, markersize=8)

    plt.title('Zero-Shot Generalization Performance Contrast (Base: 0HP Train[70%], Best Acc)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Load Condition', fontsize=12)
    plt.ylabel('Best Accuracy (%)', fontsize=12)
    
    # 纵坐标不从 0 开始，以便体现出改进模型与基础模型的差距
    # 根据数据动态设置下限，通常比最低准确率低 5 个点即可
    plt.ylim(max(0, min_acc - 5), 101)
    
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(OUTPUT_DIR, 'variable_load_comparison_line.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📈 最终对比折线图已生成: {save_path}")

if __name__ == '__main__':
    main()
