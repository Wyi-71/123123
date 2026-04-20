import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# 动态添加目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'cwru_data'))
sys.path.append(os.path.join(script_dir, '变采样率泛化测试'))
sys.path.append(os.path.join(script_dir, '自适应迁移学习_48k'))

from model import SimpleCNN
from resnet_model import BearingResNet18
from dataset import CWRUDataset
from transfer_loader import Transfer48kDataset

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR_12k = os.path.join(script_dir, 'cwru_data')
DATA_DIR_48k = os.path.join(script_dir, '变采样率泛化测试', '48k_test_set')
BATCH_SIZE = 32

def get_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # 1. 准备 12k 测试集 (通常是原训练集的 20% 测试部分)
    # 注意: 这里的 12k 数据加载逻辑需与训练时一致
    full_12k = CWRUDataset(DATA_DIR_12k)
    _, test_12k = random_split(full_12k, [int(0.8*len(full_12k)), len(full_12k)-int(0.8*len(full_12k))], 
                               generator=torch.Generator().manual_seed(42))
    loader_12k = DataLoader(test_12k, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 准备 48k 测试集 (迁移学习时的 80% 考核部分)
    full_48k = Transfer48kDataset(DATA_DIR_48k)
    _, test_48k = random_split(full_48k, [int(0.2*len(full_48k)), len(full_48k)-int(0.2*len(full_48k))], 
                               generator=torch.Generator().manual_seed(42))
    loader_48k = DataLoader(test_48k, batch_size=BATCH_SIZE, shuffle=False)
    
    models = {
        "SimpleCNN": {
            "class": SimpleCNN,
            "path_12k": os.path.join(script_dir, 'cwru_data', 'cwru_cnn_model.pth'),
            "path_48k": os.path.join(script_dir, '自适应迁移学习_48k', 'finetuned_cnn.pth')
        },
        "ResNet18": {
            "class": lambda: BearingResNet18(num_classes=4, pretrained=False),
            "path_12k": os.path.join(script_dir, 'cwru_data', 'cwru_resnet_model.pth'),
            "path_48k": os.path.join(script_dir, '自适应迁移学习_48k', 'finetuned_resnet.pth')
        }
    }
    
    results = {}
    for name, cfg in models.items():
        # 评估 12k
        m12 = cfg['class']().to(DEVICE)
        m12.load_state_dict(torch.load(cfg['path_12k'], map_location=DEVICE))
        acc12 = get_acc(m12, loader_12k)
        
        # 评估 48k Transfer
        m48 = cfg['class']().to(DEVICE)
        m48.load_state_dict(torch.load(cfg['path_48k'], map_location=DEVICE))
        acc48 = get_acc(m48, loader_48k)
        
        results[name] = (acc12, acc48)
    
    # 绘图
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.35
    
    acc12_vals = [v[0] for v in results.values()]
    acc48_vals = [v[1] for v in results.values()]
    
    rects1 = ax.bar(x - width/2, acc12_vals, width, label='12k Test Accuracy', color='#3498db')
    rects2 = ax.bar(x + width/2, acc48_vals, width, label='48k Transfer Accuracy', color='#e67e22')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Performance Comparison: 12k Baseline vs 48k Transfer')
    ax.set_xticks(x)
    ax.set_xticklabels(results.keys())
    ax.set_ylim(80, 105) # 重点关注高精度区域
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('comparison_12k_vs_48k.png', dpi=150)
    print("柱状图已生成: comparison_12k_vs_48k.png")

if __name__ == "__main__":
    main()
