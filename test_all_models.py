import os
import sys
import torch
from torch.utils.data import DataLoader
from test_dataset import Test48kDataset
import matplotlib.pyplot as plt

# 动态添加上级目录，方便直接导入所有的四个模型
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, 'cwru_data'))
sys.path.append(os.path.join(parent_dir, '时空注意力机制加强版'))
sys.path.append(os.path.join(parent_dir, '双分支 CNN-LSTM 时频特征提取架构'))

from model import SimpleCNN
from resnet_model import BearingResNet18
from cbam_model import BearingResNet18_CBAM
from cnn_lstm_model import STFT_CNN_LSTM

# 配置区
TEST_DIR = os.path.join(script_dir, '48k_test_set')
BATCH_SIZE = 32

# 四个模型的路径字典
MODELS_CONFIG = {
    "1. SimpleCNN": {
        "class": lambda: SimpleCNN(num_classes=4),
        "path": os.path.join(parent_dir, 'cwru_data', 'cwru_cnn_model.pth')
    },
    "2. ResNet18": {
        "class": lambda: BearingResNet18(num_classes=4, pretrained=False),
        "path": os.path.join(parent_dir, 'cwru_data', 'cwru_resnet_model.pth')
    },
    "3. CBAM-ResNet18": {
        "class": lambda: BearingResNet18_CBAM(num_classes=4, pretrained=False),
        "path": os.path.join(parent_dir, '时空注意力机制加强版', 'cwru_resnet_cbam_model.pth')
    },
    "4. CNN-LSTM": {
        "class": lambda: STFT_CNN_LSTM(num_classes=4, num_slices=8),
        "path": os.path.join(parent_dir, '双分支 CNN-LSTM 时频特征提取架构', 'cwru_cnn_lstm_model.pth')
    }
}

def evaluate_model(model_name, config, dataloader, device):
    print(f"\n[{model_name}] 正在评估...")
    if not os.path.exists(config['path']):
        print(f"  ❌ 找不到权重文件: {config['path']} (请先运行对应训练脚本)")
        return 0.0
    
    try:
        model = config['class']().to(device)
        model.load_state_dict(torch.load(config['path'], map_location=device))
        model.eval()
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
        
    acc = 100 * correct / total
    print(f"  ✅ 48k 跨域泛化测试准确率: {acc:.2f}% ({correct}/{total})")
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print(" 🚀 跨域泛化能力终极测试 (48k -> 12k Equivalency)")
    print(f" 评估设备: {device}")
    print("=" * 50)

    # 1. 检查测试集目录
    if not os.path.exists(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
        print(f"⚠️ 找不到测试数据！\n请将包含 48k 采样率的四个类别的文件夹放入:\n{TEST_DIR}\n(确保含有 Normal, Inner, Outer, Ball 的文件数据)")
        return

    print("📖 正在加载兵构建强降采样测试卷...")
    test_dataset = Test48kDataset(TEST_DIR)
    
    if len(test_dataset) == 0:
        print("❌ 测试集为空，未能读取到有效的 48k `.mat` 文件。")
        return
        
    print(f"🎯 成功构建有效测试样本: {len(test_dataset)} 个。")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 依次测试四个模型
    results = {}
    for name, cfg in MODELS_CONFIG.items():
        acc = evaluate_model(name, cfg, test_loader, device)
        results[name] = acc

    # 3. 绘制炫酷性能对比图
    if all(v == 0.0 for v in results.values()):
        print("\n所有的模型都没有跑通，未生成图表。")
        return
        
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    names = [n.split('. ')[1] for n in results.keys()]
    accs = list(results.values())
    
    colors = ['#5e81ac', '#88c0d0', '#ebcb8b', '#bf616a']  # 北极星主题色
    bars = plt.bar(names, accs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)
    
    plt.title('Cross-Domain Generalization Test (Train: 12k -> Test: 48k)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加网格辅助线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'generalization_benchmark.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 跨域测试完毕，对比柱状图已生成至: {plot_path}")

if __name__ == '__main__':
    main()
