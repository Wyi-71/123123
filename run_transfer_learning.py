import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transfer_loader import Transfer48kDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 动态添加上级目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(parent_dir, 'cwru_data'))
sys.path.append(os.path.join(parent_dir, '时空注意力机制加强版'))
sys.path.append(os.path.join(parent_dir, '双分支 CNN-LSTM 时频特征提取架构'))

from model import SimpleCNN
from resnet_model import BearingResNet18
from cbam_model import BearingResNet18_CBAM
from cnn_lstm_model import STFT_CNN_LSTM

# 配置区
TEST_DIR = os.path.join(parent_dir, '变采样率泛化测试', '48k_test_set')
BATCH_SIZE = 32
FINETUNE_EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.2  # 20% 拿来微调 (Few-Shot)， 80% 拿来考核

MODELS_CONFIG = {
    "1. SimpleCNN": {
        "class": lambda: SimpleCNN(num_classes=4),
        "path": os.path.join(parent_dir, 'cwru_data', 'cwru_cnn_model.pth'),
        "save_name": "finetuned_cnn.pth"
    },
    "2. ResNet18": {
        "class": lambda: BearingResNet18(num_classes=4, pretrained=False),
        "path": os.path.join(parent_dir, 'cwru_data', 'cwru_resnet_model.pth'),
        "save_name": "finetuned_resnet.pth"
    },
    "3. CBAM-ResNet18": {
        "class": lambda: BearingResNet18_CBAM(num_classes=4, pretrained=False),
        "path": os.path.join(parent_dir, '时空注意力机制加强版', 'cwru_resnet_cbam_model.pth'),
        "save_name": "finetuned_cbam_resnet.pth"
    },
    "4. CNN-LSTM": {
        "class": lambda: STFT_CNN_LSTM(num_classes=4, num_slices=8),
        "path": os.path.join(parent_dir, '双分支 CNN-LSTM 时频特征提取架构', 'cwru_cnn_lstm_model.pth'),
        "save_name": "finetuned_cnn_lstm.pth"
    }
}

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0: return 0.0, [], []
    return 100 * correct / total, [], []

def get_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

def finetune_model(model, train_loader, val_loader, device):
    """迁移学习微调函数：记录每个 Epoch 的指标"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    losses = []
    accs = []
    
    print(f"        ▶ 开始微调 {FINETUNE_EPOCHS} 个 Epoch...")
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        
        # 每个 Epoch 结束后在测试集上评估一次看进度
        acc, _, _ = test_model_simple(model, val_loader, device)
        accs.append(acc)
        print(f"          Epoch [{epoch+1}/{FINETUNE_EPOCHS}], Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")
        
    return model, losses, accs

def test_model_simple(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total), [], []

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" 🚀 域适应迁移学习 (Domain Adaptation) 与对比验证测试")
    print(f" 评估设备: {device}")
    print("=" * 60)

    if not os.path.exists(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
        print(f"⚠️ 找不到数据！请确保 {TEST_DIR} 下存放了 48k 数据集。")
        return

    print("📖 正在读取 48k 全量数据并划分为 [微调集/测试集]...")
    dataset = Transfer48kDataset(TEST_DIR)
    
    if len(dataset) == 0:
        print("❌ 未读到有效的 48k 数据。")
        return
        
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    print(f"🎯 成功划分! 总样本数: {len(dataset)} | 分配: {train_size} 个用于微调训练, {test_size} 个用于终极考核。")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    results_zero_shot = {}
    results_finetuned = {}

    for name, cfg in MODELS_CONFIG.items():
        print(f"\n=======================")
        print(f"[{name}] 处理流水线")
        
        if not os.path.exists(cfg['path']):
            print(f"  ❌ 原始 12k 权重没找到，跳过~")
            results_zero_shot[name] = 0.0
            results_finetuned[name] = 0.0
            continue
            
        # 1. 零样本裸测 (Zero-shot on test set)
        print("  1️⃣ [零样本测试] 测验原生抗偏移能力...")
        model = cfg['class']().to(device)
        model.load_state_dict(torch.load(cfg['path'], map_location=device))
        acc_zero, _, _ = test_model(model, test_loader, device)
        results_zero_shot[name] = acc_zero
        print(f"      ✅ 裸测准确率: {acc_zero:.2f}%")
        
        # 2. 迁移学习 (Few-shot Transfer Learning)
        print("  2️⃣ [迁移学习] 开始在 20% 的上海卷上突击培训老专家...")
        model_to_finetune = cfg['class']().to(device)
        model_to_finetune.load_state_dict(torch.load(cfg['path'], map_location=device))
        
        finetuned_model, losses, accs = finetune_model(model_to_finetune, train_loader, test_loader, device)
        
        # 绘制该模型的迁移学习曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, FINETUNE_EPOCHS+1), losses, marker='o')
        plt.title(f"{name} Finetune Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(range(1, FINETUNE_EPOCHS+1), accs, marker='s', color='orange')
        plt.title(f"{name} Finetune Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Acc (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f"transfer_curve_{name.split('. ')[1]}.png"))
        plt.close()

        # 保存微调后的权重
        save_path = os.path.join(script_dir, cfg['save_name'])
        torch.save(finetuned_model.state_dict(), save_path)
        print(f"      💾 微调权重已保存至: {cfg['save_name']}")
        
        # 3. 再打分并生成混淆矩阵
        print("  3️⃣ [复测考核] 培训结束，再次送上同一张试卷测验...")
        acc_fine, _, _ = test_model_simple(finetuned_model, test_loader, device)
        results_finetuned[name] = acc_fine
        
        # 生成混淆矩阵
        y_true, y_pred = get_predictions(finetuned_model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        classes = ['Normal', 'Ball', 'Inner', 'Outer']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix: {name} (After Transfer)")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f"transfer_cm_{name.split('. ')[1]}.png"))
        plt.close()
        
        print(f"      ✅ 迁移学习后准确率飙升至: {acc_fine:.2f}%  (提升了 {acc_fine - acc_zero:.2f}%)")

    # ================= 绘图环节 =================
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = [n.split('. ')[1] for n in results_zero_shot.keys()]
    x = np.arange(len(names))
    width = 0.35
    
    zero_accs = list(results_zero_shot.values())
    fine_accs = list(results_finetuned.values())
    
    bars1 = ax.bar(x - float(width)/2, zero_accs, width, label='Zero-Shot (Before Transfer)', color='#5e81ac', edgecolor='black')
    bars2 = ax.bar(x + float(width)/2, fine_accs, width, label='Few-Shot Transfer Learning (After)', color='#ebcb8b', edgecolor='black')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Domain Shift Overcoming: Zero-Shot vs Transfer Learning (12k to 48k)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold', fontsize=10)

    autolabel(bars1)
    autolabel(bars2)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(script_dir, 'transfer_learning_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 究极对比完毕！微调对比柱状图已生成至: {plot_path}")

if __name__ == '__main__':
    main()
