import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt

# ================= 动态添加系统路径 =================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(parent_dir, 'cwru_data'))
sys.path.append(os.path.join(parent_dir, '时空注意力机制加强版'))
sys.path.append(os.path.join(parent_dir, '双分支 CNN-LSTM 时频特征提取架构'))
sys.path.append(os.path.join(parent_dir, '自适应迁移学习_48k')) 

from model import SimpleCNN
from resnet_model import BearingResNet18
from cbam_model import BearingResNet18_CBAM
from cnn_lstm_model import STFT_CNN_LSTM

# ================= 测试配置 =================
TEST_DIR = os.path.join(parent_dir, '变采样率泛化测试', '48k_test_set')
BATCH_SIZE = 32
TRAIN_RATIO = 0.2

# 信噪比跑分设定 (按照要求：范围10-20dB，步长2dB)
# 这里保留了 None 作为 Clean 纯净信号基线对比
SNR_LIST = [None, 20, 18, 16, 14, 12, 10]

MODEL_WEIGHTS_DIR = os.path.join(parent_dir, '自适应迁移学习_48k')
MODELS_CONFIG = {
    "SimpleCNN": {
        "class": lambda: SimpleCNN(num_classes=4),
        "finetuned_path": os.path.join(MODEL_WEIGHTS_DIR, "finetuned_cnn.pth")
    },
    "ResNet18": {
        "class": lambda: BearingResNet18(num_classes=4, pretrained=False),
        "finetuned_path": os.path.join(MODEL_WEIGHTS_DIR, "finetuned_resnet.pth")
    },
    "CBAM-ResNet18": {
        "class": lambda: BearingResNet18_CBAM(num_classes=4, pretrained=False),
        "finetuned_path": os.path.join(MODEL_WEIGHTS_DIR, "finetuned_cbam_resnet.pth")
    },
    "CNN-LSTM": {
        "class": lambda: STFT_CNN_LSTM(num_classes=4, num_slices=8),
        "finetuned_path": os.path.join(MODEL_WEIGHTS_DIR, "finetuned_cnn_lstm.pth")
    }
}

# ================= 辅助函数类 =================
def add_awgn_noise(signal, snr_db):
    """为1D信号注入给定SNR的高斯白噪声"""
    if snr_db is None:
        return signal
    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return signal
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

def get_label_from_path(file_path):
    if 'Normal' in file_path: return 0
    elif 'Inner Race' in file_path: return 1
    elif 'Outer Race' in file_path: return 2
    elif 'Ball' in file_path: return 3
    return -1

def find_signal_key(mat_data):
    for key in mat_data.keys():
        if key.endswith('DE_time') or key.endswith('FE_time'):
            return key
    return None

class NoisyTransfer48kDataset(Dataset):
    """支持信噪比动态注入的 48k 数据测试集"""
    def __init__(self, root_dir, snr_db=None, base_sample_length=1024):
        self.root_dir = root_dir
        self.snr_db = snr_db
        self.base_sample_length = base_sample_length
        self.samples = []
        self._scan_files()

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def _scan_files(self):
        print(f"📂 正在扫描测试数据: {self.root_dir}")
        for root, _, files in os.walk(self.root_dir):
            label = get_label_from_path(root)
            if label == -1: continue
            for file in files:
                if file.endswith('.mat') and not file.startswith('._'):
                    file_path = os.path.join(root, file)
                    is_48k = '48k' in file_path or '48k' in root
                    try:
                        mat_data = scipy.io.loadmat(file_path)
                        key = find_signal_key(mat_data)
                        if key:
                            signal = mat_data[key].flatten()
                            # 下采样对齐逻辑（与训练时保持一致）
                            if is_48k:
                                signal = signal[::4] 
                            num_samples = len(signal) // self.base_sample_length
                            for i in range(min(num_samples, 200)): 
                                start = i * self.base_sample_length
                                end = start + self.base_sample_length
                                self.samples.append({
                                    'signal': signal[start:end],
                                    'label': label,
                                    'fs': 12000 # 采频映射
                                })
                    except Exception as e:
                        print(f"读取异常 {file_path}: {e}")
        print(f"✅ 数据采样切片组装完毕，总切片数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        original_signal = sample['signal']
        label = sample['label']
        fs = sample['fs']

        # 在STFT之前往时域信号上覆盖白噪声
        noisy_signal = add_awgn_noise(original_signal, self.snr_db)

        # 进行 STFT 变换提取有效频特征图
        f, t, Zxx = scipy.signal.stft(noisy_signal, fs=fs, nperseg=64, noverlap=32)
        img = np.abs(Zxx)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

        return img_tensor, label

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
    if total == 0: return 0.0
    return (100.0 * correct / total)

# ================= 核心主函数 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" 🔊 变采样率 (48k) 高斯白噪声迁移模型抗噪性比对测试")
    print(f" 评估设备: {device}")
    print("=" * 60)

    dataset = NoisyTransfer48kDataset(TEST_DIR, snr_db=None)
    
    if len(dataset) == 0:
        print("❌ 未读到有效的 48k 数据。")
        return
        
    # 为了保证验证集的绝对公平，直接按照当初的SEED做同样的分割，只取当时的测试集 (80%)
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    print(f"🎯 成功还原测试集架构! 将使用 {test_size} 个样本进行抗噪性能压力测试。\n")

    results = {name: [] for name in MODELS_CONFIG.keys()}

    for snr in SNR_LIST:
        snr_name = f"{snr}dB" if snr is not None else "Clean"
        print(f"🚀 [正在注入噪声] SNR = {snr_name} ...")
        
        # 动态切换数据集内部的 SNR 配置
        dataset.set_snr(snr)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        for name, cfg in MODELS_CONFIG.items():
            model = cfg['class']().to(device)
            if not os.path.exists(cfg['finetuned_path']):
                print(f"  [⚠️] 找不到 {name} 的微调权重，跳过此模型!")
                results[name].append(0)
                continue
            
            # 灌入对应的finetuned模型与此批次的造噪结果对抗
            model.load_state_dict(torch.load(cfg['finetuned_path'], map_location=device))
            acc = test_model_simple(model, test_loader, device)
            results[name].append(acc)
            print(f"    👉 {name.ljust(15)} 测试准确率: {acc:6.2f}%")

    # ================= 绘图落盘 =================
    plt.style.use('ggplot')
    plt.figure(figsize=(11, 6))

    snr_labels = [f"{s}dB" if s is not None else "Clean" for s in SNR_LIST]
    markers = ['o', 's', '^', 'D']
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    
    for i, (name, accs) in enumerate(results.items()):
        plt.plot(snr_labels, accs, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                 label=name, linewidth=2.5, markersize=8)
        
        # 将数值写在点旁边
        for j, val in enumerate(accs):
            if val > 0:
                plt.annotate(f"{val:.1f}%", (j, val), textcoords="offset points", 
                             xytext=(0, 6 if i % 2 == 0 else -15), # 错开一下防止重叠
                             ha='center', fontsize=9, fontweight='bold', color=colors[i % len(colors)])

    plt.title("Finetuned Model Robustness Under AWGN (48k Transfer Learning)", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("SNR (Signal-to-Noise Ratio)", fontsize=12, fontweight='bold')
    plt.ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'noise_robustness_comparison.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n📊 全部压力测试处理完毕！四模抗噪性能演化曲线已保存: \n => {plot_path}")

if __name__ == '__main__':
    main()
