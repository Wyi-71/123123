import os
import numpy as np
import scipy.io
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def get_label_from_path(file_path):
    if 'Normal' in file_path: return 0
    elif 'Inner Race' in file_path: return 1
    elif 'Outer Race' in file_path: return 2
    elif 'Ball' in file_path: return 3
    return -1

def find_signal_key(mat_data):
    for key in mat_data.keys():
        if key.endswith('DE_time'): return key
    for key in mat_data.keys():
        if key.endswith('FE_time'): return key
    return None

class Test48kDataset(Dataset):
    """
    专门为测试 48k 跨域泛化能力设计的独立数据集读取器。
    核心逻辑：在读取 48k 信号时，强制将其下采样为 12k 的等效物理长度。
    这样一来，STFT时频图上的周期脉冲特征在物理时间轴上才会和 12k 训练集保持对齐，
    从而避免模型（尤其是CNN-LSTM）因为看不见完整周期而崩溃。
    """
    def __init__(self, root_dir, base_sample_length=1024):
        self.root_dir = root_dir
        # 12k 模型是以 1024 为基准切片的
        self.base_sample_length = base_sample_length
        self.samples = []
        self._scan_files()

    def _scan_files(self):
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
                            
                            # 核心降采样逻辑：如果是 48k，每隔 4 个点抽一个点，强行压成 12k
                            if is_48k:
                                signal = signal[::4] 
                            
                            # 切分信号
                            num_samples = len(signal) // self.base_sample_length
                            for i in range(min(num_samples, 50)): # 每个文件测 50 个片段
                                start = i * self.base_sample_length
                                end = start + self.base_sample_length
                                self.samples.append({
                                    'signal': signal[start:end],
                                    'label': label,
                                    'fs': 12000 # 因为已经降采样了，统一视为 12000 算傅里叶
                                })
                    except Exception as e:
                        print(f"读取异常 {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        signal = sample['signal']
        label = sample['label']
        fs = sample['fs']

        # 生成STFT图 并插值为 64x64，和训练时一模一样
        f, t, Zxx = scipy.signal.stft(signal, fs=fs, nperseg=64, noverlap=32)
        img = np.abs(Zxx)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

        return img_tensor, label

