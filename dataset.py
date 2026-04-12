import os
import numpy as np
import scipy.io
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# 标签映射
LABEL_MAP = {
    'Normal': 0,
    'Inner Race': 1,
    'Outer Race': 2,
    'Ball': 3
}

def get_label_from_path(file_path):
    """根据文件路径解析标签"""
    if 'Normal' in file_path:
        return 0
    elif 'Inner Race' in file_path:
        return 1
    elif 'Outer Race' in file_path:
        return 2
    elif 'Ball' in file_path:
        return 3
    return -1

def get_sampling_rate(file_path):
    if '48k' in file_path:
        return 48000
    return 12000

def find_signal_key(mat_data):
    for key in mat_data.keys():
        if key.endswith('DE_time'):
            return key
    for key in mat_data.keys():
        if key.endswith('FE_time'):
            return key
    return None

class CWRUDataset(Dataset):
    def __init__(self, root_dir, sample_length=1024, transform=None, split='all'):
        self.root_dir = root_dir
        self.sample_length = sample_length
        self.transform = transform
        self.split = split
        self.samples = []
        self._scan_files()

    def _scan_files(self):
        for root, _, files in os.walk(self.root_dir):
            if 'cwru_stft_images' in root: # 跳过生成的图片目录
                continue
            
            # 严格筛选：仅允许 'Normal Baseline' 和 '12k Drive End Bearing Fault Data' 参与训练
            if not ('Normal Baseline' in root or '12k Drive End Bearing Fault Data' in root):
                continue
                
            label = get_label_from_path(root)
            if label == -1:
                continue
                
            for file in files:
                if file.endswith('.mat') and not file.startswith('._'):
                    file_path = os.path.join(root, file)
                    try:
                        mat_data = scipy.io.loadmat(file_path)
                        key = find_signal_key(mat_data)
                        if key:
                            signal = mat_data[key].flatten()
                            
                            # 防止数据泄露：基于时间的强制切分 (70% 训练, 30% 测试)
                            total_len = len(signal)
                            if self.split == 'train':
                                signal = signal[:int(total_len * 0.7)]
                            elif self.split == 'test':
                                signal = signal[int(total_len * 0.7):]
                            
                            # 切分信号
                            num_samples = len(signal) // self.sample_length
                            # 提取该文件中的所有样本
                            for i in range(num_samples): 
                                start = i * self.sample_length
                                end = start + self.sample_length
                                self.samples.append({
                                    'signal': signal[start:end],
                                    'label': label,
                                    'fs': get_sampling_rate(file_path)
                                })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        signal = sample['signal']
        label = sample['label']
        fs = sample['fs']

        # 生成STFT图：调整参数使得尺寸直接生成对应的形状而无需后期强制插值（彻底消除物理失真）
        # 1024 点 -> 126 窗口, 110 重叠 -> 产出 (64, 65) 尺寸矩阵
        f, t, Zxx = scipy.signal.stft(signal, fs=fs, nperseg=126, noverlap=110)
        img = np.abs(Zxx)
        
        # 截取精确的 64x64 大小
        img = img[:64, :64]
        
        # 归一化到 0-1
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # 转换为 tensor (1, 64, 64)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)

        return img_tensor, label

if __name__ == '__main__':
    # 简单测试
    dataset = CWRUDataset(r'c:\Users\M\Desktop\GLM版本\cwru_data')
    print(f"Total samples: {len(dataset)}")
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Sample shape: {img.shape}, Label: {label}")
