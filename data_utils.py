import os
import numpy as np
import scipy.io
import scipy.signal
import torch
from torch.utils.data import Dataset


def list_mat_files(root_dir):
    files = []
    for root, _, file_names in os.walk(root_dir):
        if "cwru_stft_images" in root or "cwru_thesis_ext" in root:
            continue
        for name in file_names:
            if name.endswith(".mat") and not name.startswith("._"):
                files.append(os.path.join(root, name))
    return files


def get_label_from_path(path):
    if "Normal" in path:
        return 0
    if "Inner Race" in path:
        return 1
    if "Outer Race" in path:
        return 2
    if "Ball" in path:
        return 3
    return -1


def get_sampling_rate(path):
    return 48000 if "48k" in path else 12000


def find_signal_key(mat_data):
    for key in mat_data.keys():
        if key.endswith("DE_time"):
            return key
    for key in mat_data.keys():
        if key.endswith("FE_time"):
            return key
    return None


def load_signal(path, sample_length):
    mat_data = scipy.io.loadmat(path)
    key = find_signal_key(mat_data)
    if key is None:
        return None
    signal = mat_data[key].flatten()
    if len(signal) < sample_length:
        return None
    return signal[:sample_length]


def add_gaussian_noise(signal, snr_db):
    if snr_db is None:
        return signal
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def stft_image(signal, fs, nperseg=126, noverlap=110, size=64):
    _, _, zxx = scipy.signal.stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    img = np.abs(zxx)
    img = img[:size, :size]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    return tensor


class CWRUDatasetTorch(Dataset):
    def __init__(self, root_dir, sample_length=1024, max_per_file=None, snr_db=None, split='all'):
        self.root_dir = root_dir
        self.sample_length = sample_length
        self.max_per_file = max_per_file # 如果为 None，则不限制数据量
        self.snr_db = snr_db
        self.split = split
        self.items = []
        self._build_index()

    def _build_index(self):
        for path in list_mat_files(self.root_dir):
            label = get_label_from_path(path)
            if label == -1:
                continue
            mat_data = scipy.io.loadmat(path)
            key = find_signal_key(mat_data)
            if key is None:
                continue
            signal = mat_data[key].flatten()
            
            # 引入切分机制防止特征泄露
            total_len = len(signal)
            if self.split == 'train':
                signal = signal[:int(total_len * 0.7)]
            elif self.split == 'test':
                signal = signal[int(total_len * 0.7):]
                
            num_segments = len(signal) // self.sample_length
            limit = min(num_segments, self.max_per_file) if self.max_per_file is not None else num_segments
            for i in range(limit):
                start = i * self.sample_length
                end = start + self.sample_length
                self.items.append((signal[start:end], label, get_sampling_rate(path)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        signal, label, fs = self.items[idx]
        signal = add_gaussian_noise(signal, self.snr_db)
        img_tensor = stft_image(signal, fs)
        return img_tensor, label


def build_svm_dataset(root_dir, sample_length=1024, max_per_file=None, snr_db=None):
    features = []
    labels = []
    for path in list_mat_files(root_dir):
        label = get_label_from_path(path)
        if label == -1:
            continue
        mat_data = scipy.io.loadmat(path)
        key = find_signal_key(mat_data)
        if key is None:
            continue
        signal = mat_data[key].flatten()
        num_segments = len(signal) // sample_length
        limit = min(num_segments, max_per_file) if max_per_file is not None else num_segments
        for i in range(limit):
            start = i * sample_length
            end = start + sample_length
            segment = signal[start:end]
            segment = add_gaussian_noise(segment, snr_db)
            fs = get_sampling_rate(path)
            img_tensor = stft_image(segment, fs)
            features.append(img_tensor.flatten().numpy())
            labels.append(label)
    return np.array(features), np.array(labels)
