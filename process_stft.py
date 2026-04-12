import os
import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
import re

# 配置
SRC_DIR = r'c:\Users\M\Desktop\GLM版本\cwru_data'
DST_DIR = r'c:\Users\M\Desktop\GLM版本\cwru_stft_images'

# 确保输出目录存在
if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)

def get_sampling_rate(file_path):
    """根据文件路径推断采样率"""
    if '48k' in file_path:
        return 48000
    # 默认 12k，包括 Normal Baseline 和 12k 文件夹
    return 12000

def find_signal_key(mat_data):
    """查找包含振动信号的 key"""
    # 优先查找 DE (Drive End)
    for key in mat_data.keys():
        if key.endswith('DE_time'):
            return key
    # 其次查找 FE (Fan End)
    for key in mat_data.keys():
        if key.endswith('FE_time'):
            return key
    return None

def process_file(file_path, output_path):
    try:
        mat_data = scipy.io.loadmat(file_path)
        signal_key = find_signal_key(mat_data)
        
        if not signal_key:
            print(f"Skipping {file_path}: No valid signal key found.")
            return

        signal = mat_data[signal_key].flatten()
        fs = get_sampling_rate(file_path)

        # STFT 参数
        nperseg = 256
        noverlap = nperseg // 2
        
        # 计算 STFT
        f, t, Zxx = scipy.signal.stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        # 使用 pcolormesh 绘制时频图，取幅度的对数以增强对比度
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        #plt.title(f'STFT Magnitude - {os.path.basename(file_path)}')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Magnitude')
        
        # 保存
        plt.savefig(output_path)
        plt.close()
        print(f"Processed: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    count = 0
    for root, dirs, files in os.walk(SRC_DIR):
        # 跳过输出目录，防止循环处理（如果输出目录在源目录内）
        if DST_DIR in root:
            continue
            
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'): # 忽略 macOS 临时文件等
                src_file_path = os.path.join(root, file)
                
                # 构建目标路径
                rel_path = os.path.relpath(root, SRC_DIR)
                dst_folder = os.path.join(DST_DIR, rel_path)
                
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                
                dst_file_path = os.path.join(dst_folder, file.replace('.mat', '.png'))
                
                process_file(src_file_path, dst_file_path)
                count += 1
                
    print(f"Done. Processed {count} files.")

if __name__ == '__main__':
    main()
