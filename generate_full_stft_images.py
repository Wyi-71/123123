import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import CWRUDataset

# 配置
ROOT_DIR = r'c:\Users\M\Desktop\GLM版本\cwru_data'
OUTPUT_DIR = r'c:\Users\M\Desktop\GLM版本\cwru_stft_images_all'
LABEL_NAMES = {0: 'Normal', 1: 'Inner Race', 2: 'Outer Race', 3: 'Ball'}

def main():
    # 初始化数据集 (此时已经移除了 50 个样本的限制)
    dataset = CWRUDataset(ROOT_DIR)
    print(f"检测到总样本数: {len(dataset)}")

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for label_idx, label_name in LABEL_NAMES.items():
        label_dir = os.path.join(OUTPUT_DIR, label_name)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

    # 遍历并保存图片
    # 为了避免保存几万张图片耗时过长，我们可以先打印进度
    print("正在开始生成并保存时频图...")
    
    # 使用 matplotlib 的 colormap 将张量转换为漂亮的图片
    for i in tqdm(range(len(dataset))):
        img_tensor, label = dataset[i]
        
        # img_tensor shape: (1, 64, 64)
        img_np = img_tensor.squeeze().numpy()
        
        # 构造保存路径
        label_name = LABEL_NAMES.get(label, 'Unknown')
        save_path = os.path.join(OUTPUT_DIR, label_name, f"sample_{i:05d}.png")
        
        # 如果图片已存在，则跳过（可选，方便断点续传）
        if os.path.exists(save_path):
            continue
            
        # 绘制并保存
        plt.imsave(save_path, img_np, cmap='viridis')
        
        # 每 500 张打印一次提示，或者依赖 tqdm
        if i % 1000 == 0 and i > 0:
            pass

    print(f"数据生成完成！所有图片已保存至: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
