import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(project_root, 'cwru_data'))

from dataset import CWRUDataset
from model import SimpleCNN
from resnet_model import BearingResNet18

try:
    sys.path.insert(0, os.path.join(project_root, '时空注意力机制加强版'))
    from cbam_model import BearingResNet18_CBAM
    HAS_CBAM = True
except ImportError:
    HAS_CBAM = False

try:
    sys.path.insert(0, os.path.join(project_root, '双分支 CNN-LSTM 时频特征提取架构'))
    from cnn_lstm_model import STFT_CNN_LSTM
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False


CLASS_NAMES = ['Normal', 'Inner Race', 'Outer Race', 'Ball']
CLASS_NAMES_CN = ['正常', '内圈故障', '外圈故障', '滚动体故障']
DATA_DIR = os.path.join(project_root, 'cwru_data')
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(model_name, num_classes=4):
    models_map = {
        'SimpleCNN': lambda: SimpleCNN(num_classes=num_classes),
        'ResNet18': lambda: BearingResNet18(num_classes=num_classes),
        'CBAM-ResNet18': lambda: BearingResNet18_CBAM(num_classes=num_classes) if HAS_CBAM else None,
        'CNN-LSTM': lambda: STFT_CNN_LSTM(num_classes=num_classes) if HAS_LSTM else None,
    }
    
    if model_name not in models_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_fn = models_map[model_name]
    if model_fn is None:
        return None
    
    return model_fn()


def get_model_path(model_name):
    path_map = {
        'SimpleCNN': os.path.join(DATA_DIR, 'cwru_cnn_model.pth'),
        'ResNet18': os.path.join(DATA_DIR, 'cwru_resnet_model.pth'),
        'CBAM-ResNet18': os.path.join(DATA_DIR, 'cwru_resnet_cbam_model.pth'),
        'CNN-LSTM': os.path.join(DATA_DIR, 'cwru_cnn_lstm_model.pth'),
    }
    return path_map.get(model_name)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, model_name, save_path, acc):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_CN, yticklabels=CLASS_NAMES_CN,
                ax=ax, cbar_kws={'label': '样本数'})
    
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title(f'{model_name}\n准确率: {acc:.2f}%', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_all_confusion_matrices(results_dict, save_path):
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(results_dict.items()):
        cm = data['cm']
        acc = data['acc']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES_CN, yticklabels=CLASS_NAMES_CN,
                    ax=axes[idx], cbar_kws={'label': '样本数'},
                    annot_kws={'size': 11})
        
        axes[idx].set_xlabel('预测标签', fontsize=11)
        axes[idx].set_ylabel('真实标签', fontsize=11)
        axes[idx].set_title(f'{model_name}\nAcc: {acc:.2f}%', fontsize=12, fontweight='bold')
    
    plt.suptitle('各模型混淆矩阵对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n加载数据集...")
    full_dataset = CWRUDataset(DATA_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集样本数: {len(test_dataset)}")
    
    model_names = ['SimpleCNN', 'ResNet18']
    if HAS_CBAM:
        model_names.append('CBAM-ResNet18')
    if HAS_LSTM:
        model_names.append('CNN-LSTM')
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"评估模型: {model_name}")
        print('='*50)
        
        try:
            model = load_model(model_name)
            if model is None:
                print(f"跳过 {model_name}: 模型不可用")
                continue
            
            model = model.to(device)
            
            model_path = get_model_path(model_name)
            if not os.path.exists(model_path):
                print(f"警告: 模型权重文件不存在: {model_path}")
                print(f"跳过 {model_name}")
                continue
            
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            
            labels, preds = evaluate_model(model, test_loader, device)
            
            acc = accuracy_score(labels, preds) * 100
            cm = confusion_matrix(labels, preds)
            
            results[model_name] = {'cm': cm, 'acc': acc}
            
            print(f"\n{model_name} 测试结果:")
            print(f"总体准确率: {acc:.2f}%")
            print("\n分类报告:")
            print(classification_report(labels, preds, target_names=CLASS_NAMES_CN, digits=4))
            
            single_save = os.path.join(SAVE_DIR, f'confusion_matrix_{model_name.replace("-", "_").replace(" ", "_")}.png')
            plot_confusion_matrix(cm, model_name, single_save, acc)
            
        except Exception as e:
            print(f"评估 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) > 0:
        comparison_path = os.path.join(SAVE_DIR, 'confusion_matrices_comparison.png')
        plot_all_confusion_matrices(results, comparison_path)
        
        print(f"\n{'='*50}")
        print("汇总结果:")
        print('='*50)
        for name, data in sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True):
            print(f"{name:20s}: 准确率 {data['acc']:.2f}%")
    else:
        print("\n没有成功评估任何模型！请检查模型权重文件是否存在。")


if __name__ == '__main__':
    main()
