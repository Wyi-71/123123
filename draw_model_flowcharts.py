"""
4种轴承故障诊断模型的程序流程图生成器
- SimpleCNN: 基础卷积神经网络
- ResNet18: 残差网络
- CBAM-ResNet18: 带注意力机制的残差网络
- CNN-LSTM: 时序特征提取网络
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

COLORS = {
    'input': '#E3F2FD',
    'conv': '#BBDEFB', 
    'pool': '#90CAF9',
    'fc': '#64B5F6',
    'output': '#42A5F5',
    'attention': '#FFCC80',
    'lstm': '#CE93D8',
    'residual': '#A5D6A7',
    'slice': '#FFAB91'
}

def draw_box(ax, x, y, w, h, text, color, fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color='#333333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def draw_simple_cnn(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SimpleCNN Architecture', fontsize=14, fontweight='bold', pad=10)
    
    boxes = [
        (4, 10.5, 2, 0.6, 'Input\n(1,64,64)', COLORS['input']),
        (4, 9.3, 2, 0.6, 'Conv1\n1->16', COLORS['conv']),
        (4, 8.4, 2, 0.5, 'BN + ReLU', COLORS['conv']),
        (4, 7.5, 2, 0.5, 'MaxPool', COLORS['pool']),
        (4, 6.3, 2, 0.6, 'Conv2\n16->32', COLORS['conv']),
        (4, 5.4, 2, 0.5, 'BN + ReLU', COLORS['conv']),
        (4, 4.5, 2, 0.5, 'MaxPool', COLORS['pool']),
        (4, 3.3, 2, 0.6, 'Conv3\n32->64', COLORS['conv']),
        (4, 2.4, 2, 0.5, 'BN + ReLU', COLORS['conv']),
        (4, 1.5, 2, 0.5, 'MaxPool', COLORS['pool']),
        (4, 0.3, 2, 0.6, 'FC: 4096->128\nDropout(0.5)', COLORS['fc']),
        (4, -0.7, 2, 0.6, 'FC: 128->4\nOutput', COLORS['output']),
    ]
    
    for x, y, w, h, text, color in boxes:
        draw_box(ax, x, y, w, h, text, color)
    
    for i in range(len(boxes)-1):
        draw_arrow(ax, 5, boxes[i][1], 5, boxes[i+1][1]+boxes[i+1][3])

def draw_resnet18(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('ResNet18 Architecture', fontsize=14, fontweight='bold', pad=10)
    
    boxes = [
        (4, 10.5, 2, 0.6, 'Input\n(1,64,64)', COLORS['input']),
        (4, 9.3, 2, 0.6, 'Conv1\n1->64, 7x7', COLORS['conv']),
        (4, 8.4, 2, 0.5, 'BN + ReLU', COLORS['conv']),
        (4, 7.5, 2, 0.5, 'MaxPool 3x3', COLORS['pool']),
        (4, 6.3, 2, 0.6, 'Layer1\n64ch', COLORS['conv']),
        (4, 5.1, 2, 0.6, 'Layer2\n128ch', COLORS['conv']),
        (4, 3.9, 2, 0.6, 'Layer3\n256ch', COLORS['conv']),
        (4, 2.7, 2, 0.6, 'Layer4\n512ch', COLORS['conv']),
        (4, 1.5, 2, 0.5, 'AvgPool', COLORS['pool']),
        (4, 0.3, 2, 0.6, 'FC: 512->4\nOutput', COLORS['output']),
    ]
    
    for x, y, w, h, text, color in boxes:
        draw_box(ax, x, y, w, h, text, color)
    
    for i in range(len(boxes)-1):
        draw_arrow(ax, 5, boxes[i][1], 5, boxes[i+1][1]+boxes[i+1][3])
    
    ax.text(7.5, 5.7, 'Each Layer:\n2 Residual Blocks\n+ Skip Connection', 
            fontsize=8, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor=COLORS['residual'], alpha=0.8))

def draw_cbam_resnet(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CBAM-ResNet18 Architecture', fontsize=14, fontweight='bold', pad=10)
    
    boxes = [
        (4, 10.5, 2, 0.6, 'Input\n(1,64,64)', COLORS['input']),
        (4, 9.3, 2, 0.6, 'Conv1\n1->64, 7x7', COLORS['conv']),
        (4, 8.4, 2, 0.5, 'BN + ReLU', COLORS['conv']),
        (4, 7.5, 2, 0.5, 'MaxPool 3x3', COLORS['pool']),
        (4, 6.3, 2, 0.5, 'Layer1', COLORS['conv']),
        (4, 5.5, 2, 0.5, 'CBAM-1', COLORS['attention']),
        (4, 4.5, 2, 0.5, 'Layer2', COLORS['conv']),
        (4, 3.7, 2, 0.5, 'CBAM-2', COLORS['attention']),
        (4, 2.7, 2, 0.5, 'Layer3', COLORS['conv']),
        (4, 1.9, 2, 0.5, 'CBAM-3', COLORS['attention']),
        (4, 0.9, 2, 0.5, 'Layer4', COLORS['conv']),
        (4, 0.1, 2, 0.5, 'CBAM-4', COLORS['attention']),
        (4, -0.9, 2, 0.6, 'FC: 512->4\nOutput', COLORS['output']),
    ]
    
    for x, y, w, h, text, color in boxes:
        draw_box(ax, x, y, w, h, text, color)
    
    for i in range(len(boxes)-1):
        draw_arrow(ax, 5, boxes[i][1], 5, boxes[i+1][1]+boxes[i+1][3])
    
    ax.text(7.5, 4.5, 'CBAM:\nChannel Attn\n+\nSpatial Attn', 
            fontsize=8, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor=COLORS['attention'], alpha=0.8))

def draw_cnn_lstm(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CNN-LSTM Architecture', fontsize=14, fontweight='bold', pad=10)
    
    boxes = [
        (4, 10.5, 2, 0.6, 'Input\n(1,64,64)', COLORS['input']),
        (4, 9.3, 2, 0.6, 'Time Slicing\n-> 8x(1,64,8)', COLORS['slice']),
        (4, 8.1, 2, 0.6, 'CNN Extractor\n(shared weights)', COLORS['conv']),
        (4, 6.9, 2, 0.6, 'Feature Seq\n(8, 128)', COLORS['conv']),
        (4, 5.7, 2, 0.6, 'LSTM\n2 layers, h=128', COLORS['lstm']),
        (4, 4.5, 2, 0.6, 'Last Timestep\n(128,)', COLORS['lstm']),
        (4, 3.3, 2, 0.6, 'Dropout(0.4)', COLORS['fc']),
        (4, 2.1, 2, 0.6, 'FC: 128->64', COLORS['fc']),
        (4, 0.9, 2, 0.6, 'FC: 64->4\nOutput', COLORS['output']),
    ]
    
    for x, y, w, h, text, color in boxes:
        draw_box(ax, x, y, w, h, text, color)
    
    for i in range(len(boxes)-1):
        draw_arrow(ax, 5, boxes[i][1], 5, boxes[i+1][1]+boxes[i+1][3])
    
    ax.text(7.5, 8.4, 'Each Slice:\nConv->BN->Pool\nConv->BN->Pool\nConv->BN->Pool\n->FC(128)', 
            fontsize=7, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor=COLORS['conv'], alpha=0.8))

def draw_cbam_detail(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CBAM Attention Module', fontsize=12, fontweight='bold', pad=10)
    
    draw_box(ax, 4, 7, 2, 0.5, 'Input Feature', COLORS['input'])
    draw_arrow(ax, 5, 7, 5, 6.5)
    
    draw_box(ax, 1.5, 5.5, 2, 0.5, 'AvgPool', COLORS['attention'])
    draw_box(ax, 6.5, 5.5, 2, 0.5, 'MaxPool', COLORS['attention'])
    draw_arrow(ax, 4.5, 6.5, 2.5, 6)
    draw_arrow(ax, 5.5, 6.5, 7.5, 6)
    
    draw_box(ax, 1.5, 4.3, 2, 0.5, 'MLP\n(reduce->expand)', COLORS['attention'])
    draw_box(ax, 6.5, 4.3, 2, 0.5, 'MLP\n(reduce->expand)', COLORS['attention'])
    draw_arrow(ax, 2.5, 5.5, 2.5, 4.8)
    draw_arrow(ax, 7.5, 5.5, 7.5, 4.8)
    
    draw_box(ax, 4, 3.3, 2, 0.5, 'Add + Sigmoid', COLORS['attention'])
    draw_arrow(ax, 2.5, 4.3, 5, 3.8)
    draw_arrow(ax, 7.5, 4.3, 5, 3.8)
    
    draw_box(ax, 4, 2.1, 2, 0.5, 'Channel Attention', COLORS['attention'])
    draw_arrow(ax, 5, 3.3, 5, 2.6)
    
    ax.text(5, 1.6, 'x', fontsize=16, ha='center', va='center', fontweight='bold')
    draw_arrow(ax, 5, 2.1, 5, 1.7)
    
    draw_box(ax, 4, 0.5, 2, 0.5, 'Spatial Attention', COLORS['attention'])
    draw_arrow(ax, 5, 1.4, 5, 1)
    
    draw_box(ax, 4, -0.5, 2, 0.5, 'Output Feature', COLORS['output'])
    draw_arrow(ax, 5, 0.5, 5, 0)
    
    ax.text(0.5, 5.7, 'Channel Attn\n(which freq bands\nare important)', fontsize=7, ha='left')
    ax.text(8, 0.7, 'Spatial Attn\n(which time-freq\nregions matter)', fontsize=7, ha='left')

def main():
    fig = plt.figure(figsize=(16, 14))
    
    ax1 = fig.add_subplot(2, 3, 1)
    draw_simple_cnn(ax1)
    
    ax2 = fig.add_subplot(2, 3, 2)
    draw_resnet18(ax2)
    
    ax3 = fig.add_subplot(2, 3, 3)
    draw_cbam_resnet(ax3)
    
    ax4 = fig.add_subplot(2, 3, 4)
    draw_cnn_lstm(ax4)
    
    ax5 = fig.add_subplot(2, 3, 5)
    draw_cbam_detail(ax5)
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    legend_text = """
    [Model Comparison]
    
    > SimpleCNN
      - 3 Conv layers + 2 FC layers
      - Fewer parameters, fast training
      - Baseline model
    
    > ResNet18
      - Residual connections (Skip)
      - Pretrained weight transfer
      - Deeper feature extraction
    
    > CBAM-ResNet18
      - Adds attention to ResNet
      - Channel Attn: important freq bands
      - Spatial Attn: important time-freq regions
    
    > CNN-LSTM
      - CNN extracts spatial features
      - LSTM models temporal dependencies
      - Captures fault evolution process
    """
    ax6.text(0.1, 0.95, legend_text, fontsize=9, va='top', 
             family='monospace', linespacing=1.5,
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))
    
    plt.suptitle('CWRU Bearing Fault Diagnosis Model Architecture Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = r'c:\Users\M\Desktop\GLM版本\C_预训练权重与项目展示\Figures_and_Plots\model_architecture_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Flowchart saved: {save_path}")
    plt.close()

if __name__ == '__main__':
    main()
