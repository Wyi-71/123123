"""
模型演进流程图：从 SimpleCNN 到 CNN-LSTM 的 "问题驱动" 进化路线
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# ========== 颜色定义 ==========
color_model = '#3b82f6'       # 蓝色 - 模型
color_problem = '#f43f5e'     # 红色 - 问题/瓶颈
color_solution = '#22c55e'    # 绿色 - 解决方案
color_arrow = '#64748b'       # 灰色 - 箭头
color_light_bg = '#f1f5f9'    # 浅灰 - 背景块

# ========== 辅助函数 ==========
def draw_box(ax, x, y, w, h, text, color, fontsize=11, text_color='white', alpha=0.95):
    box = FancyBboxPatch((x, y), w, h, 
                          boxstyle="round,pad=0.12", 
                          facecolor=color, edgecolor='#1e293b', 
                          linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, color=text_color, fontweight='bold', zorder=4, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#475569', style='->', lw=2.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, 
                               connectionstyle="arc3,rad=0.0"), zorder=2)

def draw_curved_arrow(ax, x1, y1, x2, y2, color='#475569', rad=0.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5, 
                               connectionstyle=f"arc3,rad={rad}"), zorder=2)

# ========== 标题 ==========
ax.text(8, 9.5, '轴承故障诊断模型演进路线图', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#0f172a')
ax.text(8, 9.05, 'Model Evolution Roadmap: Problem-Driven Architecture Improvement',
        ha='center', va='center', fontsize=11, color='#64748b', style='italic')

# ========== 第一列：模型 (蓝色) ==========
# 1. SimpleCNN
draw_box(ax, 0.5, 7.0, 3.2, 1.2, 'Stage 1\nSimpleCNN\n基础卷积网络', color_model, fontsize=11)

# 2. ResNet18
draw_box(ax, 0.5, 4.2, 3.2, 1.2, 'Stage 2\nResNet18\n深度残差网络', color_model, fontsize=11)

# 3. CBAM-ResNet18
draw_box(ax, 0.5, 1.4, 3.2, 1.2, 'Stage 3\nCBAM-ResNet18\n注意力增强网络', color_model, fontsize=11)

# ========== 第二列：发现的问题 (红色) ==========
draw_box(ax, 5.5, 7.0, 4.5, 1.2, 
         '⚠ 瓶颈：层数太浅，无法提取\n深层抽象特征，梯度消失', 
         color_problem, fontsize=10)

draw_box(ax, 5.5, 4.2, 4.5, 1.2, 
         '⚠ 瓶颈：对所有频率"一视同仁"\n无法聚焦故障敏感频带', 
         color_problem, fontsize=10)

draw_box(ax, 5.5, 1.4, 4.5, 1.2, 
         '⚠ 瓶颈：仅提取空间特征\n忽略了信号的时序演化规律', 
         color_problem, fontsize=10)

# ========== 第三列：解决方案 (绿色) ==========
draw_box(ax, 11.8, 5.6, 3.5, 1.2, 
         '✅ 解决方案\n引入残差跳跃连接\n(Shortcut Connection)', 
         color_solution, fontsize=10)

draw_box(ax, 11.8, 2.8, 3.5, 1.2, 
         '✅ 解决方案\n引入通道+空间注意力\n(Channel & Spatial Att.)', 
         color_solution, fontsize=10)

draw_box(ax, 11.8, 0.0, 3.5, 1.2, 
         '✅ 解决方案\n引入 LSTM 时序分支\n(Temporal Modeling)', 
         color_solution, fontsize=10)

# ========== 最终模型 ==========
draw_box(ax, 5.5, -0.2, 4.5, 1.0,
         '🏆 Stage 4: CNN-LSTM 双分支时频融合模型',
         '#7c3aed', fontsize=11)  # 紫色

# ========== 箭头连接 ==========
# 模型 -> 问题
draw_arrow(ax, 3.7, 7.6, 5.5, 7.6)
draw_arrow(ax, 3.7, 4.8, 5.5, 4.8)
draw_arrow(ax, 3.7, 2.0, 5.5, 2.0)

# 问题 -> 解决方案
draw_curved_arrow(ax, 10.0, 7.0, 11.8, 6.8, rad=-0.3)
draw_curved_arrow(ax, 10.0, 4.2, 11.8, 4.0, rad=-0.3)
draw_curved_arrow(ax, 10.0, 1.4, 11.8, 1.2, rad=-0.3)

# 解决方案 -> 下一个模型 (弧形回路)
draw_curved_arrow(ax, 11.8, 5.6, 2.1, 5.4, color=color_solution, rad=-0.2)
draw_curved_arrow(ax, 11.8, 2.8, 2.1, 2.6, color=color_solution, rad=-0.2)

# 最后的问题 -> 最终模型
draw_arrow(ax, 11.8, 0.0, 10.0, 0.3, color='#7c3aed')

# ========== 图例 ==========
legend_elements = [
    mpatches.Patch(facecolor=color_model, edgecolor='#1e293b', label='模型架构 (Model)'),
    mpatches.Patch(facecolor=color_problem, edgecolor='#1e293b', label='发现瓶颈 (Bottleneck)'),
    mpatches.Patch(facecolor=color_solution, edgecolor='#1e293b', label='改进方案 (Solution)'),
    mpatches.Patch(facecolor='#7c3aed', edgecolor='#1e293b', label='最终模型 (Final)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
          frameon=True, fancybox=True, shadow=True, 
          edgecolor='#cbd5e1', facecolor='white')

plt.tight_layout()
save_path = r'c:\Users\M\Desktop\GLM版本\模型演进流程图.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()
print(f"✅ 流程图已保存至: {save_path}")
