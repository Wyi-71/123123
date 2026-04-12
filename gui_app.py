import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import torch
import scipy.io
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# 导入附加目录的模块
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, '时空注意力机制加强版'))
sys.path.append(os.path.join(parent_dir, '双分支 CNN-LSTM 时频特征提取架构'))

# 导入模型
try:
    from model import SimpleCNN
except ImportError:
    SimpleCNN = None

try:
    from resnet_model import BearingResNet18
except ImportError:
    BearingResNet18 = None

try:
    from cbam_model import BearingResNet18_CBAM
except ImportError:
    BearingResNet18_CBAM = None

try:
    from cnn_lstm_model import STFT_CNN_LSTM
except ImportError:
    STFT_CNN_LSTM = None

# 分类标签映射
LABEL_MAP = {
    0: 'Normal (正常)',
    1: 'Inner Race Fault (内圈故障)',
    2: 'Outer Race Fault (外圈故障)',
    3: 'Ball Fault (滚动体故障)'
}

# --- 从头设计：现代工业仪表盘架构 (Dashboard Theme) ---
THEME = {
    "sidebar_bg": "#1e293b",       # 侧边栏：深色现代钢材蓝
    "sidebar_fg": "#f8fafc",       # 侧边栏文字：极简白
    "sidebar_accent": "#334155",   # 侧边栏分割线：深灰
    "main_bg": "#f1f5f9",          # 主背景：干净的亮色护眼灰蓝
    "card_bg": "#ffffff",          # 信息卡片：纯白，突显数据清晰度
    "text_main": "#0f172a",        # 黑岩亮字：主数据文本
    "text_sub": "#64748b",         # 灰蓝辅助：信息辅助说明
    "primary": "#2563eb",          # 按钮与核心：工程明亮蓝
    "success": "#059669",          # 成功：工业暗绿
    "danger": "#dc2626",           # 故障：警戒红
    "warning": "#d97706",          # 警告：警示橙
    "term_bg": "#020617",          # 终端：极暗全黑
    "term_fg": "#38bdf8",          # 终端代码：控制台蓝绿色
}

class CWRUDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CWRU 智能故障诊断云仪表盘 (Engineering & Academic Dashboard)")
        self.root.geometry("1300x850") 
        self.root.configure(bg=THEME["main_bg"])
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_model_type = "CNN"
        self.current_file_path = None
        self.current_column = 0

        self.setup_global_styles()
        self.build_architecture()
        self.load_model()

    def setup_global_styles(self):
        # 微调基于 tkinter 标准控件的主题融合风格
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass

        self.style.configure("Primary.TButton", 
                             background=THEME["primary"], foreground="#ffffff", 
                             font=("Microsoft YaHei UI", 11, "bold"), 
                             borderwidth=0, focuscolor=THEME["primary"])
        self.style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        
        self.style.configure("Outline.TButton", 
                             background="#ffffff", foreground=THEME["text_main"],
                             font=("Microsoft YaHei UI", 10),
                             borderwidth=1, bordercolor=THEME["text_sub"])
        self.style.map("Outline.TButton", background=[("active", "#e2e8f0")])

    def build_architecture(self):
        # ====================
        # 核心模块 1：系统总控侧边栏
        # ====================
        sidebar = tk.Frame(self.root, bg=THEME["sidebar_bg"], width=350)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False) 
        
        # 1.1 品牌/标题区
        title_frame = tk.Frame(sidebar, bg=THEME["sidebar_bg"], pady=35, padx=25)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="CWRU", font=("Consolas", 32, "bold"), bg=THEME["sidebar_bg"], fg=THEME["primary"]).pack(anchor=tk.W)
        tk.Label(title_frame, text="Fault Diagnosis\nPlatform", font=("Microsoft YaHei UI", 14, "bold"), bg=THEME["sidebar_bg"], fg=THEME["sidebar_fg"], justify=tk.LEFT).pack(anchor=tk.W, pady=(5,0))
        
        tk.Frame(sidebar, bg=THEME["sidebar_accent"], height=2).pack(fill=tk.X, padx=25)

        # 1.2 算法核心选择
        control_frame = tk.Frame(sidebar, bg=THEME["sidebar_bg"], padx=25, pady=25)
        control_frame.pack(fill=tk.X)
        tk.Label(control_frame, text="1. 算法计算节点引擎", font=("Microsoft YaHei UI", 11, "bold"), bg=THEME["sidebar_bg"], fg=THEME["sidebar_fg"]).pack(anchor=tk.W, pady=(0,8))
        
        self.model_var = tk.StringVar(value="SimpleCNN (CNN)")
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                        values=["SimpleCNN (CNN)", "ResNet18", "CBAM-ResNet18 (时空注意力)", "CNN-LSTM (时频特征序列)"],
                                        state="readonly", font=("Microsoft YaHei UI", 10))
        self.model_combo.pack(fill=tk.X, pady=(0, 15), ipady=3)
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self.load_model())
        ttk.Button(control_frame, text="❖ 挂载引擎", style="Outline.TButton", command=self.load_model).pack(fill=tk.X)

        tk.Frame(sidebar, bg=THEME["sidebar_accent"], height=1).pack(fill=tk.X, padx=25)
        
        # 1.3 数据通道设定
        col_frame = tk.Frame(sidebar, bg=THEME["sidebar_bg"], padx=25, pady=25)
        col_frame.pack(fill=tk.X)
        tk.Label(col_frame, text="2. 数据矩阵通道切换", font=("Microsoft YaHei UI", 11, "bold"), bg=THEME["sidebar_bg"], fg=THEME["sidebar_fg"]).pack(anchor=tk.W, pady=(0,8))
        
        c_inner = tk.Frame(col_frame, bg=THEME["sidebar_bg"])
        c_inner.pack(fill=tk.X)
        self.column_var = tk.StringVar(value="0")
        self.column_spinbox = ttk.Spinbox(c_inner, from_=0, to=0, textvariable=self.column_var, width=10, state="disabled", font=("Consolas", 12))
        self.column_spinbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10), ipady=3)
        self.btn_change_col = ttk.Button(c_inner, text="切换", style="Outline.TButton", command=self.change_column, state=tk.DISABLED)
        self.btn_change_col.pack(side=tk.RIGHT)

        # 1.4 系统底层终端
        term_label_frame = tk.Frame(sidebar, bg=THEME["sidebar_bg"], padx=25)
        term_label_frame.pack(fill=tk.X, pady=(15, 5))
        tk.Label(term_label_frame, text=">_ 后台系统终端输出", font=("Microsoft YaHei UI", 10, "bold"), bg=THEME["sidebar_bg"], fg=THEME["text_sub"]).pack(anchor=tk.W)
        
        self.info_text = tk.Text(sidebar, bg=THEME["term_bg"], fg=THEME["term_fg"],
                                 font=("Consolas", 9), borderwidth=0, padx=12, pady=12,
                                 insertbackground="white", highlightthickness=0)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 25))


        # ====================
        # 核心模块 2：工程视界主面板
        # ====================
        main_content = tk.Frame(self.root, bg=THEME["main_bg"])
        main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 2.1 顶级控制工具栏
        header_frame = tk.Frame(main_content, bg=THEME["card_bg"], height=90)
        header_frame.pack(fill=tk.X, padx=25, pady=25)
        header_frame.pack_propagate(False)
        
        ttk.Button(header_frame, text="打开数据集文件 (.mat)", style="Primary.TButton", command=self.open_file).pack(side=tk.LEFT, padx=25, pady=25, ipady=6, ipadx=15)
        
        file_info_frame = tk.Frame(header_frame, bg=THEME["card_bg"])
        file_info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 25), pady=20)
        tk.Label(file_info_frame, text="当前加载轨道:", font=("Microsoft YaHei UI", 9), bg=THEME["card_bg"], fg=THEME["text_sub"]).pack(anchor=tk.W)
        self.file_label = tk.Label(file_info_frame, text="< 尚未导入任何数据文件 >", font=("Consolas", 13, "bold"), bg=THEME["card_bg"], fg=THEME["text_main"])
        self.file_label.pack(anchor=tk.W, pady=(2, 0))
        
        # 2.1.1 右侧模块状态信号灯
        self.status_var = tk.StringVar(value="SYSTEM READY")
        self.status_badge = tk.Label(header_frame, textvariable=self.status_var, font=("Consolas", 12, "bold"),
                                     bg="#e2e8f0", fg=THEME["text_main"], padx=20, pady=8)
        self.status_badge.pack(side=tk.RIGHT, padx=25, pady=25)

        # 2.2 仪表盘数据聚合卡片区 (Kpi Cards)
        cards_frame = tk.Frame(main_content, bg=THEME["main_bg"])
        cards_frame.pack(fill=tk.X, padx=25, pady=(0, 25))
        cards_frame.columnconfigure((0, 1, 2), weight=1, uniform="eq")

        # 数据包参数板
        c1 = tk.Frame(cards_frame, bg=THEME["card_bg"], padx=25, pady=20)
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        tk.Label(c1, text="分析目标源属性", font=("Microsoft YaHei UI", 10), bg=THEME["card_bg"], fg=THEME["text_sub"]).pack(anchor=tk.W)
        self.card1_val = tk.Label(c1, text="CWRU Base", font=("Consolas", 18, "bold"), bg=THEME["card_bg"], fg=THEME["text_main"])
        self.card1_val.pack(anchor=tk.W, pady=(6,0))
        self.card1_sub = tk.Label(c1, text="Fs Sample Rate: -- kHz", font=("Consolas", 11), bg=THEME["card_bg"], fg=THEME["text_sub"])
        self.card1_sub.pack(anchor=tk.W)

        # 模型推演核心结果
        c2 = tk.Frame(cards_frame, bg=THEME["card_bg"], padx=25, pady=20)
        c2.grid(row=0, column=1, sticky="nsew", padx=12)
        tk.Label(c2, text="神经网络推演预测结论", font=("Microsoft YaHei UI", 10), bg=THEME["card_bg"], fg=THEME["text_sub"]).pack(anchor=tk.W)
        self.card2_val = tk.Label(c2, text="---", font=("Microsoft YaHei UI", 20, "bold"), bg=THEME["card_bg"], fg=THEME["text_main"])
        self.card2_val.pack(anchor=tk.W, pady=(4,0))
        self.card2_sub = tk.Label(c2, text="等待执行运算 ...", font=("Microsoft YaHei UI", 11), bg=THEME["card_bg"], fg=THEME["text_sub"])
        self.card2_sub.pack(anchor=tk.W)

        # 预测置信指标
        c3 = tk.Frame(cards_frame, bg=THEME["card_bg"], padx=25, pady=20)
        c3.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        tk.Label(c3, text="模型输出置信度", font=("Microsoft YaHei UI", 10), bg=THEME["card_bg"], fg=THEME["text_sub"]).pack(anchor=tk.W)
        self.card3_val = tk.Label(c3, text="0.00 %", font=("Consolas", 28, "bold"), bg=THEME["card_bg"], fg=THEME["primary"])
        self.card3_val.pack(anchor=tk.W, pady=(0,0))
        
        # 2.3 数据全景绘图容器 (Scientific Visualization Area)
        plot_container = tk.Frame(main_content, bg=THEME["card_bg"])
        plot_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 25))
        
        header_plot = tk.Frame(plot_container, bg=THEME["card_bg"], pady=15, padx=20)
        header_plot.pack(fill=tk.X)
        tk.Label(header_plot, text="信号时频解析特征与概率空间矩阵映射", 
                 font=("Microsoft YaHei UI", 12, "bold"), bg=THEME["card_bg"], fg=THEME["text_main"]).pack(side=tk.LEFT)
                 
        self.figure_frame = tk.Frame(plot_container, bg=THEME["card_bg"])
        self.figure_frame.pack(fill=tk.BOTH, expand=True)
        
        self._init_empty_plot()

    # =================
    # 工具函数 & UI修改器
    # =================
    def set_status(self, msg, state="info"):
        self.status_var.set(msg.upper())
        if state == "success":
            self.status_badge.config(bg=THEME["success"], fg="#ffffff")
        elif state == "err":
            self.status_badge.config(bg=THEME["danger"], fg="#ffffff")
        elif state == "warn":
            self.status_badge.config(bg=THEME["warning"], fg="#ffffff")
        else:
            self.status_badge.config(bg="#e2e8f0", fg=THEME["text_main"])
            
    def log(self, message):
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)

    # =================
    # 后台数学绘图实现 (Matplotlib 学术风组合)
    # =================
    def _init_empty_plot(self):
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=100, facecolor=THEME["card_bg"])
        fig.patch.set_facecolor(THEME["card_bg"])
        
        for ax in [ax1, ax2]:
            ax.set_facecolor("#f8fafc")
            ax.tick_params(colors='#1e293b')
            for spine in ax.spines.values():
                spine.set_color('#cbd5e1')
                spine.set_linewidth(1.5)
                
        ax1.set_title("STFT Time-Frequency Envelope", color=THEME["text_main"], fontweight='bold', fontsize=12)
        ax1.text(0.5, 0.5, "[ AWAITING SIGNAL FETCH ]", ha='center', va='center', color='#94a3b8', fontdict={'family': 'Consolas'})
        
        ax2.set_title("Model Output Probability Space", color=THEME["text_main"], fontweight='bold', fontsize=12)
        ax2.set_xlim(-0.5, 3.5)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['Normal', 'Inner', 'Outer', 'Ball'], rotation=0)
        
        fig.tight_layout(pad=2.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_results(self, f, t, Zxx, probs):
        for widget in self.figure_frame.winfo_children():
            widget.destroy()

        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
        fig = plt.Figure(figsize=(10, 4.5), dpi=100, facecolor=THEME["card_bg"])
        
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor("#f8fafc")
        c = ax1.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='jet')
        ax1.set_title('STFT Extracted Feature Responses', color=THEME["text_main"], fontweight='bold', fontsize=12)
        ax1.set_ylabel('Frequency Space [Hz]', color=THEME["text_main"], fontweight='heavy')
        ax1.set_xlabel('Time Sequence [sec]', color=THEME["text_main"], fontweight='heavy')
        ax1.grid(True, linestyle=':', alpha=0.6, color='#94a3b8')
        for spine in ax1.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.5)
        cb = fig.colorbar(c, ax=ax1, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9, colors=THEME["text_main"])
        
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor("#f8fafc")
        
        classes = ['Normal', 'Inner', 'Outer', 'Ball']
        x = np.arange(len(classes))
        
        colors = ['#e2e8f0'] * len(classes)
        max_idx = np.argmax(probs)
        colors[max_idx] = THEME["primary"] 
        
        bars = ax2.bar(x, probs * 100, color=colors, alpha=0.95, edgecolor='#1e293b', linewidth=2.0)
        ax2.set_title('Machine Confidence Levels (%)', color=THEME["text_main"], fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=0, color=THEME["text_main"], fontweight='medium')
        ax2.set_ylim(0, 105)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
        
        for spine in ax2.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.5)
            
        for bar in bars:
            height = bar.get_height()
            if height > 1: 
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=10, color=THEME["text_main"], fontweight='bold', fontdict={'family': 'Consolas'})
        
        fig.tight_layout(pad=2.5)
        
        canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # =================
    # Pytorch 核心推理管线
    # =================
    def load_model(self):
        selection = self.model_var.get()
        self.set_status("LOADING", "warn")
        self.log(f"\n[ENGINE PREP] 正在拉起 {selection} 核心构件...")
        
        if "CNN-LSTM" in selection:
            self.current_model_type = "CNN-LSTM"
            model_filename = "cwru_cnn_lstm_model.pth"
            model_class = STFT_CNN_LSTM
            model_dir = os.path.join(os.path.dirname(self.script_dir), '双分支 CNN-LSTM 时频特征提取架构')
            if STFT_CNN_LSTM is None:
                self.log(">> [FATAL] cnn_lstm_model.py 模块编译缺失，拒绝执行")
                self.set_status("DEP_ERROR", "err")
                return
        elif "CBAM" in selection:
            self.current_model_type = "CBAM-RESNET"
            model_filename = "cwru_resnet_cbam_model.pth"
            model_class = BearingResNet18_CBAM
            model_dir = os.path.join(os.path.dirname(self.script_dir), '时空注意力机制加强版')
            if BearingResNet18_CBAM is None:
                self.log(">> [FATAL] cbam_model.py 模块编译缺失，拒绝执行")
                self.set_status("DEP_ERROR", "err")
                return
        elif "ResNet" in selection:
            self.current_model_type = "RESNET"
            model_filename = "cwru_resnet_model.pth"
            model_class = BearingResNet18
            model_dir = self.script_dir
            if BearingResNet18 is None:
                self.log(">> [FATAL] resnet_model.py 模块编译缺失，拒绝执行")
                self.set_status("DEP_ERROR", "err")
                return
        else:
            self.current_model_type = "CNN"
            model_filename = "cwru_cnn_model.pth"
            model_class = SimpleCNN
            model_dir = self.script_dir
            if SimpleCNN is None:
                self.log(">> [FATAL] model.py 模块编译缺失，拒绝执行")
                self.set_status("DEP_ERROR", "err")
                return
                
        self.model_path = os.path.join(model_dir, model_filename)

        def _load():
            try:
                if not os.path.exists(self.model_path):
                    self.root.after(0, lambda: messagebox.showwarning("模型库丢失", f"执行文件中断：由于找不到 {model_filename}\n请返回命令行进行重新训练。"))
                    self.root.after(0, lambda: self.set_status(f"SYS_HALT", "err"))
                    self.root.after(0, lambda: self.log(f">> [SYS] 找不到权重对象路径: {self.model_path}"))
                    self.model = None
                    return

                if self.current_model_type in ["RESNET", "CBAM-RESNET"]:
                    self.model = model_class(num_classes=4, pretrained=False).to(self.device)
                elif self.current_model_type == "CNN-LSTM":
                    self.model = model_class(num_classes=4, num_slices=8).to(self.device)
                else:
                    self.model = model_class(num_classes=4).to(self.device)
                    
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                
                self.root.after(0, lambda: self.log(f">> [OK] 算法挂载完毕 (Engine Online @ {self.device})"))
                self.root.after(0, lambda: self.set_status("ONLINE", "success"))
                
                if self.current_file_path:
                    self.root.after(0, lambda: self.run_diagnosis(self.current_file_path))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log(f">> [ERROR] 后端执行异常: {str(e)}"))
                self.root.after(0, lambda: self.set_status("HALT", "err"))
                self.model = None

        threading.Thread(target=_load, daemon=True).start()

    # =================
    # 数据与文件驱动管理
    # =================
    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="请求目标信道矩阵",
            filetypes=[("MATLAB Records", "*.mat"), ("All Dump", "*.*")]
        )
        if file_path:
            self.current_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename, fg=THEME["primary"])
            fs = "48 kHz (High-Res)" if "48k" in file_path else "12 kHz (Standard)"
            self.card1_sub.config(text=f"Fs Sample Rate: {fs}")
            
            self.check_matrix_data(file_path)
            self.run_diagnosis(file_path)

    def run_diagnosis(self, file_path):
        if self.model is None:
            messagebox.showwarning("阻断性异常", "主引擎未拉起！不能执行推理操作。")
            return

        self.log(f"\n[JOB STARTED] 数据流导通 -> {os.path.basename(file_path)}")
        self.set_status("RUNNING", "warn")
        
        self.card2_val.config(text="Extracting ...", fg=THEME["warning"])
        self.card2_sub.config(text="正在将一维序列升维...")
        self.card3_val.config(text="-- %", fg=THEME["text_sub"])
        
        threading.Thread(target=self._process_file, args=(file_path,), daemon=True).start()

    def _process_file(self, file_path):
        try:
            result = self.preprocess_signal(file_path)
            
            if result[0] is None:
                self.root.after(0, lambda: self.set_status("DATA_CORRUPT", "err"))
                self.root.after(0, lambda: self.card2_val.config(text="解码被拒绝", fg=THEME["danger"]))
                self.root.after(0, lambda: self.card2_sub.config(text="源文件结构不适用。"))
                return
            
            img_tensor, f, t, Zxx = result

            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                
                label_idx = np.argmax(probs)
                confidence = probs[label_idx] * 100
                diagnosis = LABEL_MAP[label_idx]

            self.root.after(0, lambda: self._update_result_ui(diagnosis, confidence, probs, f, t, Zxx))

        except Exception as e:
            self.root.after(0, lambda: self.log(f">> [SYS ERROR] 调度栈溢出或异常抛出: {e}"))
            self.root.after(0, lambda: self.set_status("CRASH", "err"))

    def _update_result_ui(self, diagnosis, confidence, probs, f, t, Zxx):
        self.set_status("SUCCESS", "success")
        
        is_normal = "Normal" in diagnosis
        color = THEME["success"] if is_normal else THEME["danger"]
        
        en_desc = diagnosis.split(' (')[0]
        cn_desc = diagnosis.split('(')[1][:-1] if '(' in diagnosis else diagnosis
        
        self.card2_val.config(text=en_desc, fg=color)
        self.card2_sub.config(text=f"本地描述: {cn_desc}")
        
        self.card3_val.config(text=f"{confidence:.2f} %", fg=color)
        
        self.log(f">> [JOB DONE] 预测反馈: {diagnosis} (置信 {confidence:.2f}%)")
        self._plot_results(f, t, Zxx, probs)

    def preprocess_signal(self, file_path):
        try:
            mat_data = scipy.io.loadmat(file_path)
            
            key = None
            for k in mat_data.keys():
                if k.endswith('DE_time'): key = k; break
            if not key:
                for k in mat_data.keys():
                    if k.endswith('FE_time'): key = k; break
                    
            if not key:
                for k in mat_data.keys():
                    if k.startswith("_"): continue
                    data = mat_data[k]
                    if isinstance(data, np.ndarray) and data.ndim == 2:
                        return self._process_matrix_column(data, self.current_column)
                
                self.root.after(0, lambda: self.log(">> [WARN] 该 Mat 数据结构非标，不包含 DE_time 轨道"))
                return None, None, None, None
                
            signal = mat_data[key].flatten()
            fs = 48000 if '48k' in file_path else 12000
            
            plot_length = 120000 
            model_length = 1024
            
            if len(signal) < model_length:
                 self.root.after(0, lambda: self.log(f">> [WARN] 时域截面不足 (len: {len(signal)} < req: {model_length})"))
                 return None, None, None, None
            
            plot_signal = signal[:plot_length] if len(signal) > plot_length else signal
            f_plot, t_plot, Zxx_plot = scipy.signal.stft(plot_signal, fs=fs, nperseg=256, noverlap=128)
            
            model_signal = signal[:model_length]
            _, _, Zxx_model = scipy.signal.stft(model_signal, fs=fs, nperseg=64, noverlap=32)
            img_model = np.abs(Zxx_model)
            img_model = (img_model - img_model.min()) / (img_model.max() - img_model.min() + 1e-8)
            
            img_tensor = torch.from_numpy(img_model).float().unsqueeze(0).unsqueeze(0)
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            
            return img_tensor, f_plot, t_plot, Zxx_plot
            
        except Exception as e:
            self.root.after(0, lambda: self.log(f">> [FAIL] 管道 IO 读取损坏: {e}"))
            return None, None, None, None

    def check_matrix_data(self, file_path):
        try:
            mat_data = scipy.io.loadmat(file_path)
            for k in mat_data.keys():
                if k.startswith('_'): continue
                data = mat_data[k]
                if isinstance(data, np.ndarray) and data.ndim == 2:
                    rows, cols = data.shape
                    self.column_spinbox.config(state="normal", to=cols-1)
                    self.btn_change_col.config(state=tk.NORMAL)
                    self.current_column = 0
                    self.column_var.set("0")
                    self.log(f">> [INFO] 探测到多路复用矩阵 (Channels: {cols})")
                    return
                    
            self.column_spinbox.config(state="disabled")
            self.btn_change_col.config(state=tk.DISABLED)
            
        except Exception as e:
            self.column_spinbox.config(state="disabled")
            self.btn_change_col.config(state=tk.DISABLED)

    def change_column(self):
        try:
            self.current_column = int(self.column_var.get())
            self.log(f">> [SYS] IO通道路由重定至 Ch - {self.current_column}")
            if self.current_file_path:
                self.run_diagnosis(self.current_file_path)
        except ValueError:
            self.column_var.set(str(self.current_column))

    def _process_matrix_column(self, matrix_data, col_index):
        self.root.after(0, lambda: self.log(f">> [DATA] 分析张量第 {col_index} 通道中..."))
        try:
            signal = matrix_data[:, col_index].flatten()
            fs = 12000
            model_length = 1024
            
            if len(signal) < model_length:
                self.root.after(0, lambda: self.log(f">> [ERROR] 通道 {col_index} 有效载荷过低"))
                return None, None, None, None
            
            model_signal = signal[:model_length]
            _, _, Zxx_model = scipy.signal.stft(model_signal, fs=fs, nperseg=64, noverlap=32)
            img_model = np.abs(Zxx_model)
            img_model = (img_model - img_model.min()) / (img_model.max() - img_model.min() + 1e-8)
            
            img_tensor = torch.from_numpy(img_model).float().unsqueeze(0).unsqueeze(0)
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            
            plot_length = min(120000, len(signal))
            plot_signal = signal[:plot_length]
            f_plot, t_plot, Zxx_plot = scipy.signal.stft(plot_signal, fs=fs, nperseg=256, noverlap=128)
            
            return img_tensor, f_plot, t_plot, Zxx_plot
        except Exception as e:
            self.root.after(0, lambda: self.log(f">> [ERROR] 矩阵截取解包失败：{e}"))
            return None, None, None, None

if __name__ == '__main__':
    root = tk.Tk()
    app = CWRUDiagnosisApp(root)
    root.mainloop()
