import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import torch
import scipy.io
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import importlib

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

# 标签映射
LABEL_MAP = {
    0: 'Normal (正常)',
    1: 'Inner Race Fault (内圈故障)',
    2: 'Outer Race Fault (外圈故障)',
    3: 'Ball Fault (滚动体故障)'
}

# 颜色配置 (Academic Minimalist)
COLORS = {
    "bg": "#f4f6f9",         # 背景：极浅的灰蓝色
    "fg": "#212529",         # 文字：深黑灰，减轻对比视觉疲劳
    "panel": "#ffffff",      # 面板：纯白
    "accent": "#e9ecef",     # 强调（按钮）：淡灰
    "highlight": "#0d6efd",  # 高亮：学术蓝
    "success": "#198754",    # 成功：稳重的暗绿
    "warning": "#fd7e14",    # 警告：暗橙
    "danger": "#dc3545",     # 危险：深红
    "border": "#dee2e6",     # 边框色
}


class CWRUDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("基于深度学习的 CWRU 轴承故障智能诊断系统 (Academic Edition)")
        self.root.geometry("1100x800")
        self.root.configure(bg=COLORS["bg"])
        
        # 状态变量
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_dir = script_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_model_type = "CNN"
        
        self.setup_styles()
        self.init_ui()
        self.load_model()

    def setup_styles(self):
        style = ttk.Style()
        # 基础主题
        try:
            style.theme_use('clam')
        except:
            pass
            
        style.configure(".", 
            background=COLORS["bg"], 
            foreground=COLORS["fg"], 
            font=('Microsoft YaHei UI', 11)
        )
        
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["fg"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["fg"])
        
        style.configure("TLabelframe", 
            background=COLORS["panel"], 
            foreground=COLORS["fg"],
            bordercolor=COLORS["border"],
            borderwidth=1
        )
        style.configure("TLabelframe.Label", 
            background=COLORS["panel"], 
            foreground=COLORS["highlight"], 
            font=('Microsoft YaHei UI', 12, 'bold')
        )
        
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        
        style.configure("TButton", 
            background=COLORS["accent"], 
            foreground=COLORS["fg"],
            font=('Microsoft YaHei UI', 11),
            relief="flat",
            borderwidth=1,
            bordercolor=COLORS["border"]
        )
        style.map("TButton",
            background=[('active', '#ced4da'), ('pressed', '#adb5bd')]
        )

        style.configure("TCombobox", 
            fieldbackground="#ffffff", 
            background="#ffffff",
            foreground="black",
            darkcolor="#ffffff",
            lightcolor="#cccccc"
        )
        
        self.root.option_add("*TCombobox*Listbox.background", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.foreground", "black")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#eeeeee")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "black")

    def init_ui(self):
        # --- 顶部控制面板 ---
        control_frame = ttk.LabelFrame(self.root, text=" 控制面板 (Control Panel) ", padding=15)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        # 模型选择
        model_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        model_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(model_frame, text="算法模型:", style="Panel.TLabel", font=('', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="SimpleCNN (CNN)")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                        values=["SimpleCNN (CNN)", 
                                                "ResNet18", 
                                                "CBAM-ResNet18 (时空注意力)", 
                                                "CNN-LSTM (时频特征序列)"], 
                                        state="readonly", width=22)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self.load_model())
        
        # 文件操作
        file_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        file_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(file_frame, text="选择数据文件 (.mat)", command=self.open_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="未选择文件", style="Panel.TLabel")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # 矩阵列选择
        col_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        col_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(col_frame, text="矩阵列索引:", style="Panel.TLabel").pack(side=tk.LEFT, padx=5)
        self.column_var = tk.StringVar(value="0")
        self.column_spinbox = ttk.Spinbox(col_frame, from_=0, to=0, textvariable=self.column_var, width=6, state="disabled")
        self.column_spinbox.pack(side=tk.LEFT, padx=5)
        self.btn_change_col = ttk.Button(col_frame, text="切换", command=self.change_column, state=tk.DISABLED)
        self.btn_change_col.pack(side=tk.LEFT, padx=5)
        
        self.current_column = 0
        self.current_file_path = None
        
        ttk.Button(control_frame, text="重新加载模型", command=self.load_model).pack(side=tk.RIGHT, padx=5)
        
        # --- 中间显示区域 ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # 左侧：诊断结果和日志
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, pady=5)
        
        # 结果卡片
        result_card = ttk.LabelFrame(left_frame, text=" 诊断结果 (Diagnosis Result) ", padding=20)
        result_card.pack(side=tk.TOP, fill=tk.X, expand=False, pady=(0, 10))
        
        self.result_title = tk.Label(result_card, text="等待诊断...", font=("Microsoft YaHei UI", 19, "bold"), 
                                     bg=COLORS["panel"], fg=COLORS["fg"], wraplength=250, justify="center")
        self.result_title.pack(pady=(10, 5))
        
        self.conf_label = tk.Label(result_card, text="--", font=("Microsoft YaHei UI", 25, "bold"), 
                                   bg=COLORS["panel"], fg=COLORS["accent"])
        self.conf_label.pack(pady=(0, 10))
        
        self.detail_label = ttk.Label(result_card, text="请选择包含轴承振动信号的 .mat 文件", 
                                      style="Panel.TLabel", justify=tk.CENTER)
        self.detail_label.pack(pady=5)
        
        # 日志区域
        log_frame = ttk.LabelFrame(left_frame, text=" 运行日志 (Execution Log) ", padding=10)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(log_frame, height=12, width=35, 
                                 bg="#ffffff", fg="#333333", 
                                 insertbackground="black",
                                 font=("Consolas", 9), borderwidth=0, highlightthickness=1, 
                                 highlightcolor="#cccccc", highlightbackground="#eeeeee")
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 右侧：可视化图表
        right_frame = ttk.LabelFrame(main_frame, text=" 时频特征与概率分析 (STFT & Probability) ", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0), pady=5)
        
        self.figure_frame = ttk.Frame(right_frame, style="Panel.TFrame")
        self.figure_frame.pack(fill=tk.BOTH, expand=True)
        
        # 空白背景图
        self._init_empty_plot()
        
        # 底部状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("[状态] 就绪 (Ready)")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                              bg=COLORS["accent"], fg=COLORS["fg"], 
                              anchor=tk.W, font=("Microsoft YaHei UI", 10), padx=10, pady=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def log(self, message):
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)

    def set_status(self, msg, level="info"):
        prefix = "[成功] " if level=="success" else "[警告] " if level=="warning" else "[错误] " if level=="err" else "[信息] "
        self.status_var.set(prefix + msg)

    def _init_empty_plot(self):
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100, facecolor=COLORS["panel"])
        fig.patch.set_facecolor(COLORS["panel"])
        
        for ax in [ax1, ax2]:
            ax.set_facecolor(COLORS["bg"])
            ax.tick_params(colors='#666666')
            for spine in ax.spines.values():
                spine.set_color('#dddddd')
                
        ax1.set_title("STFT Magnitude", color="#000000")
        ax1.text(0.5, 0.5, "Waiting for data", ha='center', va='center', color='#999999')
        
        ax2.set_title("Class Probabilities", color="#000000")
        for spine in ax2.spines.values():
            spine.set_color("#cccccc")
        ax2.tick_params(axis='both', colors='#666666')
        ax2.set_xlim(-0.5, 3.5)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['Normal', 'Inner', 'Outer', 'Ball'], rotation=15)
        
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_model(self):
        selection = self.model_var.get()
        self.set_status("正在加载模型...")
        self.log(f"====================\n请求加载: {selection}")
        
        # 确定模型路径和类
        if "CNN-LSTM" in selection:
            self.current_model_type = "CNN-LSTM"
            model_filename = "cwru_cnn_lstm_model.pth"
            model_class = STFT_CNN_LSTM
            model_dir = os.path.join(os.path.dirname(self.script_dir), '双分支 CNN-LSTM 时频特征提取架构')
            if STFT_CNN_LSTM is None:
                self.log("❌ 缺少 cnn_lstm_model.py 或依赖错误")
                self.set_status("依赖错误", "err")
                return
        elif "CBAM" in selection:
            self.current_model_type = "CBAM-RESNET"
            model_filename = "cwru_resnet_cbam_model.pth"
            model_class = BearingResNet18_CBAM
            model_dir = os.path.join(os.path.dirname(self.script_dir), '时空注意力机制加强版')
            if BearingResNet18_CBAM is None:
                self.log("❌ 缺少 cbam_model.py 或依赖错误")
                self.set_status("依赖错误", "err")
                return
        elif "ResNet" in selection:
            self.current_model_type = "RESNET"
            model_filename = "cwru_resnet_model.pth"
            model_class = BearingResNet18
            model_dir = self.script_dir
            if BearingResNet18 is None:
                self.log("❌ 缺少 resnet_model.py 或依赖错误")
                self.set_status("依赖错误", "err")
                return
        else:
            self.current_model_type = "CNN"
            model_filename = "cwru_cnn_model.pth"
            model_class = SimpleCNN
            model_dir = self.script_dir
            if SimpleCNN is None:
                self.log("❌ 缺少 model.py 或依赖错误")
                self.set_status("依赖错误", "err")
                return
                
        self.model_path = os.path.join(model_dir, model_filename)

        def _load():
            try:
                if not os.path.exists(self.model_path):
                    self.root.after(0, lambda: messagebox.showwarning("模型缺失", f"未找到模型文件:\n{self.model_path}\n\n请先运行对应的训练脚本 (train.py 或 train_resnet.py)。"))
                    self.root.after(0, lambda: self.set_status(f"模型文件丢失: {model_filename}", "err"))
                    self.root.after(0, lambda: self.log(f"❌ 找不到 {model_filename}"))
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
                
                self.root.after(0, lambda: self.log(f"✅ 模型加载成功! 设备: {self.device}"))
                self.root.after(0, lambda: self.set_status(f"就绪 ({self.current_model_type})", "success"))
                
                # 如果当前有文件，用新模型重新诊断
                if self.current_file_path:
                    self.root.after(0, lambda: self.run_diagnosis(self.current_file_path))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ 模型加载出错: {e}"))
                self.root.after(0, lambda: self.set_status("模型加载错误", "err"))
                self.model = None

        threading.Thread(target=_load, daemon=True).start()

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="选择振动数据文件",
            filetypes=[("MATLAB Files", "*.mat"), ("All Files", "*.*")]
        )
        if file_path:
            self.current_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename)
            self.check_matrix_data(file_path)
            self.run_diagnosis(file_path)

    def run_diagnosis(self, file_path):
        if self.model is None:
            messagebox.showwarning("警告", "模型未成功加载，无法诊断。")
            return

        self.log(f"--------------------\n▶ 开始分析: {os.path.basename(file_path)}")
        self.set_status("正在计算...", "info")
        
        self.result_title.config(text="分析中...", fg=COLORS["warning"])
        self.conf_label.config(text="--", fg=COLORS["fg"])
        self.detail_label.config(text="提取特征并推理中...")
        
        threading.Thread(target=self._process_file, args=(file_path,), daemon=True).start()

    def _process_file(self, file_path):
        try:
            result = self.preprocess_signal(file_path)
            
            if result[0] is None:
                self.root.after(0, lambda: self.set_status("数据预处理失败", "err"))
                self.root.after(0, lambda: self.result_title.config(text="处理失败", fg=COLORS["danger"]))
                self.root.after(0, lambda: self.detail_label.config(text="请检查文件格式或选择正确的数据列"))
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
            self.root.after(0, lambda: self.log(f"❌ 诊断过程出错: {e}"))
            self.root.after(0, lambda: self.set_status("诊断抛出异常", "err"))

    def _update_result_ui(self, diagnosis, confidence, probs, f, t, Zxx):
        self.set_status(f"完成 ({self.current_model_type})", "success")
        
        is_normal = "Normal" in diagnosis
        color = COLORS["success"] if is_normal else COLORS["danger"]
        
        # 更新左侧卡片
        self.result_title.config(text=diagnosis.split(' (')[0], fg=color)
        self.conf_label.config(text=f"{confidence:.1f}%", fg=color)
        
        cn_desc = diagnosis.split('(')[1][:-1] if '(' in diagnosis else diagnosis
        self.detail_label.config(text=f"检测结论: {cn_desc}")
        
        self.log(f"✓ 结果: {diagnosis}\n✓ 置信度: {confidence:.2f}%")
        
        # 更新图表
        self._plot_results(f, t, Zxx, probs)

    def _plot_results(self, f, t, Zxx, probs):
        for widget in self.figure_frame.winfo_children():
            widget.destroy()

        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
        fig = plt.Figure(figsize=(8, 4.5), dpi=100, facecolor='#ffffff')
        
        # 左侧：STFT 图
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor("#ffffff")
        c = ax1.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
        ax1.set_title('STFT Magnitude', color='#000000')
        ax1.set_ylabel('Frequency [Hz]', color='#333333')
        ax1.set_xlabel('Time [sec]', color='#333333')
        cb = fig.colorbar(c, ax=ax1, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8, colors='#666666')
        
        # 右侧：概率分布图
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor("#ffffff")
        
        classes = ['Normal', 'Inner', 'Outer', 'Ball']
        x = np.arange(len(classes))
        
        # 所有柱子默认为白色背景配黑框，最大的一根加深颜色
        colors = ['#f8f9fa'] * len(classes)
        edgecolors = ['#212529'] * len(classes)
        max_idx = np.argmax(probs)
        colors[max_idx] = '#0d6efd' # 突出的类别用学术蓝
        
        bars = ax2.bar(x, probs * 100, color=colors, alpha=0.9, edgecolor='#212529', linewidth=1.2)
        ax2.set_title('Confidence Distribution (%)', color='#000000')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=15, color='#000000')
        ax2.set_ylim(0, 105)
        
        for spine in ax2.spines.values():
            spine.set_color('#000000')
            
        # 给柱状图加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 1: 
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, color='#000000', fontweight='bold')
        
        fig.tight_layout(pad=2.0)
        
        canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


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
                
                self.root.after(0, lambda: self.log("❌ 错误：未在文件中找到有效的振动信号 (DE_time/FE_time) 或矩阵数据"))
                return None, None, None, None
                
            signal = mat_data[key].flatten()
            fs = 48000 if '48k' in file_path else 12000
            
            plot_length = 120000 
            model_length = 1024
            
            if len(signal) < model_length:
                 self.root.after(0, lambda: self.log(f"❌ 信号长度 {len(signal)} 不足 {model_length}"))
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
            self.root.after(0, lambda: self.log(f"❌ 预处理出错: {e}"))
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
                    self.log(f"ℹ️ 检测到矩阵数据：{rows}行 x {cols}列")
                    return
                    
            # 不是矩阵数据，禁用选择
            self.column_spinbox.config(state="disabled")
            self.btn_change_col.config(state=tk.DISABLED)
            
        except Exception as e:
            self.column_spinbox.config(state="disabled")
            self.btn_change_col.config(state=tk.DISABLED)

    def change_column(self):
        try:
            self.current_column = int(self.column_var.get())
            self.log(f"📌 切换到第 {self.current_column} 列")
            if self.current_file_path:
                self.run_diagnosis(self.current_file_path)
        except ValueError:
            self.column_var.set(str(self.current_column))

    def _process_matrix_column(self, matrix_data, col_index):
        self.root.after(0, lambda: self.log(f"ℹ️ 处理矩阵列 {col_index}，形状: {matrix_data.shape}"))
        try:
            signal = matrix_data[:, col_index].flatten()
            fs = 12000
            model_length = 1024
            
            if len(signal) < model_length:
                self.root.after(0, lambda: self.log(f"❌ 信号长度 {len(signal)} 不足 {model_length}"))
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
            self.root.after(0, lambda: self.log(f"❌ 处理矩阵列数据出错：{e}"))
            return None, None, None, None

if __name__ == '__main__':
    root = tk.Tk()
    app = CWRUDiagnosisApp(root)
    root.mainloop()
