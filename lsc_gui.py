#!/usr/bin/env python3
"""
LSC校准工具 - GUI版本
支持可视化配置RAW文件参数和校准设置
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import json
import subprocess
from pathlib import Path

class LSCCalibrationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LSC镜头阴影校准工具 v3.0")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        # 配置文件路径
        self.config_file = "lsc_gui_config.json"

        # 加载上次的配置
        self.load_config()

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        """创建GUI界面"""

        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(main_frame, text="LSC镜头阴影校准工具",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # ========== RAW文件设置 ==========
        raw_frame = ttk.LabelFrame(main_frame, text="RAW文件设置", padding="10")
        raw_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # 文件路径
        ttk.Label(raw_frame, text="RAW文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.raw_file_var = tk.StringVar(value=self.config.get('raw_file', ''))
        raw_entry = ttk.Entry(raw_frame, textvariable=self.raw_file_var, width=50)
        raw_entry.grid(row=0, column=1, padx=5)
        ttk.Button(raw_frame, text="浏览", command=self.browse_raw_file).grid(row=0, column=2)

        # 图像宽度
        ttk.Label(raw_frame, text="图像宽度:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.width_var = tk.StringVar(value=self.config.get('width', '2900'))
        ttk.Entry(raw_frame, textvariable=self.width_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=5)

        # 图像高度
        ttk.Label(raw_frame, text="图像高度:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.height_var = tk.StringVar(value=self.config.get('height', '2900'))
        ttk.Entry(raw_frame, textvariable=self.height_var, width=20).grid(row=2, column=1, sticky=tk.W, padx=5)

        # 位深度
        ttk.Label(raw_frame, text="位深度:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.bit_depth_var = tk.StringVar(value=self.config.get('bit_depth', '10'))
        bit_depth_combo = ttk.Combobox(raw_frame, textvariable=self.bit_depth_var,
                                        values=['8', '10', '12', '14', '16'], width=18, state='readonly')
        bit_depth_combo.grid(row=3, column=1, sticky=tk.W, padx=5)

        # RAW格式
        ttk.Label(raw_frame, text="RAW格式:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.raw_format_var = tk.StringVar(value=self.config.get('raw_format', 'mipi_raw10'))
        format_combo = ttk.Combobox(raw_frame, textvariable=self.raw_format_var,
                                    values=['plain', 'mipi_raw10', 'mipi_raw12'],
                                    width=18, state='readonly')
        format_combo.grid(row=4, column=1, sticky=tk.W, padx=5)

        # Bayer Pattern
        ttk.Label(raw_frame, text="Bayer模式:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.bayer_var = tk.StringVar(value=self.config.get('bayer_pattern', 'GRBG'))
        bayer_combo = ttk.Combobox(raw_frame, textvariable=self.bayer_var,
                                   values=['RGGB', 'GRBG', 'GBRG', 'BGGR'],
                                   width=18, state='readonly')
        bayer_combo.grid(row=5, column=1, sticky=tk.W, padx=5)

        # ========== 校准参数设置 ==========
        calib_frame = ttk.LabelFrame(main_frame, text="校准参数", padding="10")
        calib_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # 网格尺寸
        ttk.Label(calib_frame, text="网格行数:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.grid_rows_var = tk.StringVar(value=self.config.get('grid_rows', '12'))
        ttk.Entry(calib_frame, textvariable=self.grid_rows_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(calib_frame, text="网格列数:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.grid_cols_var = tk.StringVar(value=self.config.get('grid_cols', '16'))
        ttk.Entry(calib_frame, textvariable=self.grid_cols_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=5)

        # 最大增益
        ttk.Label(calib_frame, text="最大增益:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_gain_var = tk.StringVar(value=self.config.get('max_gain', '4.0'))
        ttk.Entry(calib_frame, textvariable=self.max_gain_var, width=20).grid(row=2, column=1, sticky=tk.W, padx=5)

        # 应用对称化
        self.symmetry_var = tk.BooleanVar(value=self.config.get('apply_symmetry', True))
        ttk.Checkbutton(calib_frame, text="应用对称化处理",
                       variable=self.symmetry_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)

        # ========== 输出设置 ==========
        output_frame = ttk.LabelFrame(main_frame, text="输出设置", padding="10")
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # 输出目录
        ttk.Label(output_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value=self.config.get('output_dir', 'output'))
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50)
        output_entry.grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="浏览", command=self.browse_output_dir).grid(row=0, column=2)

        # ========== 操作按钮 ==========
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)

        self.run_button = ttk.Button(button_frame, text="开始校准",
                                     command=self.run_calibration, width=15)
        self.run_button.grid(row=0, column=0, padx=5)

        ttk.Button(button_frame, text="保存配置",
                  command=self.save_config, width=15).grid(row=0, column=1, padx=5)

        ttk.Button(button_frame, text="退出",
                  command=self.root.quit, width=15).grid(row=0, column=2, padx=5)

        # ========== 状态栏 ==========
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

    def browse_raw_file(self):
        """浏览RAW文件"""
        filename = filedialog.askopenfilename(
            title="选择RAW文件",
            filetypes=[("RAW文件", "*.raw"), ("所有文件", "*.*")]
        )
        if filename:
            self.raw_file_var.set(filename)
            # 尝试从文件名推断参数
            basename = os.path.basename(filename)
            if 'mipi' in basename.lower():
                self.raw_format_var.set('mipi_raw10')

    def browse_output_dir(self):
        """浏览输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir_var.set(dirname)

    def load_config(self):
        """加载配置文件"""
        self.config = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except:
                pass

    def save_config(self):
        """保存配置文件"""
        config = {
            'raw_file': self.raw_file_var.get(),
            'width': self.width_var.get(),
            'height': self.height_var.get(),
            'bit_depth': self.bit_depth_var.get(),
            'raw_format': self.raw_format_var.get(),
            'bayer_pattern': self.bayer_var.get(),
            'grid_rows': self.grid_rows_var.get(),
            'grid_cols': self.grid_cols_var.get(),
            'max_gain': self.max_gain_var.get(),
            'apply_symmetry': self.symmetry_var.get(),
            'output_dir': self.output_dir_var.get()
        }

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", "配置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")

    def validate_inputs(self):
        """验证输入参数"""
        if not self.raw_file_var.get():
            messagebox.showerror("错误", "请选择RAW文件")
            return False

        if not os.path.exists(self.raw_file_var.get()):
            messagebox.showerror("错误", "RAW文件不存在")
            return False

        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            if width <= 0 or height <= 0:
                raise ValueError()
        except:
            messagebox.showerror("错误", "图像宽度和高度必须是正整数")
            return False

        try:
            max_gain = float(self.max_gain_var.get())
            if max_gain <= 1.0 or max_gain > 8.0:
                raise ValueError()
        except:
            messagebox.showerror("错误", "最大增益必须在1.0-8.0之间")
            return False

        return True

    def run_calibration(self):
        """运行校准程序"""
        if not self.validate_inputs():
            return

        # 保存当前配置
        self.save_config()

        # 更新config.py文件
        self.update_config_file()

        # 准备输入文件
        raw_file = self.raw_file_var.get()
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
        os.makedirs(input_dir, exist_ok=True)

        # 复制或链接到input目录
        target_file = os.path.join(input_dir, os.path.basename(raw_file))
        if os.path.abspath(raw_file) != os.path.abspath(target_file):
            import shutil
            shutil.copy2(raw_file, target_file)

        # 禁用运行按钮
        self.run_button.config(state='disabled')
        self.status_var.set("正在运行校准程序...")
        self.root.update()

        try:
            # 运行main.py
            result = subprocess.run(
                [sys.executable, 'main.py'],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if result.returncode == 0:
                self.status_var.set("校准完成！")
                messagebox.showinfo("成功",
                    f"校准完成！\n\n输出文件保存在: {self.output_dir_var.get()}")
            else:
                self.status_var.set("校准失败")
                messagebox.showerror("错误",
                    f"校准失败:\n\n{result.stderr[:500]}")

        except Exception as e:
            self.status_var.set("发生错误")
            messagebox.showerror("错误", f"运行失败: {e}")

        finally:
            self.run_button.config(state='normal')

    def update_config_file(self):
        """更新config.py文件中的参数"""
        config_py = 'config.py'

        try:
            with open(config_py, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 更新参数
            updates = {
                'IMAGE_WIDTH': self.width_var.get(),
                'IMAGE_HEIGHT': self.height_var.get(),
                'BIT_DEPTH': self.bit_depth_var.get(),
                'RAW_FORMAT': f"'{self.raw_format_var.get()}'",
                'BAYER_PATTERN': f"'{self.bayer_var.get()}'",
                'GRID_ROWS': self.grid_rows_var.get(),
                'GRID_COLS': self.grid_cols_var.get(),
                'MAX_GAIN': self.max_gain_var.get(),
                'APPLY_SYMMETRY': str(self.symmetry_var.get()),
                'OUTPUT_DIR': f"'{self.output_dir_var.get()}'"
            }

            new_lines = []
            for line in lines:
                updated = False
                for key, value in updates.items():
                    if line.strip().startswith(f'{key} ='):
                        new_lines.append(f'{key} = {value}\n')
                        updated = True
                        break
                if not updated:
                    new_lines.append(line)

            with open(config_py, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        except Exception as e:
            messagebox.showwarning("警告", f"更新配置文件失败: {e}")

def main():
    root = tk.Tk()
    app = LSCCalibrationGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
