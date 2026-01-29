# LSC镜头阴影校准工具 - 使用说明

## 快速开始

### 方式1: 使用GUI图形界面（推荐）

1. **直接运行**
   ```bash
   python lsc_gui.py
   ```

2. **填写参数**
   - **RAW文件**: 选择待校准的RAW图像文件
   - **图像宽度**: 图像宽度（像素）
   - **图像高度**: 图像高度（像素）
   - **位深度**: 8/10/12/14/16 bit
   - **RAW格式**:
     - `plain`: 普通RAW（无打包）
     - `mipi_raw10`: MIPI RAW10打包格式
     - `mipi_raw12`: MIPI RAW12打包格式
   - **Bayer模式**: RGGB/GRBG/GBRG/BGGR

3. **点击"开始校准"**
   - 程序会自动运行校准流程
   - 结果保存在输出目录

### 方式2: 使用命令行

```bash
python main.py
```

需要先在 `config.py` 中配置参数。

## 打包成EXE可执行文件

### 步骤1: 安装PyInstaller

```bash
pip install pyinstaller
```

### 步骤2: 打包

**方法A: 使用批处理脚本**
```bash
build_exe.bat
```

**方法B: 使用spec文件**
```bash
pyinstaller lsc_tool.spec
```

**方法C: 手动打包**
```bash
pyinstaller --onefile --windowed --name="LSC校准工具" lsc_gui.py
```

### 步骤3: 分发

打包完成后，在 `dist/` 目录会生成可执行文件。

**分发包应包含：**
```
LSC校准工具/
├── LSC校准工具.exe        # 主程序
├── config.py               # 配置文件
├── input/                  # 输入文件夹（放置RAW文件）
├── output/                 # 输出文件夹（自动生成）
└── 使用说明.txt
```

## RAW文件格式说明

### Plain RAW
- 无打包，每个像素直接存储
- 文件大小 = 宽度 × 高度 × (位深度/8)
- 例如：2904×2900，10-bit -> 约16.8MB（实际存储为16-bit）

### MIPI RAW10
- MIPI联盟标准打包格式
- 每4个10-bit像素打包成5字节
- 可能包含行对齐padding
- 文件大小约为：宽度 × 高度 × 1.25

### MIPI RAW12
- MIPI联盟标准打包格式
- 每2个12-bit像素打包成3字节
- 文件大小约为：宽度 × 高度 × 1.5

## Bayer Pattern识别

如果不确定Bayer模式，可以运行测试脚本：

```bash
python test_bayer_patterns.py
```

程序会生成4种Bayer模式的预览图，选择中心为绿色的那个。

## 输出文件说明

校准完成后，`output/` 目录包含：

### 图像文件 (`images/`)
- `*_1_original_no_wb.png` - 原始图像（无白平衡）
- `*_2_original_wb_only.png` - 仅白平衡
- `*_3_compensated_lsc_only.png` - 仅LSC校正
- `*_4_final_result_lsc_wb.png` - LSC+白平衡（最终结果）
- `*_4_final_result_lsc_wb_Gr_grid.png` - Gr通道网格标注图
- `*_4_final_result_lsc_wb_Gb_grid.png` - Gb通道网格标注图

### 增益表 (`gain_tables/`)
- `*_R_gain_table.txt` - R通道增益表（浮点）
- `*_Gr_gain_table.txt` - Gr通道增益表
- `*_Gb_gain_table.txt` - Gb通道增益表
- `*_B_gain_table.txt` - B通道增益表

### 高通格式 (`qcom_tables_Q10/`)
- `*_R_Q10.txt` - Q10定点格式（可直接用于高通ISP）
- `*_Gr_Q10.txt`
- `*_Gb_Q10.txt`
- `*_B_Q10.txt`

### 热力图 (`heatmaps/`)
- 各通道增益分布热力图

### 分析报告 (`visualizations/`)
- `*_1_image_comparison.png` - 校正前后对比
- `*_2_histogram_comparison.png` - 直方图对比

## 常见问题

### Q1: 提示"无法读取RAW文件"
**A:** 检查：
- 文件路径是否正确
- 宽度/高度参数是否匹配
- RAW格式是否选择正确

### Q2: 图像颜色不对
**A:** 可能是Bayer Pattern错误，运行 `test_bayer_patterns.py` 确认正确模式。

### Q3: 边缘出现暗块或亮块
**A:** 调整 `config.py` 中的边缘衰减参数：
```python
FISHEYE_GAIN_RADIUS_RATIO = 1.0  # 从圆边缘开始衰减
FISHEYE_DAMPING_WIDTH_PIXEL = 700  # 衰减过渡宽度
```

### Q4: 增益值过大或过小
**A:** 检查：
- 曝光是否充足（推荐：中心亮度60-80%）
- 光源是否均匀
- 调整 `MAX_GAIN` 参数（建议1.5-4.0）

### Q5: 打包的exe文件过大
**A:** 可以使用以下优化：
```bash
# 使用upx压缩
pyinstaller --onefile --upx-dir=upx lsc_gui.py

# 排除不必要的库
pyinstaller --onefile --exclude-module pandas lsc_gui.py
```

## 技术支持

- GitHub: https://github.com/alvin-liu1/lsc_calibration_project
- 问题反馈: 在GitHub提交Issue

## 更新日志

### v3.0 (2025-01)
- 新增GUI图形界面
- 支持MIPI RAW格式自动检测
- 优化边缘衰减算法
- 新增网格标注可视化
- 改进圆心选择交互

### v2.0
- 多区域均匀度评估
- V6.0径向拟合算法
- 高通ISP格式输出

### v1.0
- 基础LSC校准功能
