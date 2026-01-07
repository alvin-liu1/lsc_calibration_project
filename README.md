# LSC Calibration Project for Fisheye Lenses

[English](#english) | [中文](#中文)

---

## English

### Overview

This project provides a professional **Lens Shading Correction (LSC)** calibration tool specifically designed for fisheye lenses and optimized for Qualcomm ISP pipelines. It implements an innovative **radial polynomial fitting algorithm** that generates smooth, artifact-free gain tables for RGGB Bayer sensors.

### Key Features

- **Advanced Radial Polynomial Fitting** - Novel algorithm that fits lens vignetting to smooth polynomial curves, eliminating grid artifacts and color fringing
- **Fisheye Lens Optimization** - Specialized geometric damping protection for circular fisheye masks
- **Qualcomm ISP Ready** - Direct output in 13uQ10 format compatible with Chromatix tuning tools
- **Bayer Domain Processing** - Applies LSC before demosaicing with independent per-channel black level correction
- **Comprehensive Visualization** - Heatmaps, analysis plots, and before/after comparisons
- **Interactive Mask Selection** - Manual circular mask selection with fine-tuning controls

### Algorithm Innovation

Unlike traditional LSC methods that directly use noisy grid statistics, this project employs:

1. **Radial Polynomial Fit (4th order)** - Fits brightness falloff to optical physics principles
2. **Adaptive Data Filtering** - Uses only high SNR data (r < 0.92) to prevent edge artifacts
3. **Geometric Damping** - Smoothly transitions invalid regions to unity gain, preventing noise amplification
4. **Hard Mask Protection** - Zeroes out-of-circle pixels in Bayer domain to prevent demosaic contamination

### Technical Specifications

- **Sensor Support**: 10-bit/12-bit Bayer sensors (RGGB, BGGR, GRBG, GBRG)
- **Grid Configuration**: 17×13 vertices (16×12 cells) - Qualcomm VFE standard
- **Gain Range**: 1.0 to 7.99 (hardware limited by 13uQ10 format)
- **Output Formats**:
  - Floating-point gain tables (.txt)
  - Qualcomm 13uQ10 format (.txt)
  - Gain heatmaps (PNG)
  - Analysis visualizations (PNG)

### Project Structure

```
lsc_calibration_project/
├── config.py                 # Global configuration parameters
├── main.py                   # Main calibration pipeline
├── requirements.txt          # Python dependencies
├── lsc/                      # Core algorithm modules
│   ├── calibration.py       # LSC gain calculation engine (radial fitting)
│   ├── bayer_utils.py       # Bayer pattern processing utilities
│   ├── gain_utils.py        # Gain matrix post-processing (extrapolation, smoothing)
│   ├── image_utils.py       # White balance and image utilities
│   └── visualization.py     # Visualization and I/O functions
├── analyze_shading.py       # Shading uniformity analysis tool
├── invert.py                # Gain table inversion utility
├── input/                   # RAW image input directory
└── output/                  # Calibration results
    ├── images/              # Corrected images
    ├── gain_tables/         # Floating-point gain tables
    ├── qcom_tables_Q10/     # Qualcomm Q10 format tables
    ├── heatmaps/            # Gain heatmaps
    ├── visualizations/      # Analysis plots
    └── logs/                # Execution logs
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lsc_calibration_project.git
cd lsc_calibration_project

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

1. **Prepare RAW Image**
   - Capture a uniformly lit grayscale chart with your fisheye lens
   - Place 10-bit/12-bit RAW Bayer file in `input/` directory
   - Recommended: Use D65 illumination, avoid overexposure

2. **Configure Parameters**
   - Edit `config.py`:
     ```python
     RAW_IMAGE_PATH = 'input/your_raw_file.raw'
     IMAGE_WIDTH = 2904
     IMAGE_HEIGHT = 2900
     BAYER_PATTERN = cv2.COLOR_BayerRG2BGR_VNG  # RGGB
     BLACK_LEVELS = {'R': 64, 'Gr': 64, 'Gb': 64, 'B': 64}
     ```

3. **Run Calibration**
   ```bash
   python main.py
   ```

4. **Interactive Mask Selection**
   - Click center of fisheye circle
   - Adjust radius with scroll wheel
   - Fine-tune with keyboard:
     - `W/A/S/D` - Move center
     - `Z/X` - Adjust radius
     - `Enter` - Confirm
     - `ESC` - Cancel

5. **Check Results**
   - Corrected images: `output/images/`
   - Gain tables: `output/qcom_tables_Q10/`
   - Heatmaps: `output/heatmaps/`
   - Analysis: `output/visualizations/`

### Configuration Parameters

**Core Algorithm Parameters** (`config.py`):

```python
# LSC Grid Configuration
GRID_ROWS = 12              # 13 vertices (standard Qualcomm config)
GRID_COLS = 16              # 17 vertices

# Maximum Gain
MAX_GAIN = 12.0             # Allows fitting to darkest corners (will be clipped to 7.99)

# Fisheye Protection
FISHEYE_GAIN_RADIUS_RATIO = 1.05    # Start damping at 105% of circle radius
FISHEYE_DAMPING_WIDTH_PIXEL = 50    # Damping transition width

# Smoothing
V3_POST_SMOOTH_KSIZE = 5    # Post-processing Gaussian kernel size
APPLY_SYMMETRY = True       # Force radial symmetry
```

### ISP Pipeline

This tool follows industry-standard ISP pipeline order:

```
RAW Bayer → Black Level Correction → LSC Correction → Demosaic → White Balance
```

**Key Implementation Details**:
- Independent BLC for R/Gr/Gb/B channels
- LSC applied in Bayer domain (before demosaic)
- Hard mask applied after LSC to prevent demosaic contamination
- Bicubic interpolation for full-size gain map generation

### Advanced Usage

**Analyze Shading Uniformity**:
```bash
python analyze_shading.py
```
- Generates radial brightness curves
- Calculates R/G and B/G color ratio distributions
- Quantitative uniformity metrics

**Invert Gain Table**:
```bash
python invert.py
```
- Computes inverse gain table (1024 / original_gain)
- Useful for reverse correction or verification

### Algorithm Technical Details

**Radial Polynomial Fitting (V5.1)**:

1. Convert grid coordinates to normalized radius: `r_norm = distance / radius`
2. Filter valid data: `brightness > 0.001 AND r_norm < 0.92`
3. Fit 4th-order polynomial: `fitted_brightness = poly4(r_norm)`
4. Calculate gain: `gain = max_brightness / fitted_brightness`
5. Apply geometric damping to out-of-circle regions
6. Clip to hardware limits [1.0, 7.99]

**Why Polynomial Fitting?**:
- Traditional grid statistics produce noisy, stripe-prone gain tables
- Polynomial fit guarantees mathematical smoothness
- Conforms to optical vignetting physics (radially symmetric falloff)
- Automatically eliminates grid artifacts and color fringing

### Troubleshooting

**Issue**: "RAW文件尺寸不匹配"
- **Solution**: Verify `IMAGE_WIDTH` and `IMAGE_HEIGHT` in `config.py` match your RAW file

**Issue**: Edge color fringing after LSC
- **Solution**: Increase `FISHEYE_DAMPING_WIDTH_PIXEL` for smoother transition

**Issue**: Insufficient correction in corners
- **Solution**: Increase `MAX_GAIN` in `config.py`

**Issue**: Over-correction artifacts
- **Solution**: Check black levels are correctly configured

### Performance Considerations

- **Processing Time**: ~10-30 seconds for 3MP images (CPU: Intel i7)
- **Memory Usage**: ~500MB peak for 3MP RAW processing
- **Optimization Opportunities**:
  - Grid statistics loop can be vectorized
  - Polynomial fitting can be parallelized across channels
  - Consider Numba JIT or GPU acceleration for production use

### Limitations

- **Assumes Circular Fisheye**: Not suitable for elliptical or rectangular sensors without modification
- **Single Color Temperature**: Calibration is specific to capture illumination (typically D65)
- **No Dead Pixel Correction**: DPC should be implemented separately in production pipeline
- **No Noise Model**: LSC amplifies dark region noise; consider pairing with NR algorithms

### Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Citation

If you use this project in your research or product, please cite:

```bibtex
@software{lsc_calibration_fisheye,
  title={LSC Calibration Tool for Fisheye Lenses with Radial Polynomial Fitting},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/lsc_calibration_project}
}
```

### License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 中文

### 项目简介

这是一个专为**鱼眼镜头**设计的专业级**镜头阴影校正（LSC）**标定工具，针对高通ISP流水线优化。它采用创新的**径向多项式拟合算法**，为RGGB Bayer传感器生成平滑、无伪影的增益表。

### 核心特性

- **先进的径向多项式拟合** - 创新算法将镜头渐晕拟合为平滑多项式曲线，消除网格伪影和色边
- **鱼眼镜头优化** - 针对圆形鱼眼遮罩的专用几何衰减保护
- **高通ISP就绪** - 直接输出与Chromatix调优工具兼容的13uQ10格式
- **Bayer域处理** - 在去马赛克之前应用LSC，独立通道黑电平校正
- **全面可视化** - 热力图、分析图表、前后对比
- **交互式遮罩选择** - 手动圆形遮罩选择，带微调控制

### 算法创新

不同于传统LSC方法直接使用噪声网格统计，本项目采用：

1. **径向多项式拟合（4阶）** - 将亮度衰减拟合到光学物理原理
2. **自适应数据筛选** - 仅使用高信噪比数据（r < 0.92）防止边缘伪影
3. **几何衰减** - 将无效区域平滑过渡到单位增益，防止噪声放大
4. **硬遮罩保护** - 在Bayer域清零圆外像素，防止去马赛克污染

### 技术规格

- **传感器支持**: 10位/12位 Bayer传感器（RGGB、BGGR、GRBG、GBRG）
- **网格配置**: 17×13顶点（16×12单元格）- 高通VFE标准
- **增益范围**: 1.0至7.99（受13uQ10格式硬件限制）
- **输出格式**:
  - 浮点增益表（.txt）
  - 高通13uQ10格式（.txt）
  - 增益热力图（PNG）
  - 分析可视化图（PNG）

### 快速开始

1. **准备RAW图像**
   - 使用鱼眼镜头拍摄均匀照明的灰度卡
   - 将10位/12位RAW Bayer文件放入`input/`目录
   - 推荐：使用D65照明，避免过曝

2. **配置参数**
   - 编辑`config.py`：
     ```python
     RAW_IMAGE_PATH = 'input/your_raw_file.raw'
     IMAGE_WIDTH = 2904
     IMAGE_HEIGHT = 2900
     BAYER_PATTERN = cv2.COLOR_BayerRG2BGR_VNG  # RGGB
     BLACK_LEVELS = {'R': 64, 'Gr': 64, 'Gb': 64, 'B': 64}
     ```

3. **运行标定**
   ```bash
   python main.py
   ```

4. **交互式遮罩选择**
   - 点击鱼眼圆心
   - 用滚轮调整半径
   - 键盘微调：
     - `W/A/S/D` - 移动圆心
     - `Z/X` - 调整半径
     - `Enter` - 确认
     - `ESC` - 取消

5. **查看结果**
   - 校正图像：`output/images/`
   - 增益表：`output/qcom_tables_Q10/`
   - 热力图：`output/heatmaps/`
   - 分析图：`output/visualizations/`

### ISP流水线

本工具遵循业界标准ISP流水线顺序：

```
RAW Bayer → 黑电平校正 → LSC校正 → 去马赛克 → 白平衡
```

### 算法技术细节

**径向多项式拟合（V5.1）**：

1. 将网格坐标转换为归一化半径：`r_norm = distance / radius`
2. 筛选有效数据：`brightness > 0.001 且 r_norm < 0.92`
3. 拟合4阶多项式：`fitted_brightness = poly4(r_norm)`
4. 计算增益：`gain = max_brightness / fitted_brightness`
5. 对圆外区域应用几何衰减
6. 钳位到硬件限制 [1.0, 7.99]

**为什么用多项式拟合？**：
- 传统网格统计产生噪声大、易产生条纹的增益表
- 多项式拟合保证数学级平滑
- 符合光学渐晕物理（径向对称衰减）
- 自动消除网格伪影和色边

### 常见问题

**问题**："RAW文件尺寸不匹配"
- **解决**：验证`config.py`中的`IMAGE_WIDTH`和`IMAGE_HEIGHT`与RAW文件匹配

**问题**：LSC后边缘有色边
- **解决**：增加`FISHEYE_DAMPING_WIDTH_PIXEL`以实现更平滑过渡

**问题**：暗角校正不足
- **解决**：增加`config.py`中的`MAX_GAIN`

**问题**：过度校正伪影
- **解决**：检查黑电平是否正确配置

### 性能考虑

- **处理时间**：3MP图像约10-30秒（CPU: Intel i7）
- **内存使用**：3MP RAW处理峰值约500MB
- **优化机会**：
  - 网格统计循环可向量化
  - 多项式拟合可跨通道并行化
  - 生产环境考虑Numba JIT或GPU加速

### 局限性

- **假设圆形鱼眼**：不适合椭圆或矩形传感器（需修改）
- **单一色温**：标定针对拍摄照明（通常为D65）
- **无坏点校正**：生产流水线中应单独实现DPC
- **无噪声模型**：LSC放大暗区噪声；建议与降噪算法配合

### 贡献

欢迎贡献！请随时提交问题或拉取请求。

### 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

---

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

**Developed with ❤️ for the ISP and Computational Photography community**
