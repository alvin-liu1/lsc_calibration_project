# MIPI RAW格式支持文档

## 📅 更新日期
2026-01-27

---

## 🎯 功能概述

本项目现已支持**MIPI RAW格式**，可以自动识别并处理以下三种RAW格式：

1. **Plain RAW** - 16-bit容器存储10/12-bit数据（原有格式）
2. **MIPI RAW10** - 4个10-bit像素打包成5字节（节省20%空间）
3. **MIPI RAW12** - 2个12-bit像素打包成3字节（节省25%空间）

---

## 📊 格式对比

### Plain RAW vs MIPI RAW

| 格式 | 存储方式 | 文件大小 | 优势 | 劣势 |
|------|---------|---------|------|------|
| **Plain RAW** | 每像素2字节 | 100% | 简单直接，易处理 | 浪费空间 |
| **MIPI RAW10** | 4像素5字节 | 80% | 节省20%空间 | 需要解包 |
| **MIPI RAW12** | 2像素3字节 | 75% | 节省25%空间 | 需要解包 |

### 文件大小示例

以2904x2900图像为例：

```
Plain RAW:     2904 × 2900 × 2 = 16,843,200 字节 (16.06 MB)
MIPI RAW10:    2904 × 2900 × 1.25 = 10,527,000 字节 (10.04 MB)
MIPI RAW12:    2904 × 2900 × 1.5 = 12,632,400 字节 (12.05 MB)
```

---

## 🔧 使用方法

### 1. 自动检测模式（推荐）

在 [config.py](config.py) 中设置：

```python
RAW_FORMAT = 'auto'  # 自动检测格式
```

程序会根据文件大小自动识别格式：
- 如果文件大小 ≈ width × height × 2 → Plain RAW
- 如果文件大小 ≈ width × height × 1.25 → MIPI RAW10
- 如果文件大小 ≈ width × height × 1.5 → MIPI RAW12

### 2. 手动指定格式

如果自动检测失败，可以手动指定：

```python
# Plain RAW格式
RAW_FORMAT = 'plain'

# MIPI RAW10格式
RAW_FORMAT = 'mipi_raw10'

# MIPI RAW12格式
RAW_FORMAT = 'mipi_raw12'
```

---

## 📐 MIPI RAW格式详解

### MIPI RAW10 打包格式

**原理**: 4个10-bit像素打包成5字节

```
像素数据:
P0 = 10-bit (b9 b8 b7 b6 b5 b4 b3 b2 b1 b0)
P1 = 10-bit (b9 b8 b7 b6 b5 b4 b3 b2 b1 b0)
P2 = 10-bit (b9 b8 b7 b6 b5 b4 b3 b2 b1 b0)
P3 = 10-bit (b9 b8 b7 b6 b5 b4 b3 b2 b1 b0)

打包后5字节:
Byte0: P0[9:2]
Byte1: P1[9:2]
Byte2: P2[9:2]
Byte3: P3[9:2]
Byte4: P3[1:0] P2[1:0] P1[1:0] P0[1:0]
```

**解包算法**:
```python
p0 = ((b0 << 2) | ((b4 >> 0) & 0x03)) & 0x3FF
p1 = ((b1 << 2) | ((b4 >> 2) & 0x03)) & 0x3FF
p2 = ((b2 << 2) | ((b4 >> 4) & 0x03)) & 0x3FF
p3 = ((b3 << 2) | ((b4 >> 6) & 0x03)) & 0x3FF
```

---

### MIPI RAW12 打包格式

**原理**: 2个12-bit像素打包成3字节

```
像素数据:
P0 = 12-bit (b11 b10 b9 b8 b7 b6 b5 b4 b3 b2 b1 b0)
P1 = 12-bit (b11 b10 b9 b8 b7 b6 b5 b4 b3 b2 b1 b0)

打包后3字节:
Byte0: P0[11:4]
Byte1: P1[11:4]
Byte2: P1[3:0] P0[3:0]
```

**解包算法**:
```python
p0 = ((b0 << 4) | ((b2 >> 0) & 0x0F)) & 0xFFF
p1 = ((b1 << 4) | ((b2 >> 4) & 0x0F)) & 0xFFF
```

---

## ✅ 验证方法

### 检查文件格式

运行程序后，查看日志输出：

```
自动检测RAW格式: mipi_raw10
成功读取MIPI RAW10文件: input/2904x2900.raw, 尺寸: 2904x2900
```

### 验证解包正确性

1. **检查像素值范围**:
   - 10-bit: 0-1023
   - 12-bit: 0-4095

2. **检查图像质量**:
   - 查看输出图像是否正常
   - 检查是否有明显的条纹或噪点

3. **对比Plain RAW**:
   - 如果有Plain RAW版本，对比两者的校准结果

---

## 🔧 修改的文件

### 1. lsc/bayer_utils.py
- **新增函数**: `unpack_mipi_raw10()` - MIPI RAW10解包
- **新增函数**: `unpack_mipi_raw12()` - MIPI RAW12解包
- **新增函数**: `detect_raw_format()` - 自动格式检测
- **修改函数**: `read_raw_bayer_image()` - 支持多种RAW格式

### 2. config.py
- **新增配置**: `RAW_FORMAT` - RAW格式选择（'auto', 'plain', 'mipi_raw10', 'mipi_raw12'）

### 3. main.py
- **修改**: 传递 `raw_format` 参数到 `read_raw_bayer_image()`

---

## 📚 技术参考

### MIPI CSI-2规范
- MIPI RAW10/RAW12是MIPI CSI-2接口的标准数据格式
- 广泛应用于移动设备摄像头
- 高通、联发科等芯片厂商均支持

### 优势
1. **节省存储空间**: 比Plain RAW节省20-25%
2. **节省传输带宽**: 降低MIPI接口带宽需求
3. **硬件支持**: ISP硬件直接支持解包

---

## ⚠️ 注意事项

1. **像素对齐**:
   - MIPI RAW10要求宽度是4的倍数
   - MIPI RAW12要求宽度是2的倍数

2. **字节序**:
   - 本实现假设小端序（Little Endian）
   - 如果数据是大端序，需要调整解包算法

3. **性能**:
   - 解包过程使用Python循环，速度较慢
   - 对于大图像，建议使用NumPy向量化优化

---

## 🚀 未来优化

1. **性能优化**: 使用NumPy向量化替代Python循环
2. **更多格式**: 支持MIPI RAW8、RAW14等格式
3. **硬件加速**: 考虑使用Cython或C扩展加速解包

---

**版本**: V1.0
**作者**: LSC Calibration Team
**日期**: 2026-01-27