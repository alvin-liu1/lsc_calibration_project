#!/usr/bin/env python3
"""诊断MIPI RAW处理流程"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from lsc import bayer_utils

# 测试参数
width = 2900
height = 2900
raw_path = "input/2900mipi.raw"

print("=" * 60)
print("MIPI RAW10 Processing Diagnosis")
print("=" * 60)

# Step 1: 读取并解包MIPI RAW10
print("\n[Step 1] Reading and unpacking MIPI RAW10...")
bayer_16bit = bayer_utils.read_raw_bayer_image(
    raw_path, width, height, bit_depth=10, raw_format='mipi_raw10'
)

print(f"Bayer 16-bit shape: {bayer_16bit.shape}")
print(f"Bayer 16-bit dtype: {bayer_16bit.dtype}")
print(f"Bayer 16-bit range: [{bayer_16bit.min()}, {bayer_16bit.max()}]")
print(f"Bayer 16-bit mean: {bayer_16bit.mean():.2f}")

# Step 2: 提取Bayer通道
print("\n[Step 2] Extracting Bayer channels...")
h, w = bayer_16bit.shape

# RGGB Bayer pattern
R = bayer_16bit[0::2, 0::2]   # Red
Gr = bayer_16bit[0::2, 1::2]  # Green-Red
Gb = bayer_16bit[1::2, 0::2]  # Green-Blue
B = bayer_16bit[1::2, 1::2]   # Blue

print(f"R  channel: mean={R.mean():.2f}, range=[{R.min()}, {R.max()}]")
print(f"Gr channel: mean={Gr.mean():.2f}, range=[{Gr.min()}, {Gr.max()}]")
print(f"Gb channel: mean={Gb.mean():.2f}, range=[{Gb.min()}, {Gb.max()}]")
print(f"B  channel: mean={B.mean():.2f}, range=[{B.min()}, {B.max()}]")

# Step 3: 简单的Demosaic（最近邻插值）
print("\n[Step 3] Simple demosaic (nearest neighbor)...")

# 创建RGB图像（使用最近邻插值）
rgb_16bit = np.zeros((h, w, 3), dtype=np.uint16)
rgb_16bit[0::2, 0::2, 2] = R   # Red channel
rgb_16bit[0::2, 1::2, 1] = Gr  # Green-Red
rgb_16bit[1::2, 0::2, 1] = Gb  # Green-Blue
rgb_16bit[1::2, 1::2, 0] = B   # Blue channel

# 填充缺失像素（简单复制）
rgb_16bit[0::2, 0::2, 0] = B   # Blue at Red position
rgb_16bit[0::2, 0::2, 1] = (Gr + Gb) // 2  # Green at Red position
rgb_16bit[1::2, 1::2, 2] = R   # Red at Blue position
rgb_16bit[1::2, 1::2, 1] = (Gr + Gb) // 2  # Green at Blue position

print(f"RGB 16-bit shape: {rgb_16bit.shape}")
print(f"RGB 16-bit range: [{rgb_16bit.min()}, {rgb_16bit.max()}]")
print(f"RGB 16-bit mean: {rgb_16bit.mean():.2f}")

# Step 4: 转换为8-bit显示
print("\n[Step 4] Converting to 8-bit for display...")

# 方法1: 直接缩放（10-bit -> 8-bit）
rgb_8bit_direct = (rgb_16bit >> 2).astype(np.uint8)
print(f"Direct scaling: range=[{rgb_8bit_direct.min()}, {rgb_8bit_direct.max()}], mean={rgb_8bit_direct.mean():.2f}")

# 方法2: 归一化后缩放
rgb_normalized = (rgb_16bit.astype(np.float32) / 1023.0 * 255.0).astype(np.uint8)
print(f"Normalized scaling: range=[{rgb_normalized.min()}, {rgb_normalized.max()}], mean={rgb_normalized.mean():.2f}")

# 保存测试图像
cv2.imwrite('output/test_mipi_direct.png', cv2.cvtColor(rgb_8bit_direct, cv2.COLOR_RGB2BGR))
cv2.imwrite('output/test_mipi_normalized.png', cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2BGR))

print("\n[SUCCESS] Test images saved:")
print("  - output/test_mipi_direct.png")
print("  - output/test_mipi_normalized.png")
print("\nPlease check if these images look correct.")
