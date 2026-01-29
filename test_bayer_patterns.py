#!/usr/bin/env python3
"""测试不同的Bayer pattern"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from lsc import bayer_utils

# 读取MIPI RAW10文件
width = 2900
height = 2900
raw_path = "input/2900mipi.raw"

print("Reading MIPI RAW10 file...")
bayer_16bit = bayer_utils.read_raw_bayer_image(
    raw_path, width, height, bit_depth=10, raw_format='mipi_raw10'
)

# 转换为8-bit用于显示
bayer_8bit = (bayer_16bit >> 2).astype(np.uint8)

print("\nTesting different Bayer patterns...")

# 测试所有4种Bayer pattern
patterns = {
    'RGGB': cv2.COLOR_BayerRG2BGR_VNG,
    'GRBG': cv2.COLOR_BayerGR2BGR_VNG,
    'GBRG': cv2.COLOR_BayerGB2BGR_VNG,
    'BGGR': cv2.COLOR_BayerBG2BGR_VNG
}

for name, pattern in patterns.items():
    # Demosaic
    rgb = cv2.cvtColor(bayer_8bit, pattern)

    # 计算中心区域的平均颜色
    h, w = rgb.shape[:2]
    center_y, center_x = h // 2, w // 2
    center_region = rgb[center_y-100:center_y+100, center_x-100:center_x+100]

    b_mean = center_region[:, :, 0].mean()
    g_mean = center_region[:, :, 1].mean()
    r_mean = center_region[:, :, 2].mean()

    # 判断主色调
    max_channel = max(b_mean, g_mean, r_mean)
    if max_channel == g_mean:
        color = "GREEN"
    elif max_channel == r_mean and b_mean > g_mean:
        color = "PURPLE/MAGENTA"
    elif max_channel == r_mean:
        color = "RED"
    else:
        color = "BLUE"

    print(f"{name}: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f} -> {color}")

    # 保存测试图像
    cv2.imwrite(f"output/test_pattern_{name}.png", rgb)

print("\nTest images saved to output/test_pattern_*.png")
print("Please check which one shows GREEN in the center (correct pattern)")
