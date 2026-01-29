#!/usr/bin/env python3
"""测试MIPI RAW10解包是否正确"""

import numpy as np
import sys
sys.path.insert(0, '.')

from lsc import bayer_utils

# 测试参数
width = 2900
height = 2900
raw_path = "input/2900mipi.raw"

print(f"测试MIPI RAW10解包...")
print(f"文件: {raw_path}")
print(f"尺寸: {width}x{height}")

# 读取文件
try:
    bayer_image = bayer_utils.read_raw_bayer_image(
        raw_path, width, height, bit_depth=10, raw_format='mipi_raw10'
    )

    print(f"\n[SUCCESS] Unpacking successful!")
    print(f"Output shape: {bayer_image.shape}")
    print(f"Data type: {bayer_image.dtype}")
    print(f"Pixel value range: [{bayer_image.min()}, {bayer_image.max()}]")
    print(f"Mean: {bayer_image.mean():.2f}")
    print(f"Std: {bayer_image.std():.2f}")

    # Check for abnormal values
    if bayer_image.max() > 1023:
        print(f"\n[WARNING] Detected pixel values exceeding 10-bit range!")

    # Pixel value distribution
    print(f"\nPixel value distribution:")
    print(f"  0-255: {np.sum((bayer_image >= 0) & (bayer_image < 256))}")
    print(f"  256-511: {np.sum((bayer_image >= 256) & (bayer_image < 512))}")
    print(f"  512-767: {np.sum((bayer_image >= 512) & (bayer_image < 768))}")
    print(f"  768-1023: {np.sum((bayer_image >= 768) & (bayer_image <= 1023))}")

except Exception as e:
    print(f"\n[ERROR] Unpacking failed: {e}")
    import traceback
    traceback.print_exc()
