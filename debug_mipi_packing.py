#!/usr/bin/env python3
"""调试MIPI RAW10打包格式"""

import numpy as np
import os

# 测试参数
width = 2900
height = 2900
raw_path = "input/2900mipi.raw"

print("=" * 60)
print("MIPI RAW10 Packing Format Debug")
print("=" * 60)

# 检查文件大小
file_size = os.path.getsize(raw_path)
print(f"\n[File Info]")
print(f"File path: {raw_path}")
print(f"File size: {file_size} bytes")

# 计算预期大小
total_pixels = width * height
expected_no_padding = (total_pixels * 10) // 8
bytes_per_row_no_padding = (width * 10) // 8

print(f"\n[Expected Sizes]")
print(f"Total pixels: {total_pixels}")
print(f"Expected (no padding): {expected_no_padding} bytes")
print(f"Bytes per row (no padding): {bytes_per_row_no_padding} bytes")

# 检查可能的行对齐
possible_alignments = [1, 2, 4, 8, 16, 32, 64, 128, 256]
print(f"\n[Checking Row Alignment]")

for alignment in possible_alignments:
    bytes_per_row_aligned = ((bytes_per_row_no_padding + alignment - 1) // alignment) * alignment
    total_with_alignment = bytes_per_row_aligned * height

    if abs(file_size - total_with_alignment) < 10:
        print(f"  * MATCH: {alignment}-byte alignment")
        print(f"    Bytes per row: {bytes_per_row_aligned}")
        print(f"    Total: {total_with_alignment} bytes")
        print(f"    Padding per row: {bytes_per_row_aligned - bytes_per_row_no_padding} bytes")

# 读取第一行数据进行分析
print(f"\n[First Row Analysis]")
with open(raw_path, 'rb') as f:
    first_row_bytes = f.read(bytes_per_row_no_padding + 256)  # 多读一些

print(f"Read {len(first_row_bytes)} bytes from first row")

# 尝试不同的解包方式
print(f"\n[Testing Different Unpacking Methods]")

# 方法1: 当前实现
print("\n  Method 1: Current implementation (MSB first)")
data = np.frombuffer(first_row_bytes, dtype=np.uint8, count=20)
for i in range(4):
    base = i * 5
    b0, b1, b2, b3, b4 = [int(x) for x in data[base:base+5]]

    p0 = ((b0 << 2) | ((b4 >> 0) & 0x03)) & 0x3FF
    p1 = ((b1 << 2) | ((b4 >> 2) & 0x03)) & 0x3FF
    p2 = ((b2 << 2) | ((b4 >> 4) & 0x03)) & 0x3FF
    p3 = ((b3 << 2) | ((b4 >> 6) & 0x03)) & 0x3FF

    print(f"    Group {i}: [{p0:4d}, {p1:4d}, {p2:4d}, {p3:4d}]  bytes=[{b0:3d}, {b1:3d}, {b2:3d}, {b3:3d}, {b4:3d}]")

# 方法2: LSB在高位
print("\n  Method 2: LSB in high bits")
for i in range(4):
    base = i * 5
    b0, b1, b2, b3, b4 = [int(x) for x in data[base:base+5]]

    p0 = (b0 | ((b4 & 0x03) << 8)) & 0x3FF
    p1 = (b1 | (((b4 >> 2) & 0x03) << 8)) & 0x3FF
    p2 = (b2 | (((b4 >> 4) & 0x03) << 8)) & 0x3FF
    p3 = (b3 | (((b4 >> 6) & 0x03) << 8)) & 0x3FF

    print(f"    Group {i}: [{p0:4d}, {p1:4d}, {p2:4d}, {p3:4d}]")

# 方法3: 字节顺序反转
print("\n  Method 3: Different byte order")
for i in range(4):
    base = i * 5
    b4, b3, b2, b1, b0 = [int(x) for x in data[base:base+5]]

    p0 = ((b0 << 2) | ((b4 >> 0) & 0x03)) & 0x3FF
    p1 = ((b1 << 2) | ((b4 >> 2) & 0x03)) & 0x3FF
    p2 = ((b2 << 2) | ((b4 >> 4) & 0x03)) & 0x3FF
    p3 = ((b3 << 2) | ((b4 >> 6) & 0x03)) & 0x3FF

    print(f"    Group {i}: [{p0:4d}, {p1:4d}, {p2:4d}, {p3:4d}]")

print("\n" + "=" * 60)
print("Please check which method produces reasonable values (0-1023)")
print("and whether the values change smoothly without sudden jumps.")
print("=" * 60)
