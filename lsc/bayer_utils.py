# lsc_calibration_project/lsc/bayer_utils.py

import numpy as np
import cv2
import logging
import os

def unpack_mipi_raw10(packed_data, width, height):
    """
    解包MIPI RAW10格式数据。

    MIPI RAW10: 4个10-bit像素打包成5字节
    [P0: 9-2] [P1: 9-2] [P2: 9-2] [P3: 9-2] [P3:1-0 P2:1-0 P1:1-0 P0:1-0]

    参数:
        packed_data (np.array): 打包的MIPI RAW10数据
        width (int): 图像宽度
        height (int): 图像高度

    返回:
        np.array: 解包后的16-bit Bayer图像
    """
    # 计算每行的字节数（无padding）
    bytes_per_row_no_pad = (width * 10) // 8
    total_pixels = width * height
    expected_bytes_no_pad = (total_pixels * 10) // 8

    # 检测行对齐padding
    actual_bytes = len(packed_data)
    bytes_per_row_actual = actual_bytes // height

    # 计算padding大小
    padding_per_row = bytes_per_row_actual - bytes_per_row_no_pad

    if padding_per_row < 0:
        raise ValueError(f"MIPI RAW10数据不足: 期望至少{expected_bytes_no_pad}字节, 实际{actual_bytes}字节")

    if padding_per_row > 0:
        logging.info(f"检测到行对齐: 每行{bytes_per_row_actual}字节 (数据{bytes_per_row_no_pad} + padding{padding_per_row})")

    # 逐行解包，跳过每行的padding
    unpacked = np.zeros(total_pixels, dtype=np.uint16)

    for row in range(height):
        # 计算当前行在packed_data中的起始位置
        row_start = row * bytes_per_row_actual
        row_data = packed_data[row_start:row_start + bytes_per_row_no_pad]

        # 解包当前行的所有像素
        num_groups_per_row = width // 4
        row_offset = row * width

        for i in range(num_groups_per_row):
            base = i * 5
            # 读取5字节并转换为int
            b0, b1, b2, b3, b4 = [int(x) for x in row_data[base:base+5]]

            # 解包4个10-bit像素
            p0 = ((b0 << 2) | ((b4 >> 0) & 0x03)) & 0x3FF
            p1 = ((b1 << 2) | ((b4 >> 2) & 0x03)) & 0x3FF
            p2 = ((b2 << 2) | ((b4 >> 4) & 0x03)) & 0x3FF
            p3 = ((b3 << 2) | ((b4 >> 6) & 0x03)) & 0x3FF

            unpacked[row_offset + i*4:row_offset + (i+1)*4] = [p0, p1, p2, p3]

    return unpacked.reshape((height, width))


def unpack_mipi_raw12(packed_data, width, height):
    """
    解包MIPI RAW12格式数据。

    MIPI RAW12: 2个12-bit像素打包成3字节
    [P0: 11-4] [P1: 11-4] [P1:3-0 P0:3-0]

    参数:
        packed_data (np.array): 打包的MIPI RAW12数据
        width (int): 图像宽度
        height (int): 图像高度

    返回:
        np.array: 解包后的16-bit Bayer图像
    """
    # 计算每行的字节数（无padding）
    bytes_per_row_no_pad = (width * 12) // 8
    total_pixels = width * height
    expected_bytes_no_pad = (total_pixels * 12) // 8

    # 检测行对齐padding
    actual_bytes = len(packed_data)
    bytes_per_row_actual = actual_bytes // height

    # 计算padding大小
    padding_per_row = bytes_per_row_actual - bytes_per_row_no_pad

    if padding_per_row < 0:
        raise ValueError(f"MIPI RAW12数据不足: 期望至少{expected_bytes_no_pad}字节, 实际{actual_bytes}字节")

    if padding_per_row > 0:
        logging.info(f"检测到行对齐: 每行{bytes_per_row_actual}字节 (数据{bytes_per_row_no_pad} + padding{padding_per_row})")

    # 逐行解包，跳过每行的padding
    unpacked = np.zeros(total_pixels, dtype=np.uint16)

    for row in range(height):
        # 计算当前行在packed_data中的起始位置
        row_start = row * bytes_per_row_actual
        row_data = packed_data[row_start:row_start + bytes_per_row_no_pad]

        # 解包当前行的所有像素
        num_groups_per_row = width // 2
        row_offset = row * width

        for i in range(num_groups_per_row):
            base = i * 3
            # 读取3字节并转换为int
            b0, b1, b2 = [int(x) for x in row_data[base:base+3]]

            # 解包2个12-bit像素
            p0 = ((b0 << 4) | ((b2 >> 0) & 0x0F)) & 0xFFF
            p1 = ((b1 << 4) | ((b2 >> 4) & 0x0F)) & 0xFFF

            unpacked[row_offset + i*2:row_offset + (i+1)*2] = [p0, p1]

    return unpacked.reshape((height, width))


def detect_raw_format(raw_path, width, height, bit_depth):
    """
    自动检测RAW文件格式（Plain RAW vs MIPI RAW）。

    参数:
        raw_path (str): RAW文件路径
        width (int): 图像宽度
        height (int): 图像高度
        bit_depth (int): 位深（10或12）

    返回:
        str: 'plain', 'mipi_raw10', 或 'mipi_raw12'
    """
    file_size = os.path.getsize(raw_path)
    total_pixels = width * height

    # Plain RAW: 每像素2字节
    plain_size = total_pixels * 2

    # MIPI RAW10: 每4像素5字节
    mipi10_size = (total_pixels * 10) // 8

    # MIPI RAW12: 每2像素3字节
    mipi12_size = (total_pixels * 12) // 8

    # 允许±1%误差
    tolerance = 0.01

    if abs(file_size - plain_size) / plain_size < tolerance:
        return 'plain'
    elif bit_depth == 10 and abs(file_size - mipi10_size) / mipi10_size < tolerance:
        return 'mipi_raw10'
    elif bit_depth == 12 and abs(file_size - mipi12_size) / mipi12_size < tolerance:
        return 'mipi_raw12'
    else:
        logging.warning(f"无法识别RAW格式: 文件大小={file_size}, Plain={plain_size}, MIPI10={mipi10_size}, MIPI12={mipi12_size}")
        return 'plain'  # 默认使用Plain格式


def read_raw_bayer_image(raw_path, width, height, bit_depth=10, raw_format='auto'):
    """
    从文件读取RAW Bayer数据，支持Plain RAW和MIPI RAW格式。

    支持的格式:
    - Plain RAW: 16-bit容器存储10/12-bit数据
    - MIPI RAW10: 4个10-bit像素打包成5字节
    - MIPI RAW12: 2个12-bit像素打包成3字节

    参数:
        raw_path (str): .raw文件的路径
        width (int): 图像宽度
        height (int): 图像高度
        bit_depth (int): 数据的位深, 如10或12
        raw_format (str): RAW格式 ('auto', 'plain', 'mipi_raw10', 'mipi_raw12')

    返回:
        np.array: 包含16位Bayer数据的二维Numpy数组，如果失败则返回None
    """
    try:
        # 自动检测格式
        if raw_format == 'auto':
            raw_format = detect_raw_format(raw_path, width, height, bit_depth)
            logging.info(f"自动检测RAW格式: {raw_format}")

        # 根据格式读取数据
        if raw_format == 'mipi_raw10':
            # MIPI RAW10格式 - 读取整个文件（包含可能的行对齐padding）
            packed_data = np.fromfile(raw_path, dtype=np.uint8)
            bayer_image_16bit = unpack_mipi_raw10(packed_data, width, height)
            logging.info(f"成功读取MIPI RAW10文件: {raw_path}, 尺寸: {width}x{height}")

        elif raw_format == 'mipi_raw12':
            # MIPI RAW12格式 - 读取整个文件（包含可能的行对齐padding）
            packed_data = np.fromfile(raw_path, dtype=np.uint8)
            bayer_image_16bit = unpack_mipi_raw12(packed_data, width, height)
            logging.info(f"成功读取MIPI RAW12文件: {raw_path}, 尺寸: {width}x{height}")

        else:
            # Plain RAW格式（默认）
            expected_pixels = width * height
            bayer_data_16bit_container = np.fromfile(raw_path, dtype='<u2', count=expected_pixels)

            if bayer_data_16bit_container.size != expected_pixels:
                logging.error(f"RAW文件尺寸不匹配。期望 {expected_pixels} 像素, 实际读取到 {bayer_data_16bit_container.size}。")
                raise ValueError("Raw file size mismatch.")

            # 根据位深，使用位掩码提取有效数据
            mask = (1 << bit_depth) - 1
            bayer_image_16bit = (bayer_data_16bit_container.astype(np.uint16) & mask)
            bayer_image_16bit = bayer_image_16bit.reshape((height, width))
            logging.info(f"成功读取Plain RAW文件: {raw_path}, 尺寸: {width}x{height}, 位深: {bit_depth}-bit")

        return bayer_image_16bit

    except Exception as e:
        logging.error(f"读取RAW文件时发生错误: {e}")
        return None

def extract_bayer_channels(bayer_image_16bit, bayer_pattern_code, black_levels, sensor_max_val):
    """
    从Bayer图中分离R, Gr, Gb, B四个通道，并对每个通道独立进行黑电平校正和归一化。
    这样做可以从根本上保证颜色比例的正确性。

    参数:
        bayer_image_16bit (np.array): 原始Bayer图像数据。
        bayer_pattern_code (int): OpenCV的Bayer pattern转换码。
        black_levels (dict): 包含{'R', 'Gr', 'Gb', 'B'}的黑电平字典。
        sensor_max_val (float): 传感器的最大值 (例如10-bit为1023.0)。

    返回:
        dict: 一个包含四个归一化浮点数通道图的字典 {'R', 'Gr', 'Gb', 'B'}。
    """
    h, w = bayer_image_16bit.shape
    # 创建与bayer图同样大小的浮点数0矩阵
    channels = {ch: np.zeros((h, w), dtype=np.float32) for ch in ['R', 'Gr', 'Gb', 'B']}

    # 临时字典，用于根据pattern将数据填充到正确位置
    temp_channels_bayer = {'R': np.zeros_like(bayer_image_16bit, dtype=np.float32),
                           'Gr': np.zeros_like(bayer_image_16bit, dtype=np.float32),
                           'Gb': np.zeros_like(bayer_image_16bit, dtype=np.float32),
                           'B': np.zeros_like(bayer_image_16bit, dtype=np.float32)}

    # 根据Bayer模式分离通道
    if bayer_pattern_code in [cv2.COLOR_BayerRG2BGR_VNG, cv2.COLOR_BayerRG2RGB]:  # RGGB
        temp_channels_bayer['R'][0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        temp_channels_bayer['Gr'][0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        temp_channels_bayer['Gb'][1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        temp_channels_bayer['B'][1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    elif bayer_pattern_code in [cv2.COLOR_BayerGR2BGR_VNG, cv2.COLOR_BayerGR2RGB]:  # GRBG
        temp_channels_bayer['Gr'][0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        temp_channels_bayer['R'][0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        temp_channels_bayer['B'][1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        temp_channels_bayer['Gb'][1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    elif bayer_pattern_code in [cv2.COLOR_BayerBG2BGR_VNG, cv2.COLOR_BayerBG2RGB]:  # BGGR
        temp_channels_bayer['B'][0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        temp_channels_bayer['Gb'][0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        temp_channels_bayer['Gr'][1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        temp_channels_bayer['R'][1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    elif bayer_pattern_code in [cv2.COLOR_BayerGB2BGR_VNG, cv2.COLOR_BayerGB2RGB]:  # GBRG
        temp_channels_bayer['Gb'][0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        temp_channels_bayer['B'][0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        temp_channels_bayer['R'][1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        temp_channels_bayer['Gr'][1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    else:
        raise ValueError(f"不支持的Bayer Pattern代码: {bayer_pattern_code}")

    # 对每个通道独立进行黑电平校正和归一化
    for ch_name in ['R', 'Gr', 'Gb', 'B']:
        bl = black_levels[ch_name]
        # 减去黑电平，并确保值不小于0
        blc_ch = np.maximum(0, temp_channels_bayer[ch_name] - bl)

        # 计算归一化因子 (白电平)
        white_level = sensor_max_val - bl
        if white_level <= 0:
            logging.warning(f"通道 {ch_name} 的白电平 <= 0，归一化可能出错。")
            white_level = sensor_max_val

        # 归一化到 [0, 1] 范围
        # 只对非零像素进行除法，避免除以0的警告
        normalized_ch = channels[ch_name] # 指向最终要填充的矩阵
        mask = blc_ch > 0
        normalized_ch[mask] = blc_ch[mask] / white_level
        channels[ch_name] = normalized_ch

    logging.info("Bayer通道提取、黑电平校正和归一化完成。")
    return channels


def apply_gains_to_bayer(bayer_blc_float, gain_maps, bayer_pattern_code, hard_mask=None): # [修改] hard_mask改为可选
    """
    将通过插值后得到的全尺寸增益图应用到已减去黑电平的浮点Bayer数据上。

    【V3.0 - 全景拼接优化版】
    重要变更：不再使用 hard_mask 清零！

    原因：
    1. 全景拼接需要保留圆外的原始数据（虽然很暗）
    2. LSC增益表已通过 dampen_gains_by_geometry 将圆外增益压回1.0
    3. 增益1.0 × 原始暗值 = 保持原样（不放大噪声，也不删除数据）

    对比旧版：
    - V2.2: compensated_bayer[mask==0] = 0.0  ❌ 破坏拼接
    - V3.0: 圆外增益=1.0，保持原始值         ✅ 拼接友好

    参数:
        bayer_blc_float (np.array): 减去黑电平后的浮点Bayer图。
        gain_maps (dict): 包含{'R', 'Gr', 'Gb', 'B'}四个通道全尺寸增益图的字典。
        bayer_pattern_code (int): OpenCV的Bayer pattern转换码。
        hard_mask (np.array, optional): 保留用于兼容性，但不再使用。
    """
    compensated_bayer = bayer_blc_float.copy()

    # 根据Bayer Pattern，将对应通道的增益应用到对应的像素位置
    # (这部分逻辑不变，增益会应用到所有地方，包括无效区)
    if bayer_pattern_code in [cv2.COLOR_BayerRG2BGR_VNG, cv2.COLOR_BayerRG2RGB]:  # RGGB
        compensated_bayer[0::2, 0::2] *= gain_maps['R'][0::2, 0::2]
        compensated_bayer[0::2, 1::2] *= gain_maps['Gr'][0::2, 1::2]
        compensated_bayer[1::2, 0::2] *= gain_maps['Gb'][1::2, 0::2]
        compensated_bayer[1::2, 1::2] *= gain_maps['B'][1::2, 1::2]
    elif bayer_pattern_code in [cv2.COLOR_BayerGR2BGR_VNG, cv2.COLOR_BayerGR2RGB]:  # GRBG
        compensated_bayer[0::2, 0::2] *= gain_maps['Gr'][0::2, 0::2]
        compensated_bayer[0::2, 1::2] *= gain_maps['R'][0::2, 1::2]
        compensated_bayer[1::2, 0::2] *= gain_maps['B'][1::2, 0::2]
        compensated_bayer[1::2, 1::2] *= gain_maps['Gb'][1::2, 1::2]
    elif bayer_pattern_code in [cv2.COLOR_BayerBG2BGR_VNG, cv2.COLOR_BayerBG2RGB]:  # BGGR
        compensated_bayer[0::2, 0::2] *= gain_maps['B'][0::2, 0::2]
        compensated_bayer[0::2, 1::2] *= gain_maps['Gb'][0::2, 1::2]
        compensated_bayer[1::2, 0::2] *= gain_maps['Gr'][1::2, 0::2]
        compensated_bayer[1::2, 1::2] *= gain_maps['R'][1::2, 1::2]
    elif bayer_pattern_code in [cv2.COLOR_BayerGB2BGR_VNG, cv2.COLOR_BayerGB2RGB]:  # GBRG
        compensated_bayer[0::2, 0::2] *= gain_maps['Gb'][0::2, 0::2]
        compensated_bayer[0::2, 1::2] *= gain_maps['B'][0::2, 1::2]
        compensated_bayer[1::2, 0::2] *= gain_maps['R'][1::2, 0::2]
        compensated_bayer[1::2, 1::2] *= gain_maps['Gr'][1::2, 1::2]

    # --- [V3.0 重要变更] ---
    # 不再清零圆外区域！原因：
    # 1. 增益表已通过 dampen_gains_by_geometry 将圆外增益设为1.0
    # 2. 增益1.0 × 暗像素 = 保持原样（不放大，不删除）
    # 3. 全景拼接需要这些原始数据进行图像融合
    #
    # 旧代码（已移除）: compensated_bayer[hard_mask == 0] = 0.0
    # -------------------

    return compensated_bayer