# lsc_calibration_project/lsc/bayer_utils.py

import numpy as np
import cv2
import logging

def read_raw_bayer_image(raw_path, width, height, bit_depth=10):
    """
    从文件读取裸的RAW Bayer数据。
    此函数针对10-bit数据存储在16-bit容器中的情况作了优化。

    参数:
        raw_path (str): .raw文件的路径。
        width (int): 图像宽度。
        height (int): 图像高度。
        bit_depth (int): 数据的位深, 如10或12。

    返回:
        np.array: 包含16位Bayer数据的二维Numpy数组，如果失败则返回None。
    """
    try:
        expected_pixels = width * height
        # 以无符号16位小端格式读取文件
        bayer_data_16bit_container = np.fromfile(raw_path, dtype='<u2', count=expected_pixels)

        if bayer_data_16bit_container.size != expected_pixels:
            logging.error(f"RAW文件尺寸不匹配。期望 {expected_pixels} 像素, 实际读取到 {bayer_data_16bit_container.size}。")
            raise ValueError("Raw file size mismatch.")

        # 根据位深，使用位掩码提取有效数据
        # 例如，对于10-bit数据，掩码是 0x03FF (二进制的10个1)
        mask = (1 << bit_depth) - 1
        bayer_image_16bit = (bayer_data_16bit_container.astype(np.uint16) & mask)
        bayer_image_16bit = bayer_image_16bit.reshape((height, width))

        logging.info(f"成功读取RAW文件: {raw_path}, 尺寸: {width}x{height}, 位深: {bit_depth}-bit")
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


def apply_gains_to_bayer(bayer_blc_float, gain_maps, bayer_pattern_code):
    """
    将通过插值后得到的全尺寸增益图应用到已减去黑电平的浮点Bayer数据上。

    参数:
        bayer_blc_float (np.array): 减去黑电平后的浮点Bayer图。
        gain_maps (dict): 包含{'R', 'Gr', 'Gb', 'B'}四个通道全尺寸增益图的字典。
        bayer_pattern_code (int): OpenCV的Bayer pattern转换码。

    返回:
        np.array: 应用增益后的浮点Bayer数据。
    """
    compensated_bayer = bayer_blc_float.copy()

    # 根据Bayer Pattern，将对应通道的增益应用到对应的像素位置
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
        
    return compensated_bayer