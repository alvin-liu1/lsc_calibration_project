# lsc_calibration_project/lsc/image_utils.py

import numpy as np
import logging

def simple_white_balance(image_rgb_float, mask_2d=None):
    """
    对RGB浮点图像进行一个简单的自动白平衡。
    通过统计图像中心区域的R,G,B平均值，并以G通道为基准来调整R和B通道。

    参数:
        image_rgb_float (np.array): 范围在 [0, 1] 的三通道RGB图像。
        mask_2d (np.array, optional): 一个二维掩码，只有掩码内的像素会用于计算。

    返回:
        np.array: 白平衡后的RGB图像。
    """
    if image_rgb_float.shape[2] != 3:
        raise ValueError("输入图像必须是3通道RGB图像。")

    balanced_image = image_rgb_float.copy()
    h, w, _ = image_rgb_float.shape

    # 定义中心区域 (30% - 70% 区域)
    y_start, y_end = int(h * 0.3), int(h * 0.7)
    x_start, x_end = int(w * 0.3), int(w * 0.7)

    # 提取中心区域
    central_patch_R = balanced_image[y_start:y_end, x_start:x_end, 0]
    central_patch_G = balanced_image[y_start:y_end, x_start:x_end, 1]
    central_patch_B = balanced_image[y_start:y_end, x_start:x_end, 2]

    # 如果提供了掩码，则只在掩码内的有效区域进行计算
    if mask_2d is not None:
        central_mask_patch = mask_2d[y_start:y_end, x_start:x_end]
        valid_pixels_mask = central_mask_patch > 0.1 # 使用一个小的阈值
    else:
        valid_pixels_mask = np.ones_like(central_patch_G, dtype=bool)

    # 检查G通道是否有效，防止除以0
    g_channel_valid_pixels = central_patch_G[valid_pixels_mask]
    if g_channel_valid_pixels.size == 0 or np.mean(g_channel_valid_pixels) < 1e-6:
        logging.warning("白平衡计算区域的G通道均值过低或无有效像素，跳过白平衡。")
        return image_rgb_float

    avg_G = np.mean(g_channel_valid_pixels)
    avg_R = np.mean(central_patch_R[valid_pixels_mask])
    avg_B = np.mean(central_patch_B[valid_pixels_mask])

    # 计算增益
    gain_R = avg_G / (avg_R + 1e-6) # 加一个极小值避免除以0
    gain_B = avg_G / (avg_B + 1e-6)
    
    logging.info(f"计算出的白平衡增益 (基于中心区域): R={gain_R:.3f}, G=1.00, B={gain_B:.3f}")

    # 应用增益并裁剪到[0, 1]范围
    balanced_image[:, :, 0] = np.clip(balanced_image[:, :, 0] * gain_R, 0, 1.0)
    balanced_image[:, :, 2] = np.clip(balanced_image[:, :, 2] * gain_B, 0, 1.0)

    return balanced_image