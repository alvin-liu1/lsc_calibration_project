# lsc_calibration_project/lsc/calibration.py

import numpy as np
import cv2
import logging
from . import gain_utils

def calculate_lsc_gains(bayer_channels_float, grid_rows, grid_cols, circle_info,
                        min_pixels_per_grid, valid_grid_threshold_ratio, falloff_factor, max_gain):
    """
    LSC核心算法：计算各通道的增益矩阵。
    【已优化】优化了增益计算逻辑，避免“除以零”的警告。
    """
    h, w = bayer_channels_float['Gr'].shape
    H_grid_cell = h // grid_rows
    W_grid_cell = w // grid_cols
    
    logging.info("步骤2: 计算每个网格的平均亮度...")
    grid_brightness = {ch: np.zeros((grid_rows, grid_cols), dtype=np.float32) for ch in ['R', 'Gr', 'Gb', 'B']}
    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hard_mask, (circle_info[0], circle_info[1]), circle_info[2], 1, -1)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y_start, y_end = r * H_grid_cell, (r + 1) * H_grid_cell
            x_start, x_end = c * W_grid_cell, (c + 1) * W_grid_cell
            mask_patch = hard_mask[y_start:y_end, x_start:x_end]
            gr_patch = bayer_channels_float['Gr'][y_start:y_end, x_start:x_end]
            num_valid_pixels = np.count_nonzero((gr_patch > 0) & (mask_patch == 1))
            if num_valid_pixels > min_pixels_per_grid:
                for ch_name in ['R', 'Gr', 'Gb', 'B']:
                    patch = bayer_channels_float[ch_name][y_start:y_end, x_start:x_end]
                    valid_pixel_mask = (patch > 0) & (mask_patch == 1)
                    if np.any(valid_pixel_mask):
                        grid_brightness[ch_name][r, c] = np.mean(patch[valid_pixel_mask])
    
    logging.info("步骤3: 计算LSC增益...")
    epsilon = 1e-6
    G_avg_map = (grid_brightness['Gr'] + grid_brightness['Gb']) / 2.0
    center_r, center_c = grid_rows // 2, grid_cols // 2
    center_G_avg = G_avg_map[center_r, center_c]
    center_R = grid_brightness['R'][center_r, center_c]
    center_B = grid_brightness['B'][center_r, center_c]
    logging.info(f"  - 中心网格参考亮度: R={center_R:.4f}, G_avg={center_G_avg:.4f}, B={center_B:.4f}")
    if center_G_avg < epsilon or center_R < epsilon or center_B < epsilon:
        logging.error("中心网格亮度过低或无效，无法继续计算。")
        raise ValueError("中心网格亮度过低。")

    master_valid_mask = G_avg_map > (center_G_avg * valid_grid_threshold_ratio)

    # --- Bug修复：优化计算方式以避免RuntimeWarning ---
    def safe_calculate_gain(center_val, grid_vals):
        gain_raw = np.ones_like(grid_vals, dtype=np.float32)
        valid_mask = grid_vals > epsilon
        gain_raw[valid_mask] = center_val / grid_vals[valid_mask]
        return gain_raw

    gain_R_raw = safe_calculate_gain(center_R, grid_brightness['R'])
    gain_G_raw = safe_calculate_gain(center_G_avg, G_avg_map)
    gain_B_raw = safe_calculate_gain(center_B, grid_brightness['B'])
    # ----------------------------------------------------

    luma_falloff_map = gain_utils.create_falloff_map(grid_rows, grid_cols, falloff_factor)
    final_gain_R = np.power(gain_R_raw, luma_falloff_map)
    final_gain_G = np.power(gain_G_raw, luma_falloff_map)
    final_gain_B = np.power(gain_B_raw, luma_falloff_map)

    final_gain_R[~master_valid_mask] = 1.0
    final_gain_G[~master_valid_mask] = 1.0
    final_gain_B[~master_valid_mask] = 1.0
    
    raw_gains = {
        'R': np.clip(final_gain_R, 1.0, max_gain),
        'Gr': np.clip(final_gain_G, 1.0, max_gain),
        'Gb': np.clip(final_gain_G, 1.0, max_gain),
        'B': np.clip(final_gain_B, 1.0, max_gain)
    }
    
    logging.info("原始增益矩阵计算完成。")
    return raw_gains