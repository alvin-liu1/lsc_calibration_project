# lsc_calibration_project/lsc/calibration.py

import numpy as np
import cv2
import logging
from . import gain_utils

def dampen_gains_by_geometry(gain_matrix, rows, cols, circle_info, image_w, image_h, damping_ratio, damping_width_px):
    """
    [鱼眼专用 - 全景拼接优化版] 径向衰减保护函数。
    基于物理几何位置，强制将圆外无效区的增益平滑压回 1.0。

    **为什么增益要压回1.0而不是清零？**
    - 增益1.0 = 保持原始像素值不变
    - 无效区（很暗）应该保持暗，不要被LSC放大（避免噪声）
    - 保留原始数据供全景拼接算法使用（不要硬清零）

    参数:
        damping_ratio: 开始衰减的半径比例 (例如 1.0 表示从圆边缘开始)
        damping_width_px: 衰减过渡区的宽度 (像素)
    """
    cx, cy, r_pixel = circle_info

    # 1. 生成 Gain Table 对应的像素坐标网格
    step_h = image_h / (rows - 1)
    step_w = image_w / (cols - 1)

    y_indices, x_indices = np.indices((rows, cols))

    pixel_x = x_indices * step_w
    pixel_y = y_indices * step_h

    # 2. 计算每个顶点到圆心 (cx, cy) 的欧几里得距离
    dist_map = np.sqrt((pixel_x - cx)**2 + (pixel_y - cy)**2)

    # 3. 计算衰减权重 (Weight Mask)
    # 距离 < start_damp_r  -> 权重 1.0 (保留 LSC 增益)
    # 距离 > end_damp_r    -> 权重 0.0 (强制变为 1.0)

    start_damp_r = r_pixel * damping_ratio
    end_damp_r = start_damp_r + damping_width_px

    # weights 代表 "原始高增益的保留比例"
    weights = np.clip((end_damp_r - dist_map) / (end_damp_r - start_damp_r + 1e-6), 0.0, 1.0)

    # 4. 执行混合
    # Target Gain = 1.0 (Unity Gain，即不拉亮，保持死黑)
    target_gain = 1.0

    dampened_matrix = gain_matrix * weights + target_gain * (1.0 - weights)

    return dampened_matrix.astype(np.float32)


def fit_radial_gain_table(brightness_grid, rows, cols, circle_info, image_w, image_h, max_gain):
    """
    【核心算法 V5.1】径向多项式拟合
    将嘈杂的网格数据拟合成光滑的径向曲线，生成无条纹、无色偏的增益表。

    修复: 降低阶数至 4 并收缩拟合范围至 92%，解决边缘翘起问题。
    """
    cx, cy, radius = circle_info

    # 1. 建立坐标系
    step_h = image_h / (rows - 1)
    step_w = image_w / (cols - 1)
    y_idx, x_idx = np.indices((rows, cols))
    px_x = x_idx * step_w
    px_y = y_idx * step_h

    # 计算每个网格点到圆心的距离 r
    r_dist = np.sqrt((px_x - cx)**2 + (px_y - cy)**2)

    # 2. 收集有效数据点进行拟合
    # 归一化半径 (0.0 ~ 1.0)
    norm_r = r_dist / radius

    # 展平数组用于拟合
    flat_r = norm_r.flatten()
    flat_val = brightness_grid.flatten()

    # [关键优化] 筛选拟合数据
    # 1. brightness > 0.001: 排除完全死黑的点
    # 2. flat_r < 0.92: 仅使用圆心到 92% 处的数据。
    #    排除最边缘 8% 的低信噪比区域，防止噪点导致曲线末端上翘 (Overshoot)。
    valid_mask = (flat_val > 0.001) & (flat_r < 0.92)

    if np.sum(valid_mask) < 10:
        logging.warning("有效拟合点过少，拟合失败，返回全1矩阵")
        return np.ones_like(brightness_grid)

    train_r = flat_r[valid_mask]
    train_val = flat_val[valid_mask]

    # 3. 寻找中心最大亮度 (用于计算 Gain = Max / Fit_Val)
    max_brightness = np.max(train_val)

    # 4. 多项式拟合 (Polynomial Fit)
    try:
        # [关键优化] 阶数降为 4
        # 6阶容易过拟合导致边缘震荡，4阶足够描述光衰且更加刚性平滑
        coeffs = np.polyfit(train_r, train_val, 4)
        poly_func = np.poly1d(coeffs)

        # 5. 重建光滑的增益表
        # 计算所有网格点的拟合亮度 (包括被 mask 掉的边缘)
        fitted_brightness = poly_func(flat_r).reshape(rows, cols)

        # 防止分母为0或负数 (曲线拟合到远处可能小于0)
        fitted_brightness = np.maximum(fitted_brightness, 0.001)

        # 计算增益
        smooth_gain_grid = max_brightness / fitted_brightness

    except Exception as e:
        logging.error(f"径向拟合失败: {e}，回退到原始数据")
        smooth_gain_grid = max_brightness / np.maximum(brightness_grid, 0.001)

    return smooth_gain_grid


def calculate_lsc_gains(
    bayer_channels_float,
    grid_rows, grid_cols,
    hard_mask,
    min_pixels_per_grid,
    valid_grid_threshold_ratio,
    falloff_factor, # (在拟合模式下此参数被忽略)
    max_gain,
    circle_info,
    image_width,
    image_height,
    fisheye_config,
    smooth_kernel_size=(3, 3)
):
    """
    LSC 增益计算主入口。
    流程: 粗略统计 -> 径向多项式拟合 -> 径向死黑衰减 -> 硬件钳位
    """
    h, w = bayer_channels_float['Gr'].shape

    # 顶点数
    num_v_verts = grid_rows + 1
    num_h_verts = grid_cols + 1
    H_grid_cell = h / grid_rows
    W_grid_cell = w / grid_cols

    logging.info(f"步骤2: 统计单元格亮度 (Grid: {grid_rows}x{grid_cols})...")

    # 1. 原始数据统计 (Raw Statistics)
    # 先算出粗糙的网格亮度图，包含噪点和条纹，但这只是中间数据
    raw_brightness_map = {}

    for ch in ['R', 'Gr', 'Gb', 'B']:
        cell_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        for r in range(grid_rows):
            for c in range(grid_cols):
                y_start, y_end = int(r * H_grid_cell), int((r + 1) * H_grid_cell)
                x_start, x_end = int(c * W_grid_cell), int((c + 1) * W_grid_cell)

                mask_patch = hard_mask[y_start:y_end, x_start:x_end]
                patch = bayer_channels_float[ch][y_start:y_end, x_start:x_end]

                # 仅统计有效像素
                valid_vals = patch[(patch > 0) & (mask_patch > 0)]
                if valid_vals.size > min_pixels_per_grid:
                    cell_map[r, c] = np.mean(valid_vals)

        # 简单插值到顶点尺寸 (NxM -> N+1 x M+1)
        # 这里用最近邻或线性都行，因为后面会做强大的径向拟合重建整个表面
        vertex_map = cv2.resize(cell_map, (num_h_verts, num_v_verts), interpolation=cv2.INTER_LINEAR)
        raw_brightness_map[ch] = vertex_map

    # 2. [核心] 执行径向拟合，生成光滑 Gain Table
    logging.info("步骤3: 执行径向多项式拟合 (消除条纹与色偏)...")

    raw_gains = {}
    damp_ratio = fisheye_config.get('radius_ratio', 1.05)
    damp_width = fisheye_config.get('damping_width', 50)
    hw_max = fisheye_config.get('hw_max_gain', 7.99)

    # 取硬件限制和用户设置的较小值
    final_limit = min(max_gain, hw_max)

    for ch_name in ['R', 'Gr', 'Gb', 'B']:
        logging.info(f"  - 拟合 {ch_name} 通道曲线...")

        # 调用拟合函数，直接得到光滑的增益表
        fitted_gain = fit_radial_gain_table(
            raw_brightness_map[ch_name],
            num_v_verts, num_h_verts,
            circle_info, image_width, image_height,
            final_limit
        )

        # 3. 径向衰减 (Dampening)
        # 仅在最边缘死黑区生效，防止 Bicubic 插值溢出
        damped_gain = dampen_gains_by_geometry(
            fitted_gain, num_v_verts, num_h_verts,
            circle_info, image_width, image_height,
            damp_ratio, damp_width
        )

        raw_gains[ch_name] = np.clip(damped_gain, 1.0, final_limit).astype(np.float32)

    # [高通ISP最佳实践] Gr/Gb通道平衡，避免Demosaic迷宫纹理
    logging.info("应用Gr/Gb通道平衡（高通Chromatix最佳实践）...")
    avg_green = (raw_gains['Gr'] + raw_gains['Gb']) / 2.0
    gr_gb_diff = np.abs(raw_gains['Gr'] - raw_gains['Gb']).max()
    logging.info(f"  - 平衡前Gr/Gb最大差异: {gr_gb_diff:.4f}")
    raw_gains['Gr'] = avg_green
    raw_gains['Gb'] = avg_green
    logging.info(f"  - 平衡后Gr/Gb差异: 0.0000 (完全一致)")

    return raw_gains