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
    【核心算法 V6.0】改进版径向多项式拟合
    将嘈杂的网格数据拟合成光滑的径向曲线，生成无条纹、无色偏的增益表。

    V6.0 改进:
    - 加权拟合：给高质量数据更大权重，减少噪点影响
    - 分段拟合：内圈用4阶，外圈用2阶，提升边缘精度
    - 平滑拼接：避免分段边界的不连续
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

    # [V6.0 改进] 扩展拟合范围到 98%，使用分段拟合处理边缘
    # 不再丢弃边缘8%数据，而是用更稳定的方法处理
    valid_mask = (flat_val > 0.001) & (flat_r < 0.98)

    if np.sum(valid_mask) < 10:
        logging.warning("有效拟合点过少，拟合失败，返回全1矩阵")
        return np.ones_like(brightness_grid)

    train_r = flat_r[valid_mask]
    train_val = flat_val[valid_mask]

    # 3. 寻找中心最大亮度 (用于计算 Gain = Max / Fit_Val)
    max_brightness = np.max(train_val)

    # 4. [V6.0 改进] 计算加权系数
    # 给高质量数据更大权重：中心权重大，边缘权重小；亮点权重大，暗点权重小
    weights = np.ones_like(train_val)

    # 权重1：距离权重（高斯分布，中心权重大）
    distance_weight = np.exp(-((train_r - 0.5) / 0.35)**2)
    weights *= distance_weight

    # 权重2：亮度权重（暗点权重降低，因为噪声大）
    brightness_weight = np.clip(train_val / max_brightness, 0.3, 1.0)
    weights *= brightness_weight

    logging.info(f"  - 使用加权拟合，权重范围: [{weights.min():.3f}, {weights.max():.3f}]")

    # 5. [V6.0 改进] 分段拟合 + 加权拟合
    try:
        # 内圈拟合：0-80%半径，用4阶多项式（精确描述中心区域）
        inner_mask = train_r < 0.80
        if np.sum(inner_mask) > 10:
            coeffs_inner = np.polyfit(train_r[inner_mask], train_val[inner_mask], 4, w=weights[inner_mask])
            poly_inner = np.poly1d(coeffs_inner)
        else:
            # 内圈数据不足，回退到全局拟合
            coeffs_inner = None
            poly_inner = None

        # 外圈拟合：70-98%半径，用2阶多项式（稳定处理边缘）
        outer_mask = (train_r >= 0.70) & (train_r < 0.98)
        if np.sum(outer_mask) > 10:
            coeffs_outer = np.polyfit(train_r[outer_mask], train_val[outer_mask], 2, w=weights[outer_mask])
            poly_outer = np.poly1d(coeffs_outer)
        else:
            # 外圈数据不足，回退到全局拟合
            coeffs_outer = None
            poly_outer = None

        # 6. 重建光滑的增益表（分段拼接）
        fitted_brightness = np.zeros_like(flat_r)

        if poly_inner is not None and poly_outer is not None:
            # 分段拼接模式
            for i, r in enumerate(flat_r):
                if r < 0.75:
                    # 内圈：使用4阶多项式
                    fitted_brightness[i] = poly_inner(r)
                elif r > 0.85:
                    # 外圈：使用2阶多项式
                    fitted_brightness[i] = poly_outer(r)
                else:
                    # 过渡区（75%-85%）：线性混合
                    blend_weight = (r - 0.75) / 0.10
                    fitted_brightness[i] = poly_inner(r) * (1 - blend_weight) + poly_outer(r) * blend_weight

            logging.info("  - 使用分段拟合：内圈4阶 + 外圈2阶 + 平滑拼接")
        else:
            # 回退到全局4阶拟合
            coeffs_global = np.polyfit(train_r, train_val, 4, w=weights)
            poly_global = np.poly1d(coeffs_global)
            fitted_brightness = poly_global(flat_r)
            logging.info("  - 使用全局4阶加权拟合（分段数据不足）")

        fitted_brightness = fitted_brightness.reshape(rows, cols)

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