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

    # [V7.0] 拟合范围扩展到 99.5%，纳入鱼眼边缘全部暗角数据
    valid_mask = (flat_val > 0.001) & (flat_r < 0.995)

    if np.sum(valid_mask) < 10:
        logging.warning("有效拟合点过少，拟合失败，返回全1矩阵")
        return np.ones_like(brightness_grid)

    train_r = flat_r[valid_mask]
    train_val = flat_val[valid_mask]

    # 3. 寻找中心最大亮度 (用于计算 Gain = Max / Fit_Val)
    max_brightness = np.max(train_val)

    # 4. [V8.0] 边缘加权 + 亮度权重
    # 鱼眼最大暗角在边缘，边缘数据准确拟合更重要
    # 用线性递增权重：中心0.4，边缘1.0，边缘比中心重要2.5x
    distance_weight = 0.4 + 0.6 * train_r
    brightness_weight = np.clip(train_val / max_brightness, 0.3, 1.0)
    weights = distance_weight * brightness_weight

    logging.info(f"  - 边缘加权拟合，权重范围: [{weights.min():.3f}, {weights.max():.3f}]")

    # 5. 分段拟合（内圈4阶 + 外圈2阶，外圈范围扩展到 99.5%）
    try:
        # 内圈拟合：0-80%半径，用4阶多项式（精确描述中心区域）
        inner_mask = train_r < 0.80
        if np.sum(inner_mask) > 10:
            coeffs_inner = np.polyfit(train_r[inner_mask], train_val[inner_mask], 4, w=weights[inner_mask])
            poly_inner = np.poly1d(coeffs_inner)
        else:
            poly_inner = None

        # 外圈拟合：70-99.5%半径，用2阶多项式（稳定处理边缘）
        outer_mask = (train_r >= 0.70) & (train_r < 0.995)
        if np.sum(outer_mask) > 10:
            coeffs_outer = np.polyfit(train_r[outer_mask], train_val[outer_mask], 2, w=weights[outer_mask])
            poly_outer = np.poly1d(coeffs_outer)
        else:
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


def cos4_compensation(radius_map, strength=0.3):
    """
    cos^4 光学暗角预补偿。

    真实鱼眼镜头暗角近似遵循 cos^4(θ) 规律，先做理论预补偿可以降低多项式拟合的难度
    （只需拟合残差，而不是整条陡峭曲线），边缘拟合精度更高。

    strength 控制预补偿强度 (0=不补偿, 1=完全cos^4补偿):
    - 设为 0.3 而非 1.0，避免在 r→1 时 cos^4 → ∞ 的数值问题
    - 对于 r=0.9: cos²(81°) ≈ 1/42 → gain^0.3 ≈ 2.5，合理
    """
    theta = np.clip(radius_map, 0, 0.98) * (np.pi / 2.0)
    cos4 = np.cos(theta) ** 4
    gain = (1.0 / (cos4 + 1e-6)) ** strength
    # 安全上限：鱼眼实际最大暗角通常不超过 8x，预补偿不应超出此范围
    gain = np.clip(gain, 1.0, 8.0)
    return gain.astype(np.float32)


def residual_2d_correction(brightness_map, gain_radial, valid_mask):
    """
    2D 残差校正：修正径向拟合未能捕捉到的非对称/非径向不均匀性。

    在径向校正后，计算校正图与目标亮度（有效区域均值）的偏差，
    生成平滑的 2D 补偿因子。
    """
    corrected = brightness_map * gain_radial

    valid_vals = corrected[valid_mask]
    if valid_vals.size < 5:
        return np.ones_like(gain_radial, dtype=np.float32)

    target = np.mean(valid_vals)

    residual = np.ones_like(corrected, dtype=np.float32)
    residual[valid_mask] = target / (corrected[valid_mask] + 1e-6)

    # 平滑：(5,5) 核在 13×17 矩阵上等效于全局平滑，避免过拟合噪声
    residual = cv2.GaussianBlur(residual, (5, 5), 0)

    # 限制修正幅度（避免极端值破坏增益表）
    residual = np.clip(residual, 0.7, 1.5)

    logging.info(f"  - 2D残差校正范围: [{residual.min():.4f}, {residual.max():.4f}]")
    return residual


def fit_ratio_smooth(ratio_map, rows, cols, circle_info, image_w, image_h):
    """
    对色度比值图 (如 R/G, B/G) 拟合光滑的二阶径向曲线。
    返回拟合后的比值，而非增益。用于 Luma+Chroma LSC 中的色度校正。
    """
    cx, cy, radius = circle_info

    step_h = image_h / (rows - 1)
    step_w = image_w / (cols - 1)
    y_idx, x_idx = np.indices((rows, cols))
    px_x = x_idx * step_w
    px_y = y_idx * step_h
    r_dist = np.sqrt((px_x - cx)**2 + (px_y - cy)**2)
    norm_r = r_dist / radius

    flat_r = norm_r.flatten()
    flat_val = ratio_map.flatten()

    valid_mask = (flat_val > 0.01) & (flat_r < 0.98)

    if np.sum(valid_mask) < 10:
        logging.warning("色度比值拟合数据不足，返回原始比值图")
        return ratio_map.copy()

    train_r = flat_r[valid_mask]
    train_val = flat_val[valid_mask]

    try:
        # [V8.0] 三阶多项式 + 边缘加权（chroma 在边缘变化更明显）
        weights = 0.5 + 0.5 * train_r
        coeffs = np.polyfit(train_r, train_val, 3, w=weights)
        poly = np.poly1d(coeffs)
        fitted_vals = poly(flat_r).reshape(rows, cols)
        fitted_vals = np.maximum(fitted_vals, 0.01)
        logging.info(f"  - 色度比值拟合范围: [{fitted_vals.min():.4f}, {fitted_vals.max():.4f}]")
        return fitted_vals.astype(np.float32)
    except Exception as e:
        logging.warning(f"色度比值拟合失败: {e}，返回原始比值图")
        return ratio_map.copy()


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
    raw_brightness_map = {}
    raw_valid_masks = {}    # Fix 2: 记录真正有像素数据的单元格

    # Fix 1: 解析预平滑核大小
    if hasattr(smooth_kernel_size, '__len__'):
        kh, kw = int(smooth_kernel_size[0]), int(smooth_kernel_size[1])
    else:
        kh = kw = int(smooth_kernel_size)
    kh = kh if kh % 2 == 1 else kh + 1
    kw = kw if kw % 2 == 1 else kw + 1
    do_presmooth = (kh > 1 or kw > 1)

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

        # Fix 2: 在平滑前记录原始有效掩码（cell_map > 0 表示有真实数据）
        valid_cell_mask = cell_map > 0

        # Fix 1: 预平滑（仅对有效单元格区域），减少测量噪声再拟合
        if do_presmooth and np.any(valid_cell_mask):
            valid_vals_mean = cell_map[valid_cell_mask].mean()
            cell_filled = cell_map.copy()
            cell_filled[~valid_cell_mask] = valid_vals_mean   # 临时填充无效单元格
            cell_smoothed = cv2.GaussianBlur(cell_filled, (kw, kh), 0)
            cell_map[valid_cell_mask] = cell_smoothed[valid_cell_mask]  # 只更新有效区域

        # 插值到顶点尺寸 (NxM -> N+1 x M+1)
        vertex_map = cv2.resize(cell_map, (num_h_verts, num_v_verts), interpolation=cv2.INTER_LINEAR)
        raw_brightness_map[ch] = vertex_map

        # Fix 2: 同步将有效掩码缩放到顶点尺寸
        vertex_valid = cv2.resize(valid_cell_mask.astype(np.float32),
                                  (num_h_verts, num_v_verts),
                                  interpolation=cv2.INTER_LINEAR) > 0.5
        raw_valid_masks[ch] = vertex_valid

    # 2. [核心] 执行径向拟合，生成光滑 Gain Table
    logging.info("步骤3: 执行 Luma+Chroma 径向拟合 (分离亮度与色度，消除色偏)...")

    raw_gains = {}
    damp_ratio = fisheye_config.get('radius_ratio', 1.05)
    damp_width = fisheye_config.get('damping_width', 50)
    hw_max = fisheye_config.get('hw_max_gain', 7.99)

    # 取硬件限制和用户设置的较小值
    final_limit = min(max_gain, hw_max)

    # --- [核心算法 V8.0] Luma + Chroma + cos^4预补偿 + 2D残差校正 ---
    # 升级点：
    #   V7.0: Luma+Chroma 分离（已解决偏色）
    #   V8.0: + cos^4预补偿（降低边缘拟合难度）+ 2D残差（修正非径向不均匀性）

    # Step 1: 计算 Luma 图 (四通道均值) 和绿色参考图
    luma_map = (raw_brightness_map['R'] + raw_brightness_map['Gr'] +
                raw_brightness_map['Gb'] + raw_brightness_map['B']) / 4.0
    green_map = (raw_brightness_map['Gr'] + raw_brightness_map['Gb']) / 2.0

    # Step 2: 生成归一化半径图
    radius_map = generate_radius_map(num_v_verts, num_h_verts,
                                     circle_info, image_width, image_height)

    # Step 3: cos^4 预补偿
    # 鱼眼暗角近似遵循 cos^4(θ)。先做理论预补偿（strength=0.3，保守值）
    # 使多项式只需拟合残差，避免在陡峭曲线末端出现拟合偏差
    logging.info("  应用 cos^4 光学模型预补偿 (strength=0.3)...")
    cos4_gain = cos4_compensation(radius_map, strength=0.3)
    luma_precomp = luma_map * cos4_gain  # 亮度预补偿后应更平坦，更易拟合

    # Step 4: 对预补偿后的 Luma 做径向拟合
    logging.info("  拟合 Luma 通道曲线（cos^4预补偿后）...")
    gain_luma_radial = fit_radial_gain_table(
        luma_precomp, num_v_verts, num_h_verts,
        circle_info, image_width, image_height, final_limit
    )

    # Step 5: 2D 残差校正（修正径向模型未能捕捉到的非对称不均匀性）
    # 在原始 luma（未预补偿）上计算残差，确保最终校正基于实测数据
    luma_valid_mask = (raw_valid_masks['R'] & raw_valid_masks['Gr'] &
                       raw_valid_masks['Gb'] & raw_valid_masks['B'])
    logging.info("  计算 2D 残差校正（修正非径向不均匀性）...")
    residual_2d = residual_2d_correction(luma_map, gain_luma_radial, luma_valid_mask)

    # 最终 Luma 增益 = 径向增益 × 2D残差
    gain_luma = gain_luma_radial * residual_2d

    # Step 6: 计算 R/G 和 B/G 色度比值图，拟合色度校正
    eps = 1e-6
    r_g_ratio = raw_brightness_map['R'] / (green_map + eps)
    b_g_ratio = raw_brightness_map['B'] / (green_map + eps)

    logging.info("  拟合 R/G 色度比值曲线（边缘加权3阶）...")
    fitted_r_g = fit_ratio_smooth(r_g_ratio, num_v_verts, num_h_verts,
                                   circle_info, image_width, image_height)
    logging.info("  拟合 B/G 色度比值曲线（边缘加权3阶）...")
    fitted_b_g = fit_ratio_smooth(b_g_ratio, num_v_verts, num_h_verts,
                                   circle_info, image_width, image_height)

    # Step 7: 以圆心处色度比值为参考（保持中心白平衡不变）
    cy_idx = num_v_verts // 2
    cx_idx = num_h_verts // 2
    center_r_g = float(fitted_r_g[cy_idx, cx_idx])
    center_b_g = float(fitted_b_g[cy_idx, cx_idx])
    logging.info(f"  圆心色度比值: R/G={center_r_g:.4f}, B/G={center_b_g:.4f}")

    # Step 8: 色度校正系数（圆心=1.0，边缘按实际偏差补偿）
    chroma_R = center_r_g / (fitted_r_g + eps)
    chroma_B = center_b_g / (fitted_b_g + eps)
    logging.info(f"  R色度校正范围: [{chroma_R.min():.4f}, {chroma_R.max():.4f}]")
    logging.info(f"  B色度校正范围: [{chroma_B.min():.4f}, {chroma_B.max():.4f}]")

    # Step 9: 组合最终增益 = Luma × Chroma
    raw_gains['R']  = gain_luma * chroma_R
    raw_gains['Gr'] = gain_luma.copy()
    raw_gains['Gb'] = gain_luma.copy()
    raw_gains['B']  = gain_luma * chroma_B

    # Step 10: 边缘衰减 + 硬件钳位
    for ch_name in ['R', 'Gr', 'Gb', 'B']:
        damped_gain = dampen_gains_by_geometry(
            raw_gains[ch_name], num_v_verts, num_h_verts,
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

    return raw_gains, raw_brightness_map, raw_valid_masks


def generate_radius_map(rows, cols, circle_info, image_w, image_h):
    """
    生成归一化半径图 (0.0 ~ 1.0)

    参数:
        rows, cols: 网格行数和列数
        circle_info: (cx, cy, r) 圆心坐标和半径
        image_w, image_h: 图像宽度和高度

    返回:
        radius_map: 归一化半径图，形状为 (rows, cols)
    """
    cx, cy, r = circle_info

    step_h = image_h / (rows - 1)
    step_w = image_w / (cols - 1)

    y, x = np.indices((rows, cols))

    px = x * step_w
    py = y * step_h

    dist = np.sqrt((px - cx)**2 + (py - cy)**2)

    r_norm = dist / r

    return r_norm


