# lsc_calibration_project/lsc/validation.py

import os
import logging
import numpy as np

def validate_config(config_module):
    """
    验证配置参数的合法性。
    在运行主程序前调用，确保所有参数在合理范围内。

    参数:
        config_module: 导入的 config 模块对象

    返回:
        tuple: (is_valid, error_messages)
            - is_valid (bool): 配置是否有效
            - error_messages (list): 错误信息列表
    """
    errors = []

    # 1. 检查输入文件是否存在
    if not os.path.exists(config_module.RAW_IMAGE_PATH):
        errors.append(f"RAW图像文件不存在: {config_module.RAW_IMAGE_PATH}")

    # 2. 检查图像尺寸参数
    if config_module.IMAGE_WIDTH <= 0 or config_module.IMAGE_HEIGHT <= 0:
        errors.append(f"图像尺寸无效: {config_module.IMAGE_WIDTH}x{config_module.IMAGE_HEIGHT}")

    # 3. 检查传感器参数
    if config_module.SENSOR_MAX_VALUE <= 0:
        errors.append(f"传感器最大值无效: {config_module.SENSOR_MAX_VALUE}")

    # 4. 检查黑电平
    for ch, bl in config_module.BLACK_LEVELS.items():
        if bl < 0 or bl >= config_module.SENSOR_MAX_VALUE:
            errors.append(f"黑电平 {ch} 超出范围 [0, {config_module.SENSOR_MAX_VALUE}): {bl}")

    # 5. 检查网格配置
    if config_module.GRID_ROWS <= 0 or config_module.GRID_COLS <= 0:
        errors.append(f"网格配置无效: {config_module.GRID_ROWS}x{config_module.GRID_COLS}")

    if config_module.GRID_ROWS > 20 or config_module.GRID_COLS > 20:
        logging.warning(f"网格尺寸较大 ({config_module.GRID_ROWS}x{config_module.GRID_COLS})，可能导致过拟合")

    # 6. 检查增益参数
    if config_module.MAX_GAIN < 1.0:
        errors.append(f"MAX_GAIN 不能小于1.0: {config_module.MAX_GAIN}")

    if config_module.MAX_GAIN > 15.0:
        logging.warning(f"MAX_GAIN过大 ({config_module.MAX_GAIN})，可能放大噪声")

    hw_limit = config_module.HW_MAX_GAIN_FLOAT
    if hw_limit < 1.0 or hw_limit > 10.0:
        errors.append(f"硬件增益限制不合理: {hw_limit}")

    # 7. 检查鱼眼参数
    if hasattr(config_module, 'FISHEYE_GAIN_RADIUS_RATIO'):
        ratio = config_module.FISHEYE_GAIN_RADIUS_RATIO
        if ratio < 0.8 or ratio > 1.2:
            errors.append(f"鱼眼半径比例超出合理范围 [0.8, 1.2]: {ratio}")

    if hasattr(config_module, 'FISHEYE_DAMPING_WIDTH_PIXEL'):
        width = config_module.FISHEYE_DAMPING_WIDTH_PIXEL
        if width < 10 or width > 500:
            logging.warning(f"衰减宽度可能不合理: {width}像素")

    # 8. 检查平滑参数
    if hasattr(config_module, 'V3_POST_SMOOTH_KSIZE'):
        ksize = config_module.V3_POST_SMOOTH_KSIZE
        if isinstance(ksize, tuple):
            if ksize[0] < 0 or ksize[1] < 0:
                errors.append(f"平滑核大小无效: {ksize}")
        elif ksize < 0:
            errors.append(f"平滑核大小无效: {ksize}")

    # 9. 检查输出目录是否可写
    output_dir = config_module.OUTPUT_DIR
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        errors.append(f"输出目录不可写 ({output_dir}): {e}")

    # 10. 检查 RAW 文件大小是否匹配预期
    if os.path.exists(config_module.RAW_IMAGE_PATH):
        expected_size = config_module.IMAGE_WIDTH * config_module.IMAGE_HEIGHT * 2  # 16-bit
        actual_size = os.path.getsize(config_module.RAW_IMAGE_PATH)

        if actual_size != expected_size:
            logging.warning(
                f"RAW文件大小不匹配。"
                f"预期: {expected_size} 字节 ({config_module.IMAGE_WIDTH}x{config_module.IMAGE_HEIGHT}), "
                f"实际: {actual_size} 字节"
            )

    # 返回验证结果
    is_valid = len(errors) == 0

    if is_valid:
        logging.info("[成功] 配置验证通过")
    else:
        logging.error("[失败] 配置验证失败:")
        for err in errors:
            logging.error(f"  - {err}")

    return is_valid, errors


def validate_gain_table(gain_matrix, channel_name, min_gain=1.0, max_gain=8.0):
    """
    验证增益表的质量。

    参数:
        gain_matrix (np.array): 增益矩阵
        channel_name (str): 通道名称 ('R', 'Gr', 'Gb', 'B')
        min_gain (float): 允许的最小增益
        max_gain (float): 允许的最大增益

    返回:
        dict: 验证结果统计
    """
    results = {
        'channel': channel_name,
        'valid': True,
        'warnings': [],
        'stats': {}
    }

    # 1. 检查范围
    min_val = np.min(gain_matrix)
    max_val = np.max(gain_matrix)
    mean_val = np.mean(gain_matrix)
    std_val = np.std(gain_matrix)

    results['stats'] = {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val
    }

    if min_val < min_gain or max_val > max_gain:
        results['valid'] = False
        results['warnings'].append(
            f"增益超出范围 [{min_gain}, {max_gain}]: [{min_val:.3f}, {max_val:.3f}]"
        )

    # 2. 检查是否有异常值（过大的梯度）
    dy, dx = np.gradient(gain_matrix)
    gradient_mag = np.sqrt(dx**2 + dy**2)
    max_gradient = np.max(gradient_mag)

    if max_gradient > 0.5:  # 相邻网格增益变化超过0.5
        results['warnings'].append(
            f"检测到过大梯度: {max_gradient:.3f} (可能存在伪影)"
        )

    # 3. 检查中心区域增益是否接近1.0
    rows, cols = gain_matrix.shape
    center_region = gain_matrix[
        rows//3:2*rows//3,
        cols//3:2*cols//3
    ]
    center_mean = np.mean(center_region)

    if center_mean < 0.9 or center_mean > 1.5:
        results['warnings'].append(
            f"中心区域平均增益异常: {center_mean:.3f} (预期接近1.0)"
        )

    return results


def calculate_uniformity_metrics(image, mask, center_info):
    """
    计算图像均匀性指标。

    参数:
        image (np.array): BGR或灰度图像 (归一化到 [0, 1])
        mask (np.array): 有效区域遮罩
        center_info (tuple): (cx, cy, radius)

    返回:
        dict: 均匀性指标
    """
    cx, cy, radius = center_info
    h, w = image.shape[:2]

    # 创建距离图
    y_idx, x_idx = np.indices((h, w))
    dist_map = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    norm_r = dist_map / radius

    # 转换为灰度
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # 定义区域
    center_mask = (norm_r < 0.3) & (mask > 0.5)
    edge_mask = (norm_r > 0.7) & (norm_r < 0.95) & (mask > 0.5)

    # 计算均匀性
    center_mean = np.mean(gray[center_mask]) if np.sum(center_mask) > 0 else 0
    edge_mean = np.mean(gray[edge_mask]) if np.sum(edge_mask) > 0 else 0

    uniformity_ratio = edge_mean / center_mean if center_mean > 1e-6 else 0

    # 计算色彩一致性（如果是彩色图像）
    color_std = {}
    if len(image.shape) == 3:
        for i, ch in enumerate(['B', 'G', 'R']):
            valid_region = image[:, :, i][mask > 0.5]
            color_std[ch] = np.std(valid_region)

    metrics = {
        'center_brightness': center_mean,
        'edge_brightness': edge_mean,
        'uniformity_ratio': uniformity_ratio,
        'color_std': color_std
    }

    # 评估等级
    if uniformity_ratio > 0.95:
        grade = 'Excellent'
    elif uniformity_ratio > 0.90:
        grade = 'Good'
    elif uniformity_ratio > 0.85:
        grade = 'Acceptable'
    else:
        grade = 'Poor'

    metrics['grade'] = grade

    return metrics
