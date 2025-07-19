# lsc_calibration_project/lsc/gain_utils.py

import numpy as np
import cv2
import logging

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Scipy库未安装，高级外插平滑功能将不可用。将回退到标准高斯平滑。")
    logging.warning("请运行 'pip install scipy' 来安装。")


def extrapolate_and_smooth_gains(gain_matrix, gaussian_ksize=5):
    """
    通过先外插再平滑的方式，智能地处理增益矩阵。
    此方法可以有效填充因亮度过低而产生的无效网格(值为1.0)，避免在平滑时污染有效区域。

    参数:
        gain_matrix (np.array): 原始LSC增益矩阵，其中无效区域的值为1.0。
        gaussian_ksize (int): 高斯模糊的核大小，必须是奇数。

    返回:
        np.array: 经过外插和平滑处理后的高质量增益矩阵。
    """
    if not SCIPY_AVAILABLE:
        # 如果scipy不可用，则执行简单的高斯模糊
        logging.warning("回退到标准高斯平滑。")
        if gaussian_ksize % 2 == 0: gaussian_ksize += 1
        return cv2.GaussianBlur(gain_matrix.astype(np.float32), (gaussian_ksize, gaussian_ksize), 0)

    logging.info("执行高级平滑：先外插，后高斯模糊...")
    if gaussian_ksize % 2 == 0: gaussian_ksize += 1
        
    filled_matrix = gain_matrix.astype(np.float32)

    # 制作一个掩码，标记出无效区域（值为1.0）
    invalid_mask = (np.abs(gain_matrix - 1.0) < 1e-6)
    
    if not np.any(invalid_mask):
        logging.info("  - 未检测到无效区域，执行标准高斯平滑。")
        return cv2.GaussianBlur(filled_matrix, (gaussian_ksize, gaussian_ksize), 0)

    # 使用Scipy的cKDTree来高效查找最近邻
    valid_coords = np.argwhere(~invalid_mask)
    invalid_coords = np.argwhere(invalid_mask)
    
    # 用所有有效点构建一个KD树，用于快速查询
    tree = cKDTree(valid_coords)
    
    # 为所有无效点查询其最近的有效点的索引
    _, indices = tree.query(invalid_coords)
    
    # 根据索引找到最近的有效点的具体坐标
    nearest_valid_coords = valid_coords[indices]
    
    # 将无效点的值替换为它最近的有效点的值（外插）
    filled_matrix[invalid_coords[:, 0], invalid_coords[:, 1]] = filled_matrix[nearest_valid_coords[:, 0], nearest_valid_coords[:, 1]]
    
    logging.info(f"  - 外插完成，填充了 {len(invalid_coords)} 个无效网格。")

    # 在已经没有“悬崖”的矩阵上进行高斯平滑
    final_smoothed_matrix = cv2.GaussianBlur(filled_matrix, (gaussian_ksize, gaussian_ksize), 0)
    logging.info(f"  - 在外插后的矩阵上完成高斯平滑 (核大小: {gaussian_ksize})。")
    
    return final_smoothed_matrix

def symmetrize_table(table):
    """
    通过取对称点平均值的方式，强制使增益矩阵中心对称。
    这可以消除一些由灯光或摆放不完美导致的轻微不对称。

    参数:
        table (np.array): 输入的增益矩阵。

    返回:
        np.array: 中心对称化处理后的矩阵。
    """
    rows, cols = table.shape
    symmetrized_table = table.copy()

    # 遍历矩阵的左上角四分之一（包括中心行和列）
    for r in range((rows + 1) // 2):
        for c in range(cols):
            # 找到对称点
            sym_r, sym_c = rows - 1 - r, cols - 1 - c
            # 计算平均值
            avg_val_row = (symmetrized_table[r, c] + symmetrized_table[sym_r, c]) / 2.0
            symmetrized_table[r, c] = avg_val_row
            symmetrized_table[sym_r, c] = avg_val_row

    # 再次处理以确保列对称
    for c in range((cols + 1) // 2):
        for r in range(rows):
            sym_c = cols - 1 - c
            avg_val_col = (symmetrized_table[r, c] + symmetrized_table[r, sym_c]) / 2.0
            symmetrized_table[r, c] = avg_val_col
            symmetrized_table[r, sym_c] = avg_val_col
            
    logging.info("增益矩阵已完成中心对称处理。")
    return symmetrized_table


def create_falloff_map(rows, cols, falloff_at_edge):
    """
    创建一个从中心到边缘线性变化的衰减因子地图。
    中心值为1.0，边缘值由falloff_at_edge决定。

    参数:
        rows (int): 网格行数。
        cols (int): 网格列数。
        falloff_at_edge (float): 边缘的衰减值 (如0.9)。

    返回:
        np.array: 衰减图。
    """
    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
    # 计算到角落的最大距离
    max_dist = np.sqrt(center_r**2 + center_c**2)
    if max_dist == 0: max_dist = 1 # 防止行列都为1时除以0

    falloff_map = np.ones((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
            # 根据距离进行线性插值
            ratio = dist / max_dist
            falloff_map[r, c] = 1.0 * (1 - ratio) + falloff_at_edge * ratio
    
    logging.info(f"已创建亮度衰减图, 边缘因子为: {falloff_at_edge:.2f}")
    return falloff_map