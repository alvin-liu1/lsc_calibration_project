# lsc_calibration_project/config.py

import cv2

# ==============================================================================
# LSC 标定 - 全局配置文件
# ==============================================================================

# --- 1. 输入与输出配置 ---
RAW_IMAGE_PATH = 'input/25k.raw'  # 输入的RAW图像路径
OUTPUT_DIR = 'output'                 # 所有输出文件的根目录

# --- 2. 图像基本属性 ---
IMAGE_WIDTH = 1256
IMAGE_HEIGHT = 1256
# 【重要】根据你的Sensor选择正确的Bayer Pattern
# 可选项:
# cv2.COLOR_BayerBG2BGR_VNG  (BGGR)
# cv2.COLOR_BayerGB2BGR_VNG  (GBRG)
# cv2.COLOR_BayerRG2BGR_VNG  (RGGB)
# cv2.COLOR_BayerGR2BGR_VNG  (GRBG)
BAYER_PATTERN = cv2.COLOR_BayerGR2BGR_VNG

# --- 3. 传感器参数 ---
# 【重要】请务必测量并使用每个通道真实的黑电平 (Black Level)
BLACK_LEVELS = {'R': 64, 'Gr': 64, 'Gb': 64, 'B': 64}
# 图像数据的位深，用于归一化。10-bit RAW 的最大值是 1023
SENSOR_MAX_VALUE = 1023.0

# --- 4. LSC 核心算法参数 ---
# LSC标定网格的尺寸 (行 x 列)
GRID_ROWS = 13
GRID_COLS = 17
# 用于保证边缘亮度统计稳定性的最小像素数
MIN_PIXELS_PER_GRID = 2
# 过暗网格的放弃阈值 (相对于中心亮度的比例, 建议 0.1-0.4)
# 如果一个网格的平均亮度低于 中心亮度 * 此阈值，则该网格被认为是无效的
VALID_GRID_THRESHOLD_RATIO = 0.3
# 全局最大增益限制，防止边缘噪声被过度放大
MAX_GAIN = 3.0

# --- 5. 效果微调与高级功能 ---
# 亮度衰减因子，用于抑制边缘过亮。值越小，边缘提亮越保守 (建议 0.7-1.0)
# 1.0 代表完全校正到中心亮度，< 1.0 代表边缘亮度适当低于中心
FALLOFF_FACTOR = 1.0

# 是否对增益表应用中心对称处理，可以使效果更规整
APPLY_SYMMETRY = True

# 高级增益外插与平滑的核大小 (必须是奇数)
# 用于填充无效网格并使增益过渡更自然
EXTRAPOLATE_SMOOTH_KSIZE = 5

# --- 6. 可视化与交互配置 ---
# 手动选择圆形有效区域时，边缘羽化的像素宽度
MASK_FEATHER_PIXELS = 30
# 手动微调圆形区域时，每次按键移动/缩放的步长
MANUAL_ADJUST_STEP = 1
# 是否启用手动选择有效区域的功能。对于鱼眼镜头，强烈建议保持为 True
USE_MANUAL_CIRCLE_SELECTION = True