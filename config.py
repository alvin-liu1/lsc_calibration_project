# lsc_calibration_project/config.py

import cv2

# ==============================================================================
# LSC 标定 - 全局配置文件 (V5.0 径向拟合自动化版)
# ==============================================================================

# --- 1. 输入与输出配置 ---
# 请修改为您当前的 RAW 文件路径
RAW_IMAGE_PATH = 'input/qbchdr.raw'
OUTPUT_DIR = 'output'

# RAW 文件格式配置
# 支持的格式:
#   'auto'       - 自动检测格式（推荐）
#   'plain'      - Plain RAW (16-bit容器存储10/12-bit数据)
#   'mipi_raw10' - MIPI RAW10 (4个10-bit像素打包成5字节)
#   'mipi_raw12' - MIPI RAW12 (2个12-bit像素打包成3字节)
RAW_FORMAT = 'plain'  # qbchdr.raw是Plain RAW格式

# --- 2. 图像基本属性 ---
IMAGE_WIDTH = 2900
IMAGE_HEIGHT = 2900
# Bayer 格式: RGGB, BGGR, GBRG, GRBG
# 请根据 Sensor 规格书确认 (高通通常是 MIPI 顺序)
# 修正: 根据测试结果，此sensor使用GRBG pattern
BAYER_PATTERN = 'GRBG'

# --- 3. 传感器参数 ---
# 【关键】请务必填入准确的黑电平 (Black Level)
# 如果扣除不干净，暗部会有严重的偏色
BLACK_LEVELS = {'R': 64, 'Gr': 64, 'Gb': 64, 'B': 64}
SENSOR_MAX_VALUE = 1023.0

# [高通硬件限制]
# 13uQ10 格式: Gain 1.0 = 1024
# 硬件最大支持约为 7.99 (8191 / 1024)
HW_MAX_GAIN_FLOAT = 8191.0 / 1024.0

# --- 4. LSC 网格配置 ---
# 高通 VFE 标准配置通常为:
# 17x13 顶点 (Vertices) -> 对应 16x12 网格 (Cells)
# 如果您的驱动代码里定义的是 [13][17]，请保持 GRID_COLS = 16
# 如果您的驱动代码里定义的是 [13][16]，请改为 GRID_COLS = 15
GRID_ROWS = 12
GRID_COLS = 16

# 统计阈值 (用于排除坏点/噪点)
MIN_PIXELS_PER_GRID = 100
VALID_GRID_THRESHOLD_RATIO = 0.1

# --- 5. 核心算法参数 (Fisheye Auto-Fitting) ---

# [修改] 自动拟合模式下，不再需要手动 Falloff
# 新的 calibration.py 会自动拟合 R/G/B 的光衰曲线，
# 从而自动解决边缘发绿/发红问题。这里重置为 1.0 即可。
FALLOFF_FACTOR = 1.0

# [修改] 最大增益限制
# 建议设大一点 (如 10.0 ~ 12.0)，让拟合曲线能自然延伸到最暗角。
# 最终输出时会自动钳位到 HW_MAX_GAIN_FLOAT (7.99)。
MAX_GAIN = 4.0

# [修改 V3.0 - 全景拼接优化] 径向保护 (Dampening)
# 重要变更：改为 1.0（从圆边缘开始衰减）
#
# 为什么从 1.05 改为 1.0？
# - 全景拼接场景：需要圆外区域增益=1.0（保持原始暗值，不要清零）
# - 1.0 = 从鱼眼圆边缘开始，增益逐渐从 LSC校正值 → 1.0
# - 圆外完全是1.0增益，保留原始数据供拼接算法使用
#
# [V3.2 最终优化] 精确控制圆外衰减
# 策略：仅衰减圆外区域（radius以外），圆内完全保留LSC校正
#
# 计算逻辑：
# - 圆心=(1448, 1413), 半径=1422
# - 四角距离约2023像素
# - 需要从1422像素（圆边缘）→ 2023+像素（四角及以外）平滑衰减到1.0
#
FISHEYE_GAIN_RADIUS_RATIO = 1.0  # 从100%半径（圆边缘）开始衰减

# 衰减过渡区宽度：从圆边缘到四角的距离
# 四角到圆心≈2023，半径=1422，差值=601
# 使用稍大的过渡区（约700像素）确保四角完全衰减到1.0
FISHEYE_DAMPING_WIDTH_PIXEL = 700  # 从圆边缘开始，700像素过渡区衰减到1.0

# 无效区域的目标增益 (1.0 = 保持原样/死黑)
INVALID_AREA_TARGET_GAIN = 1.0

# --- 6. 平滑参数 ---
# 预平滑 (计算前)
V3_PRE_SMOOTH_KSIZE = (5, 5)

# [修改] 后平滑 (计算后)
# 因为现在的“多项式拟合”生成的曲线本身就是数学级光滑的，
# 所以不需要很大的后平滑核，5 或 3 足够消除微小误差。
V3_POST_SMOOTH_KSIZE = 5

APPLY_SYMMETRY = True

# --- 7. 交互配置 ---
MASK_FEATHER_PIXELS = 30
MANUAL_ADJUST_STEP = 5
USE_MANUAL_CIRCLE_SELECTION = True