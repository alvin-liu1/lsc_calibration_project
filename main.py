# lsc_calibration_project/main.py

import os
import sys
import logging
import cv2
import numpy as np

# 导入配置和自定义模块
import config
from lsc import bayer_utils, image_utils, gain_utils, visualization, calibration

def setup_logging():
    """配置日志系统，同时输出到控制台和文件。"""
    log_dir = os.path.join(config.OUTPUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'calibration_run.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("日志系统已启动。")

def main():
    """LSC标定主流程函数。"""
    # 1. 初始化
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    setup_logging()
    
    logging.info("="*50)
    logging.info("LSC 标定脚本启动 (V2 - 修复版)")
    logging.info(f"输入文件: {config.RAW_IMAGE_PATH}")
    logging.info(f"输出目录: {os.path.abspath(config.OUTPUT_DIR)}")
    logging.info("="*50)

    # 2. 读取原始Bayer图像
    original_bayer_16bit = bayer_utils.read_raw_bayer_image(
        config.RAW_IMAGE_PATH, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, bit_depth=10
    )
    if original_bayer_16bit is None:
        logging.error("无法读取RAW文件，程序终止。")
        return

    h, w = original_bayer_16bit.shape

    # 3. 准备预览图并获取有效区域掩码
    preview_bayer_8bit = (np.clip(original_bayer_16bit, 0, 1023) / 4).astype(np.uint8)
    preview_rgb_float = cv2.cvtColor(preview_bayer_8bit, config.BAYER_PATTERN).astype(np.float32) / 255.0
    
    if config.USE_MANUAL_CIRCLE_SELECTION:
        temp_display_img = image_utils.simple_white_balance(preview_rgb_float.copy())
        feathered_mask, circle_info = visualization.get_manual_circle_mask(
            temp_display_img, config.MASK_FEATHER_PIXELS, config.OUTPUT_DIR, config.MANUAL_ADJUST_STEP
        )
        logging.info(f"手动选择已确认: 圆心=({circle_info[0]},{circle_info[1]}), 半径={circle_info[2]}")
    else:
        feathered_mask = np.ones((h, w), dtype=np.float32)
        circle_info = (w // 2, h // 2, min(w, h) // 2)
        logging.info("未使用手动选择，使用整个图像作为有效区域。")

    # 4. 核心计算流程
    logging.info("步骤1: 提取Bayer通道, 进行黑电平校正和归一化...")
    bayer_channels_float = bayer_utils.extract_bayer_channels(
        original_bayer_16bit, config.BAYER_PATTERN, config.BLACK_LEVELS, config.SENSOR_MAX_VALUE
    )

    raw_gain_matrices = calibration.calculate_lsc_gains(
        bayer_channels_float, config.GRID_ROWS, config.GRID_COLS, circle_info,
        config.MIN_PIXELS_PER_GRID, config.VALID_GRID_THRESHOLD_RATIO,
        config.FALLOFF_FACTOR, config.MAX_GAIN
    )

    # 5. 增益矩阵后处理
    logging.info("步骤4: 对增益矩阵进行后处理（外插平滑、对称化）...")
    final_gain_matrices = {}
    for ch, matrix in raw_gain_matrices.items():
        logging.info(f"--- 处理 {ch} 通道增益 ---")
        smoothed_matrix = gain_utils.extrapolate_and_smooth_gains(matrix, config.EXTRAPOLATE_SMOOTH_KSIZE)
        if config.APPLY_SYMMETRY:
            smoothed_matrix = gain_utils.symmetrize_table(smoothed_matrix)
        final_gain_matrices[ch] = np.clip(smoothed_matrix, 1.0, config.MAX_GAIN)

    # 6. 将增益应用到图像
    logging.info("步骤5: 将最终增益应用回Bayer图像...")
    full_size_gains = {ch: cv2.resize(matrix, (w, h), interpolation=cv2.INTER_LINEAR) 
                       for ch, matrix in final_gain_matrices.items()}
    
    avg_bl = np.mean(list(config.BLACK_LEVELS.values()))
    bayer_blc_float = np.maximum(0, original_bayer_16bit.astype(np.float32) - avg_bl)

    compensated_bayer_float = bayer_utils.apply_gains_to_bayer(
        bayer_blc_float, full_size_gains, config.BAYER_PATTERN
    )

    # 7. 生成并保存最终结果
    logging.info("步骤6: 生成最终可视化图像并保存所有结果...")
    
    max_val_after_blc = config.SENSOR_MAX_VALUE - avg_bl
    original_bayer_8bit = (np.clip(bayer_blc_float, 0, max_val_after_blc) * (255.0 / max_val_after_blc)).astype(np.uint8)
    compensated_bayer_8bit = (np.clip(compensated_bayer_float, 0, max_val_after_blc) * (255.0 / max_val_after_blc)).astype(np.uint8)

    original_rgb_no_wb = cv2.cvtColor(original_bayer_8bit, config.BAYER_PATTERN).astype(np.float32) / 255.0
    compensated_rgb_no_wb = cv2.cvtColor(compensated_bayer_8bit, config.BAYER_PATTERN).astype(np.float32) / 255.0
    
    mask_3d = np.stack([feathered_mask] * 3, axis=-1)
    original_rgb_no_wb *= mask_3d
    compensated_rgb_no_wb *= mask_3d

    original_rgb_wb = image_utils.simple_white_balance(original_rgb_no_wb.copy(), feathered_mask)
    compensated_rgb_wb = image_utils.simple_white_balance(compensated_rgb_no_wb.copy(), feathered_mask)

    base_filename = os.path.splitext(os.path.basename(config.RAW_IMAGE_PATH))[0]
    img_dir = os.path.join(config.OUTPUT_DIR, 'images')
    visualization.save_final_images(
        [original_rgb_no_wb, original_rgb_wb, compensated_rgb_no_wb, compensated_rgb_wb],
        [f"{base_filename}_1_original_no_wb", 
         f"{base_filename}_2_original_wb_only",
         f"{base_filename}_3_compensated_lsc_only",
         f"{base_filename}_4_final_result_lsc_wb"],
        img_dir
    )

    visualization.save_gain_tables(final_gain_matrices, base_filename, config.OUTPUT_DIR)
    heatmap_dir = os.path.join(config.OUTPUT_DIR, 'heatmaps')
    for ch, matrix in final_gain_matrices.items():
        visualization.plot_gain_heatmap(
            matrix, ch, 
            os.path.join(heatmap_dir, f"{base_filename}_{ch}_heatmap.png")
        )

    # 8. 【新增】调用最终的综合分析绘图函数
    image_results_for_plot = {
        'original_no_wb': original_rgb_no_wb,
        'original_wb': original_rgb_wb,
        'compensated_no_wb': compensated_rgb_no_wb,
        'compensated_wb': compensated_rgb_wb
    }
    visualization.create_and_save_analysis_plots(image_results_for_plot, base_filename, config.OUTPUT_DIR)

    logging.info("\n" + "="*50)
    logging.info("所有任务已成功完成！")
    logging.info(f"所有输出文件已保存在目录: {os.path.abspath(config.OUTPUT_DIR)}")
    logging.info("="*50)


if __name__ == '__main__':
    main()