# lsc_calibration_project/main.py

import os
import sys
import logging
import cv2
import numpy as np

# 导入配置和自定义模块
import config
from lsc import bayer_utils, image_utils, gain_utils, visualization, calibration, validation

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
    logging.info("LSC 标定脚本启动 (V2.2 - 高通ISP & 鱼眼适配版)")
    logging.info(f"输入文件: {config.RAW_IMAGE_PATH}")
    logging.info(f"输出目录: {os.path.abspath(config.OUTPUT_DIR)}")
    logging.info("="*50)

    # 1.5. 配置验证
    logging.info("\n正在验证配置参数...")
    is_valid, errors = validation.validate_config(config)
    if not is_valid:
        logging.error("配置验证失败，请修正以下错误后重试:")
        for err in errors:
            logging.error(f"  - {err}")
        return

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

    # --- [关键新增] ---
    # 在 main 函数中统一生成 Hard Mask (硬遮罩)
    # 这个 0/1 遮罩将用于 LSC 计算 和 LSC 应用
    logging.info("正在生成用于Bayer域校正的硬遮罩 (Hard Mask)...")
    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hard_mask, (circle_info[0], circle_info[1]), circle_info[2], 1, -1)
    # -------------------

    # 4. 核心计算流程
    logging.info("步骤1: 提取Bayer通道, 进行黑电平校正和归一化...")
    bayer_channels_float = bayer_utils.extract_bayer_channels(
        original_bayer_16bit, config.BAYER_PATTERN, config.BLACK_LEVELS, config.SENSOR_MAX_VALUE
    )

    # --- [修复点 1] 补全被漏掉的 fisheye_cfg 定义 ---
    fisheye_cfg = {
        'radius_ratio': getattr(config, 'FISHEYE_GAIN_RADIUS_RATIO', 0.90),
        'damping_width': getattr(config, 'FISHEYE_DAMPING_WIDTH_PIXEL', 250),
        'hw_max_gain': getattr(config, 'HW_MAX_GAIN_FLOAT', 7.99)
    }

    # --- [修复点 2] 修复重复参数和逗号缺失 ---
    raw_gain_matrices = calibration.calculate_lsc_gains(
        bayer_channels_float, config.GRID_ROWS, config.GRID_COLS,
        hard_mask,
        config.MIN_PIXELS_PER_GRID, config.VALID_GRID_THRESHOLD_RATIO,
        config.FALLOFF_FACTOR, config.MAX_GAIN,
        circle_info=circle_info,
        image_width=w,
        image_height=h,
        fisheye_config=fisheye_cfg,
        smooth_kernel_size=config.V3_PRE_SMOOTH_KSIZE  # 这里修复了
    )

    # 5. 增益矩阵后处理
    logging.info("步骤4: 对增益矩阵进行后处理（外插平滑、对称化）...")
    final_gain_matrices = {}
    for ch, matrix in raw_gain_matrices.items():
        logging.info(f"--- 处理 {ch} 通道增益 ---")
        smoothed_matrix = gain_utils.extrapolate_and_smooth_gains(matrix, config.V3_POST_SMOOTH_KSIZE)
        if config.APPLY_SYMMETRY:
            smoothed_matrix = gain_utils.symmetrize_table(smoothed_matrix)

        # [修改] 最终安全钳位
        hw_max_gain = 8191.0 / 1024.0
        final_max_gain = min(config.MAX_GAIN, hw_max_gain)
        final_gain_matrices[ch] = np.clip(smoothed_matrix, 1.0, final_max_gain)

        # [新增] 验证增益表质量
        val_result = validation.validate_gain_table(final_gain_matrices[ch], ch, 1.0, final_max_gain)
        logging.info(f"  增益表统计: Min={val_result['stats']['min']:.3f}, "
                    f"Max={val_result['stats']['max']:.3f}, "
                    f"Mean={val_result['stats']['mean']:.3f}, "
                    f"Std={val_result['stats']['std']:.3f}")
        if val_result['warnings']:
            for warning in val_result['warnings']:
                logging.warning(f"  ⚠ {warning}")


    # 6. 将增益应用到图像
    logging.info("步骤5: 将最终增益应用回Bayer图像...")

    # --- [风险点优化] ---
    logging.info("  - 使用 'cv2.INTER_CUBIC' (双三次插值) 生成全尺寸预览增益图...")
    full_size_gains = {ch: cv2.resize(matrix, (w, h), interpolation=cv2.INTER_CUBIC)
                       for ch, matrix in final_gain_matrices.items()}

    avg_bl = np.mean(list(config.BLACK_LEVELS.values()))
    bayer_blc_float = np.maximum(0, original_bayer_16bit.astype(np.float32) - avg_bl)

    # [修改] 调用V2.2版的 apply_gains_to_bayer，传入 hard_mask
    compensated_bayer_float = bayer_utils.apply_gains_to_bayer(
        bayer_blc_float, full_size_gains, config.BAYER_PATTERN, hard_mask
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

    # 保存高通格式表格
    visualization.save_gain_tables_qcom_format(final_gain_matrices, base_filename, config.OUTPUT_DIR)
    # 保存常规表格
    visualization.save_gain_tables(final_gain_matrices, base_filename, config.OUTPUT_DIR)

    heatmap_dir = os.path.join(config.OUTPUT_DIR, 'heatmaps')
    for ch, matrix in final_gain_matrices.items():
        visualization.plot_gain_heatmap(
            matrix, ch,
            os.path.join(heatmap_dir, f"{base_filename}_{ch}_heatmap.png")
        )

    # 8. 调用最终的综合分析绘图函数
    image_results_for_plot = {
        'original_no_wb': original_rgb_no_wb,
        'original_wb': original_rgb_wb,
        'compensated_no_wb': compensated_rgb_no_wb,
        'compensated_wb': compensated_rgb_wb
    }
    visualization.create_and_save_analysis_plots(image_results_for_plot, base_filename, config.OUTPUT_DIR)

    # 9. [新增] 计算并报告均匀性指标
    logging.info("\n步骤7: 计算图像均匀性指标...")
    original_metrics = validation.calculate_uniformity_metrics(original_rgb_wb, feathered_mask, circle_info)
    corrected_metrics = validation.calculate_uniformity_metrics(compensated_rgb_wb, feathered_mask, circle_info)

    logging.info(f"\n原始图像均匀性:")
    logging.info(f"  中心亮度: {original_metrics['center_brightness']:.4f}")
    logging.info(f"  边缘亮度: {original_metrics['edge_brightness']:.4f}")
    logging.info(f"  均匀性比: {original_metrics['uniformity_ratio']:.4f}")
    logging.info(f"  评级: {original_metrics['grade']}")

    logging.info(f"\nLSC校正后均匀性:")
    logging.info(f"  中心亮度: {corrected_metrics['center_brightness']:.4f}")
    logging.info(f"  边缘亮度: {corrected_metrics['edge_brightness']:.4f}")
    logging.info(f"  均匀性比: {corrected_metrics['uniformity_ratio']:.4f}")
    logging.info(f"  评级: {corrected_metrics['grade']}")

    improvement = corrected_metrics['uniformity_ratio'] - original_metrics['uniformity_ratio']
    logging.info(f"\n均匀性改善: {improvement:+.4f} ({improvement*100:+.2f}%)")

    logging.info("\n" + "="*50)
    logging.info("所有任务已成功完成！ (V2.2 - 鱼眼安全版)")
    logging.info(f"所有输出文件已保存在目录: {os.path.abspath(config.OUTPUT_DIR)}")
    logging.info("="*50)


if __name__ == '__main__':
    main()