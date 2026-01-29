# lsc_calibration_project/lsc/visualization.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

def set_matplotlib_english_font():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

set_matplotlib_english_font()

# get_manual_circle_mask 函数没有变化，为简洁起见此处省略，请保留你文件中的原样

def get_manual_circle_mask(image_rgb_float, feather_pixels, output_dir, adjust_step):
    h, w, _ = image_rgb_float.shape
    display_image = (image_rgb_float * 255).astype(np.uint8)
    max_display_dim = 900
    scale = min(max_display_dim / w, max_display_dim / h)
    display_w, display_h = int(w * scale), int(h * scale)
    display_image_resized = cv2.resize(display_image, (display_w, display_h))
    display_image_bgr = cv2.cvtColor(display_image_resized, cv2.COLOR_RGB2BGR)

    params_path = os.path.join(output_dir, 'circle_params.npy')
    current_circle = {'center': None, 'radius': 0}
    drawing = False
    window_name = "Select Fisheye Region (check console for instructions)"

    def draw_circle_on_image(img, circle_info, color=(0, 255, 0), thickness=2):
        temp_img = img.copy()
        if circle_info['center'] and circle_info['radius'] > 0:
            center_x_orig = int(circle_info['center'][0] / scale)
            center_y_orig = int(circle_info['center'][1] / scale)
            radius_orig = int(circle_info['radius'] / scale)
            cv2.circle(temp_img, circle_info['center'], circle_info['radius'], color, thickness)
            cv2.putText(temp_img, f"Center: ({center_x_orig},{center_y_orig})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(temp_img, f"Radius: {radius_orig}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return temp_img

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, current_circle, display_image_bgr
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_circle['center'] = (x, y)
            current_circle['radius'] = 0
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            dx, dy = x - current_circle['center'][0], y - current_circle['center'][1]
            current_circle['radius'] = int(np.sqrt(dx*dx + dy*dy))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    start_main_loop = True
    if os.path.exists(params_path):
        try:
            loaded_params = np.load(params_path)
            loaded_center_orig = (int(loaded_params[0]), int(loaded_params[1]))
            loaded_radius_orig = int(loaded_params[2])
            preview_circle = {
                'center': (int(loaded_center_orig[0] * scale), int(loaded_center_orig[1] * scale)),
                'radius': int(loaded_radius_orig * scale)
            }
            preview_img = draw_circle_on_image(display_image_bgr, preview_circle, color=(0, 255, 255), thickness=2)
            cv2.putText(preview_img, "Previous selection found:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(preview_img, "Press [R] to Reuse", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(preview_img, "Press [E] to Edit", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(preview_img, "Press [N] for New selection", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow(window_name, preview_img)

            # 等待用户选择
            logging.info("检测到上次的圆心参数，等待用户选择: [R]复用 / [E]编辑 / [N]新建")
            while True:
                key = cv2.waitKey(50) & 0xFF  # 50ms刷新，保持窗口响应
                if key == ord('r') or key == ord('R'):
                    logging.info("用户选择: 复用之前的圆心参数")
                    current_circle = preview_circle
                    start_main_loop = False
                    cv2.destroyWindow(window_name)
                    break
                elif key == ord('e') or key == ord('E'):
                    logging.info("用户选择: 编辑之前的圆心参数")
                    current_circle = preview_circle
                    start_main_loop = True
                    break
                elif key == ord('n') or key == ord('N'):
                    logging.info("用户选择: 创建新的圆心选择")
                    current_circle = {'center': None, 'radius': 0}
                    start_main_loop = True
                    break
                # 持续显示预览图像，保持窗口响应
                cv2.imshow(window_name, preview_img)
        except Exception as e:
            logging.warning(f"加载圆形参数失败: {e}. 将创建新的选择。")

    if start_main_loop:
        logging.info("请手动选择圆形有效区域: 左键拖动画圆, 'wasd/zx'微调, 'r'重置, 'q'确认 (或直接关闭窗口)")
        # 重新显示窗口并刷新
        cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        # 多次刷新确保窗口激活
        for _ in range(5):
            cv2.waitKey(1)
        while True:
            key = cv2.waitKey(1) & 0xFF

            # 检测窗口关闭事件
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logging.info("检测到窗口关闭，使用当前圆形参数")
                break

            if key == ord('q'): break
            elif key == ord('w'): current_circle['radius'] += adjust_step
            elif key == ord('s'): current_circle['radius'] = max(0, current_circle['radius'] - adjust_step)
            elif key == ord('a'): current_circle['center'] = (current_circle['center'][0] - adjust_step, current_circle['center'][1])
            elif key == ord('d'): current_circle['center'] = (current_circle['center'][0] + adjust_step, current_circle['center'][1])
            elif key == ord('z'): current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] - adjust_step)
            elif key == ord('x'): current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] + adjust_step)
            elif key == ord('r'): current_circle = {'center': None, 'radius': 0}
            elif key == 27:  # ESC键也可以退出
                logging.info("检测到ESC键，使用当前圆形参数")
                break

            cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))

    cv2.destroyAllWindows()

    final_cx = int(current_circle['center'][0] / scale) if current_circle.get('center') else w // 2
    final_cy = int(current_circle['center'][1] / scale) if current_circle.get('center') else h // 2
    final_r = int(current_circle['radius'] / scale) if current_circle.get('radius', 0) > 0 else min(h, w) // 2

    try:
        os.makedirs(output_dir, exist_ok=True)
        np.save(params_path, np.array([final_cx, final_cy, final_r]))
        logging.info(f"圆形选择参数已保存至: {params_path}")
    except Exception as e:
        logging.warning(f"无法保存圆形参数: {e}")

    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hard_mask, (final_cx, final_cy), final_r, 255, -1)
    kernel_size = int(2 * feather_pixels / 3) | 1
    feathered_mask = cv2.GaussianBlur(hard_mask.astype(np.float32), (kernel_size, kernel_size), 0) / 255.0

    return np.clip(feathered_mask, 0.0, 1.0), (final_cx, final_cy, final_r)

def save_gain_tables_qcom_format(gain_matrices, base_filename, output_dir):
    """
    [新增] 保存为高通 Chromatix 兼容的 13uQ10 格式。
    数值 = round(float_gain * 1024)
    范围 = [1024, 8191]
    """
    qcom_dir = os.path.join(output_dir, "qcom_tables_Q10")
    os.makedirs(qcom_dir, exist_ok=True)

    for ch_name, matrix in gain_matrices.items():
        # [cite_start] 转换逻辑 [cite: 201, 365]
        # 文档显示 Mesh Gain 是 13uQ10 格式
        q10_data = np.round(matrix * 1024.0).astype(np.int32)
        q10_data = np.clip(q10_data, 1024, 8191)  # 硬件安全钳位

        filename = os.path.join(qcom_dir, f"{base_filename}_{ch_name}_Q10.txt")

        # 保存为制表符分隔的整数矩阵，方便复制到 Excel 或 Chromatix
        header = (
            f"Qualcomm 13uQ10 Gain Table ({ch_name})\n"
            f"Rows: {matrix.shape[0]}, Cols: {matrix.shape[1]}"
        )
        np.savetxt(
            filename,
            q10_data,
            fmt="%d",
            delimiter="\t",
            header=header
        )

        logging.info(f"Q10格式增益表已保存: {filename}")


def plot_gain_heatmap(matrix, channel_name, save_path):
    """绘制并保存增益热力图"""
    plt.figure(figsize=(12, 9))

    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val < 0.01:
        max_val = min_val + 0.01

    im = plt.imshow(
        matrix,
        cmap="jet",
        origin="upper",
        vmin=min_val,
        vmax=max_val
    )

    for (r, c), val in np.ndenumerate(matrix):
        normalized_val = (val - min_val) / (max_val - min_val + 1e-6)
        text_color = "black" if 0.2 < normalized_val < 0.8 else "white"
        plt.text(
            c, r,
            f"{val:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="black",
                ec="none",
                lw=0.5,
                alpha=0.3
            )
        )

    plt.colorbar(im, label="Gain Value")
    plt.title(f"{channel_name} Channel - Final Smoothed Gain Map")
    plt.xlabel("Grid Column Index")
    plt.ylabel("Grid Row Index")
    plt.xticks(np.arange(matrix.shape[1]))
    plt.yticks(np.arange(matrix.shape[0]))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logging.info(f"已保存 {channel_name} 通道热力图至 {save_path}")

def save_final_images(images, names, output_dir):
    """
    【已修复】修正了 `cv2.COLOR_RGB_BGR` 的手误。
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_float, name in zip(images, names):
        path = os.path.join(output_dir, f"{name}.png")
        # --- Bug修复：将COLOR_RGB_BGR修正为COLOR_RGB2BGR ---
        img_uint8_bgr = cv2.cvtColor((np.clip(img_float, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_uint8_bgr)
        logging.info(f"图像已保存: {path}")

def save_gain_tables(gain_matrices, base_filename, output_dir):
    matrices_dir = os.path.join(output_dir, 'gain_tables')
    os.makedirs(matrices_dir, exist_ok=True)
    for ch_name, matrix in gain_matrices.items():
        filename = os.path.join(matrices_dir, f"{base_filename}_{ch_name}_gain_table.txt")
        header = f"LSC Gain Table for {ch_name} Channel\nRows: {matrix.shape[0]}, Cols: {matrix.shape[1]}"
        np.savetxt(filename, matrix.flatten().reshape(1, -1), fmt='%.4f', header=header)
        logging.info(f"增益表已保存: {filename}")


def create_and_save_analysis_plots(image_results, base_filename, output_dir):
    """
    【V4版 - 最终版】创建并保存综合分析图。
    【已修复】将直方图也更新为完整的四图对比。
    """
    logging.info("正在生成Matplotlib综合分析图...")
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 四张关键图像对比 (这部分已是正确的，保持不变)
    plt.figure(figsize=(12, 12))
    plt.suptitle('Image Correction Pipeline: Step-by-Step Comparison', fontsize=16)

    titles = ['[1] Original (No LSC, No WB)',
              '[2] Original + WB (Reference)',
              '[3] LSC Corrected (No WB)',
              '[4] Final Result (LSC + WB)']

    images_to_plot = [image_results['original_no_wb'],
                      image_results['original_wb'],
                      image_results['compensated_no_wb'],
                      image_results['compensated_wb']]

    for i, (img, title) in enumerate(zip(images_to_plot, titles)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_img = os.path.join(vis_dir, f'{base_filename}_1_image_comparison.png')
    plt.savefig(save_path_img, bbox_inches='tight')
    logging.info(f"图像对比分析图已保存至: {save_path_img}")


    # 2. 【已修改】四张图像的亮度直方图对比
    plt.figure(figsize=(15, 10)) # 调整画布尺寸以适应2x2布局
    plt.suptitle('Brightness Distribution Histograms', fontsize=16)
    bins = 128

    # 使用与上面完全一致的标题和图像列表
    for i, (img, title) in enumerate(zip(images_to_plot, titles)):
        plt.subplot(2, 2, i + 1) # 修改为2x2布局
        gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        valid_pixels = gray_img[gray_img > 10]
        if valid_pixels.size > 0:
            plt.hist(valid_pixels.flatten(), bins=bins, color='gray', alpha=0.8)
        plt.title(title)
        plt.xlabel('Brightness (0-255)')
        plt.ylabel('Pixel Count')
        plt.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_hist = os.path.join(vis_dir, f'{base_filename}_2_histogram_comparison.png')
    plt.savefig(save_path_hist, bbox_inches='tight')
    logging.info(f"亮度直方图已保存至: {save_path_hist}")

    logging.info("显示交互式预览窗口... (关闭窗口后程序才会结束)")
    plt.show()


def plot_grid_with_gain_brightness(rgb_image, gain_matrix, brightness_map, channel_name, save_path, grid_size=(13, 17)):
    """
    在RGB图像上绘制网格，并标注每个网格的增益值和亮度值。

    参数:
        rgb_image (np.array): RGB图像 (H, W, 3), uint8格式
        gain_matrix (np.array): 增益矩阵 (grid_rows, grid_cols)
        brightness_map (np.array): 每个网格的亮度值 (grid_rows, grid_cols)
        channel_name (str): 通道名称 ('Gr' or 'Gb')
        save_path (str): 保存路径
        grid_size (tuple): 网格尺寸 (rows, cols)
    """
    # 复制图像避免修改原图
    img_with_grid = rgb_image.copy()
    h, w = img_with_grid.shape[:2]
    grid_rows, grid_cols = grid_size

    # 确保gain_matrix和brightness_map尺寸匹配
    if gain_matrix.shape != (grid_rows, grid_cols):
        logging.warning(f"增益矩阵尺寸 {gain_matrix.shape} 与网格尺寸 {grid_size} 不匹配")
        return

    if brightness_map.shape != (grid_rows, grid_cols):
        logging.warning(f"亮度图尺寸 {brightness_map.shape} 与网格尺寸 {grid_size} 不匹配")
        return

    # 计算每个网格的尺寸
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    # 绘制网格线
    for i in range(grid_rows + 1):
        y = int(i * cell_h)
        # 确保最后一条横线正好在底边
        if i == grid_rows:
            y = h - 1
        cv2.line(img_with_grid, (0, y), (w - 1, y), (0, 255, 255), 1)  # 黄色横线

    for j in range(grid_cols + 1):
        x = int(j * cell_w)
        # 确保最后一条竖线正好在右边
        if j == grid_cols:
            x = w - 1
        cv2.line(img_with_grid, (x, 0), (x, h - 1), (0, 255, 255), 1)  # 黄色竖线

    # 在每个网格中标注增益值和亮度值
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # 增大字体
    thickness = 1

    grid_num = 1  # 网格编号从1开始

    for i in range(grid_rows):
        for j in range(grid_cols):
            # 网格中心位置
            center_x = int((j + 0.5) * cell_w)
            center_y = int((i + 0.5) * cell_h)

            # 获取增益值和亮度值
            gain = gain_matrix[i, j]
            brightness = brightness_map[i, j]

            # 文本内容
            text_num = f"#{grid_num}"
            text_gain = f"G:{gain:.3f}"
            text_brightness = f"B:{brightness:.0f}"

            grid_num += 1  # 编号递增

            # 计算文本尺寸
            (text_w0, text_h0), _ = cv2.getTextSize(text_num, font, font_scale, thickness)
            (text_w1, text_h1), _ = cv2.getTextSize(text_gain, font, font_scale, thickness)
            (text_w2, text_h2), _ = cv2.getTextSize(text_brightness, font, font_scale, thickness)

            # 文本位置（上中下排列）
            text_y_num = center_y - text_h1 - 8
            text_y_gain = center_y
            text_y_brightness = center_y + text_h1 + 8

            # 绘制半透明背景
            padding = 3
            max_width = max(text_w0, text_w1, text_w2)
            bg_x1 = center_x - max_width // 2 - padding
            bg_y1 = text_y_num - text_h0 - padding
            bg_x2 = center_x + max_width // 2 + padding
            bg_y2 = text_y_brightness + padding

            overlay = img_with_grid.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img_with_grid, 0.3, 0, img_with_grid)

            # 绘制编号（黄色）
            cv2.putText(img_with_grid, text_num,
                       (center_x - text_w0 // 2, text_y_num),
                       font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

            # 绘制增益值（绿色）
            cv2.putText(img_with_grid, text_gain,
                       (center_x - text_w1 // 2, text_y_gain),
                       font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            # 绘制亮度值（白色）
            cv2.putText(img_with_grid, text_brightness,
                       (center_x - text_w2 // 2, text_y_brightness),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 添加标题
    title = f"{channel_name} Channel - Grid with Gain & Brightness"
    cv2.putText(img_with_grid, title, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # 保存图像
    cv2.imwrite(save_path, img_with_grid)
    logging.info(f"网格标注图已保存至: {save_path}")