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
            cv2.putText(preview_img, "(R)euse, (E)dit, or (N)ew selection?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(window_name, preview_img)

            logging.info("检测到上次的选择参数，请在弹出的窗口操作: 'r' - 复用, 'e' - 编辑, 'n' - 新建")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                current_circle = preview_circle
                start_main_loop = False
            elif key == ord('e'):
                current_circle = preview_circle
        except Exception as e:
            logging.warning(f"加载圆形参数失败: {e}. 将创建新的选择。")

    if start_main_loop:
        logging.info("请手动选择圆形有效区域: 左键拖动画圆, 'wasd/zx'微调, 'r'重置, 'q'确认")
        cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('w'): current_circle['radius'] += adjust_step
            elif key == ord('s'): current_circle['radius'] = max(0, current_circle['radius'] - adjust_step)
            elif key == ord('a'): current_circle['center'] = (current_circle['center'][0] - adjust_step, current_circle['center'][1])
            elif key == ord('d'): current_circle['center'] = (current_circle['center'][0] + adjust_step, current_circle['center'][1])
            elif key == ord('z'): current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] - adjust_step)
            elif key == ord('x'): current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] + adjust_step)
            elif key == ord('r'): current_circle = {'center': None, 'radius': 0}
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


def plot_gain_heatmap(matrix, channel_name, save_path):
    plt.figure(figsize=(12, 9))
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val < 0.01:
        max_val = min_val + 0.01

    im = plt.imshow(matrix, cmap='jet', origin='upper', vmin=min_val, vmax=max_val)

    for (r, c), val in np.ndenumerate(matrix):
        normalized_val = (val - min_val) / (max_val - min_val + 1e-6)
        text_color = 'black' if 0.2 < normalized_val < 0.8 else 'white'
        plt.text(c, r, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", lw=0.5, alpha=0.3))

    plt.colorbar(im, label='Gain Value')
    plt.title(f"{channel_name} Channel - Final Smoothed Gain Map")
    plt.xlabel('Grid Column Index')
    plt.ylabel('Grid Row Index')
    plt.xticks(np.arange(matrix.shape[1]))
    plt.yticks(np.arange(matrix.shape[0]))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
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