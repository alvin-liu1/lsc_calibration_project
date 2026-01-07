import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_fisheye_uniformity(image_path, circle_info=None):
    """
    分析内切鱼眼图像的亮度(Luma)和色彩(Color)均匀性。

    参数:
        image_path: 图像路径 (建议使用 LSC+WB 后的最终结果图)
        circle_info: (cx, cy, radius) 元组。如果为 None，自动推断或设为中心最大圆。
    """
    print(f"--- 开始分析: {os.path.basename(image_path)} ---")

    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    # 转为浮点 RGB (注意 OpenCV 是 BGR)
    img_float = img.astype(np.float32) / 255.0
    h, w, _ = img_float.shape

    # 2. 确定圆心和半径
    if circle_info is None:
        cx, cy = w // 2, h // 2
        # 默认取最短边的 48% 作为半径，留一点余量
        radius = int(min(w, h) / 2 * 0.98)
        print(f"未提供圆信息，默认使用中心: ({cx}, {cy}), 半径: {radius}")
    else:
        cx, cy, radius = circle_info
        print(f"使用圆信息: ({cx}, {cy}), 半径: {radius}")

    # 3. 准备数据结构
    # 计算每个像素到圆心的距离图
    y_idx, x_idx = np.indices((h, w))
    dist_map = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)

    # 创建有效区域 Mask (只分析圆内)
    valid_mask = dist_map <= radius

    # 提取通道 (OpenCV BGR -> R, G, B)
    B = img_float[:, :, 0]
    G = img_float[:, :, 1]
    R = img_float[:, :, 2]

    # 防止除零
    epsilon = 1e-6
    G_safe = np.maximum(G, epsilon)

    # 计算色彩比值图
    Rg_ratio = R / G_safe
    Bg_ratio = B / G_safe

    # 4. 径向统计分析 (Radial Profile)
    # 将半径分为若干个 bin (例如 100 个同心环)
    num_bins = 100
    bin_edges = np.linspace(0, radius, num_bins + 1)

    radial_luma = []
    radial_rg = []
    radial_bg = []
    radial_dist = []

    print("正在计算径向分布曲线...")

    for i in range(num_bins):
        r_min = bin_edges[i]
        r_max = bin_edges[i+1]

        # 提取当前环内的像素掩码
        ring_mask = (dist_map >= r_min) & (dist_map < r_max)

        # 统计该环内的平均值
        if np.any(ring_mask):
            # 亮度以 G 通道代表，或用 luminance 公式
            # 这里简单用 G 通道，因为它占亮度权重最大
            mean_luma = np.mean(G[ring_mask])
            mean_rg = np.mean(Rg_ratio[ring_mask])
            mean_bg = np.mean(Bg_ratio[ring_mask])

            radial_dist.append((r_min + r_max) / 2.0 / radius) # 归一化半径 (0.0 ~ 1.0)
            radial_luma.append(mean_luma)
            radial_rg.append(mean_rg)
            radial_bg.append(mean_bg)

    # 5. 可视化绘图
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Fisheye Uniformity Analysis: {os.path.basename(image_path)}', fontsize=16)

    # 子图 1: 径向亮度曲线 (Luma Falloff)
    plt.subplot(2, 2, 1)
    plt.plot(radial_dist, radial_luma, 'g-', linewidth=2, label='G Channel (Luma)')
    plt.title('Luma Uniformity (Radial Profile)')
    plt.xlabel('Normalized Radius (0=Center, 1=Edge)')
    plt.ylabel('Normalized Brightness')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    # 标注中心和边缘亮度比
    if radial_luma:
        center_luma = radial_luma[0]
        edge_luma = radial_luma[-1]
        ratio = edge_luma / (center_luma + 1e-6) * 100
        plt.text(0.5, 0.5, f"Edge/Center = {ratio:.1f}%", transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # 子图 2: 径向色比曲线 (Color Shading)
    plt.subplot(2, 2, 2)
    plt.plot(radial_dist, radial_rg, 'r-', label='R/G Ratio')
    plt.plot(radial_dist, radial_bg, 'b-', label='B/G Ratio')
    # 绘制理想参考线 (中心的值)
    if radial_rg:
        plt.axhline(y=radial_rg[0], color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=radial_bg[0], color='b', linestyle='--', alpha=0.3)

    plt.title('Color Shading (Radial R/G & B/G)')
    plt.xlabel('Normalized Radius')
    plt.ylabel('Color Ratio')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 子图 3: R/G 比值热力图 (可视化偏色位置)
    plt.subplot(2, 2, 3)
    # 仅显示圆内，圆外设为 NaN 或 0
    vis_rg = Rg_ratio.copy()
    vis_rg[~valid_mask] = np.nan

    # 自动调整显示范围，排除极端值
    vmin, vmax = np.nanpercentile(vis_rg, 5), np.nanpercentile(vis_rg, 95)
    plt.imshow(vis_rg, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar(label='R/G Ratio')
    plt.title('R/G Ratio Heatmap (Red/Green Shading)')
    plt.axis('off')

    # 子图 4: B/G 比值热力图
    plt.subplot(2, 2, 4)
    vis_bg = Bg_ratio.copy()
    vis_bg[~valid_mask] = np.nan
    vmin, vmax = np.nanpercentile(vis_bg, 5), np.nanpercentile(vis_bg, 95)
    plt.imshow(vis_bg, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar(label='B/G Ratio')
    plt.title('B/G Ratio Heatmap (Blue/Green Shading)')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = image_path.replace('.png', '_analysis.png').replace('.jpg', '_analysis.png')
    plt.savefig(output_filename)
    print(f"分析图表已保存至: {output_filename}")
    plt.show()

# --- 使用示例 ---
if __name__ == '__main__':
    # 请替换为你刚刚生成的最终结果图路径
    # 比如: 'output/images/2904x2900_4_final_result_lsc_wb.png'
    IMAGE_PATH = 'output/images/2904x2900_4_final_result_lsc_wb.png'

    # 如果你知道具体的圆心半径 (从 calibration log 里找)，填在这里更准
    # 例如: CIRCLE_INFO = (1452, 1450, 1450)
    CIRCLE_INFO = None

    if os.path.exists(IMAGE_PATH):
        analyze_fisheye_uniformity(IMAGE_PATH, CIRCLE_INFO)
    else:
        print("请检查图片路径是否正确。")