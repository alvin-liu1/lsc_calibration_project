import numpy as np
import os

# ==============================================================================
# 配置项
# ==============================================================================

# --- 输入文件 ---
# 请将您包含原始增益值的文本文件路径放在这里
INPUT_FILENAME = './gain_table_to_invert.txt'

# --- 输出文件 ---
# 计算出的新表格将保存到这个文件
OUTPUT_FILENAME = './inverted_gain_table.txt'

# --- 除数 ---
# 您希望用哪个值来做除法
NUMERATOR = 1024.0

# ==============================================================================
# 主逻辑 (通常无需修改)
# ==============================================================================

def invert_gain_table(input_path, output_path, numerator):
    """
    读取一个包含增益值的文本文件，计算其逆关系，并保存。
    新值 = 除数 / 原始增益值
    """
    print(f"--- 开始处理文件: {input_path} ---")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"[错误] 输入文件未找到: {input_path}")
        print("请确保文件存在，并且 INPUT_FILENAME 配置正确。")
        return

    try:
        # 1. 使用 NumPy 加载文本文件为一个矩阵
        original_gain_table = np.loadtxt(input_path)
        print(f"成功加载原始增益表，尺寸: {original_gain_table.shape}")

        # 2. 创建一个与输入同样大小的浮点数矩阵，用于存放结果
        inverted_table = np.zeros_like(original_gain_table, dtype=np.float32)
        
        # 3. 创建一个掩码(mask)，只选择增益值大于一个极小值的区域进行计算
        #    这可以完美避免“除以零”的错误
        epsilon = 1e-9
        valid_mask = original_gain_table > epsilon
        
        # 4. 在有效区域内执行核心计算： 新值 = 除数 / 原始增益
        inverted_table[valid_mask] = numerator / original_gain_table[valid_mask]

        # 对于无效区域(值为0或负数)，结果设为0或一个安全值
        inverted_table[~valid_mask] = 0.0
        
        print("逆运算完成。")

        # 5. [修改] 将计算出的新表格展平为一行并保存
        #    fmt='%.4f' 表示保存为4位小数的浮点数
        header_text = (f'Inverted table calculated from {os.path.basename(input_path)}\n'
                       f'Formula: New Value = {numerator} / Original Value\n'
                       f'Output format: Single line')
        
        # 将2D矩阵展平为1D数组
        flattened_table = inverted_table.flatten()
        
        # np.savetxt需要一个2D数组来写入行，所以我们将1D数组重塑为 (1, N) 的形状
        # 使用 delimiter=' ' 确保数字之间用空格分隔
        np.savetxt(output_path, flattened_table.reshape(1, -1), fmt='%.4f', header=header_text, delimiter=' ')
        
        print(f"成功将新表格（单行格式）保存至: {output_path}")

    except Exception as e:
        print(f"[错误] 处理文件时发生意外: {e}")
        print("请检查您的输入文件格式是否正确（只包含数字和空格）。")

    print("--- 处理完成 ---\n")


if __name__ == '__main__':
    # 运行转换函数
    invert_gain_table(INPUT_FILENAME, OUTPUT_FILENAME, NUMERATOR)