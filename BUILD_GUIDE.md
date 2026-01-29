# LSC校准工具 - 打包发布指南

## 快速打包（推荐）

### Windows用户

双击运行：
```
一键打包.bat
```

等待完成后，在 `dist/LSC校准工具_发布/` 目录找到可执行文件。

### Linux/Mac用户

```bash
chmod +x build.sh
./build.sh
```

## 手动打包

### 1. 安装依赖

```bash
pip install -r requirements_build.txt
```

### 2. 打包

```bash
pyinstaller lsc_tool.spec
```

或者

```bash
pyinstaller --onefile --windowed --name="LSC校准工具" lsc_gui.py
```

## 使用打包后的程序

### 开发者分发

将以下文件夹打包成zip发给用户：
```
LSC校准工具_发布/
├── LSC校准工具.exe
├── config.py
├── 使用说明.md
└── input/  (空文件夹)
```

### 最终用户使用

1. 解压缩文件夹
2. 将RAW文件放入 `input/` 文件夹
3. 双击运行 `LSC校准工具.exe`
4. 在GUI界面中配置参数
5. 点击"开始校准"
6. 查看 `output/` 文件夹中的结果

## GUI界面功能

- ✅ 可视化配置所有参数
- ✅ 浏览选择RAW文件
- ✅ 自动保存/加载配置
- ✅ 实时状态显示
- ✅ 错误提示和验证

## 配置说明

### RAW文件设置
- **RAW文件**: 选择待校准的原始图像
- **图像宽度/高度**: 图像分辨率（像素）
- **位深度**: 8/10/12/14/16 bit
- **RAW格式**:
  - `plain`: 普通RAW（每像素独立存储）
  - `mipi_raw10`: MIPI RAW10打包格式
  - `mipi_raw12`: MIPI RAW12打包格式
- **Bayer模式**: RGGB/GRBG/GBRG/BGGR

### 校准参数
- **网格行数/列数**: 增益表分辨率（推荐12×16或13×17）
- **最大增益**: 增益上限（推荐2.0-4.0）
- **应用对称化**: 消除工艺偏差导致的不对称

### 输出设置
- **输出目录**: 结果保存位置

## 常见问题

### Q: exe文件太大（>100MB）
A: 使用UPX压缩或排除不必要的模块

### Q: 运行报错"找不到模块"
A: 检查是否所有依赖都包含在spec文件中

### Q: GUI界面不显示
A: 确保使用 `--windowed` 参数打包

### Q: 打包后无法选择文件
A: 检查文件对话框权限，可能被杀毒软件拦截

## 技术细节

### 打包内容
- Python解释器（嵌入式）
- 所有依赖库（numpy, opencv, scipy, matplotlib等）
- 源代码（lsc模块）
- GUI界面（tkinter）

### 打包大小优化
- 单文件打包：~100-150MB
- 使用UPX压缩：~60-80MB
- 目录打包：~80-100MB（但需要整个文件夹）

### 兼容性
- Windows 7/8/10/11 (x64)
- 无需安装Python
- 无需额外依赖
- 独立运行

## 更新版本

修改代码后重新打包：

```bash
# 清理旧文件
rmdir /s /q build dist

# 重新打包
一键打包.bat
```

## 授权和分发

本工具遵循MIT许可证，可自由分发和修改。

分发时建议包含：
- 使用说明文档
- 示例RAW文件
- config.py配置模板
