# LSC校准工具打包脚本
# 使用PyInstaller打包成单个exe文件

# ========== 安装依赖 ==========
# pip install pyinstaller

# ========== 打包命令 ==========
pyinstaller --onefile --windowed ^
    --name="LSC校准工具" ^
    --icon=icon.ico ^
    --add-data "config.py;." ^
    --add-data "lsc;lsc" ^
    --hidden-import=scipy ^
    --hidden-import=scipy.spatial ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=matplotlib ^
    lsc_gui.py

# ========== 说明 ==========
# 执行后会在dist目录生成 LSC校准工具.exe
# 将以下文件复制到同一目录:
#   - config.py (配置文件)
#   - input/ (输入文件夹)
#   - output/ (输出文件夹，可选)

pause
