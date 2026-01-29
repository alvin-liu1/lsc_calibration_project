@echo off
chcp 65001 >nul
echo ============================================
echo LSC校准工具 - 一键打包脚本
echo ============================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo [1/4] 安装打包依赖...
pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [错误] 安装PyInstaller失败
    pause
    exit /b 1
)

echo.
echo [2/4] 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "LSC校准工具.spec" del /f /q "LSC校准工具.spec"

echo.
echo [3/4] 开始打包...
pyinstaller --onefile ^
    --windowed ^
    --name="LSC校准工具" ^
    --add-data="lsc;lsc" ^
    --hidden-import=scipy.spatial ^
    --hidden-import=scipy.spatial.cKDTree ^
    --icon=NONE ^
    lsc_gui.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo.
echo [4/4] 准备分发包...
if not exist "dist\LSC校准工具_发布" mkdir "dist\LSC校准工具_发布"
copy "dist\LSC校准工具.exe" "dist\LSC校准工具_发布\"
copy config.py "dist\LSC校准工具_发布\"
copy USER_GUIDE.md "dist\LSC校准工具_发布\使用说明.md"
if not exist "dist\LSC校准工具_发布\input" mkdir "dist\LSC校准工具_发布\input"

echo.
echo ============================================
echo 打包完成！
echo ============================================
echo.
echo 可执行文件位置: dist\LSC校准工具_发布\LSC校准工具.exe
echo.
echo 分发时请将整个 "LSC校准工具_发布" 文件夹打包给用户。
echo.
pause
