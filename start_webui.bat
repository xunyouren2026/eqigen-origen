@echo off
chcp 65001 >nul
title AI 视频生成系统 - 启动器

echo ================================================
echo          AI 视频生成系统 启动器
echo ================================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查虚拟环境
if not exist "venv" (
    echo [错误] 未找到虚拟环境 "venv"，请先运行 setup.bat 创建环境。
    pause
    exit /b 1
)

:: 激活虚拟环境
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败！
    pause
    exit /b 1
)

:: 检查 Python（使用虚拟环境中的 Python）
venv\Scripts\python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 虚拟环境中的 Python 不可用！
    pause
    exit /b 1
)

:: 检查 main.py
if not exist "main.py" (
    echo [错误] 未找到 main.py！
    pause
    exit /b 1
)

:menu
cls
echo ================================================
echo 请选择要启动的服务：
echo.
echo   [1] API 服务 (FastAPI)   → http://127.0.0.1:8000
echo   [2] WebUI 界面 (Gradio)  → http://127.0.0.1:7860
echo   [3] 退出
echo.
set /p choice=请输入数字 (1-3)：

if "%choice%"=="1" goto start_api
if "%choice%"=="2" goto start_webui
if "%choice%"=="3" goto end

echo [错误] 无效输入，请输入 1、2 或 3。
pause
goto menu

:start_api
echo.
echo ================================================
echo 正在启动 API 服务...
echo 访问地址: http://127.0.0.1:8000
echo API 文档: http://127.0.0.1:8000/docs
echo 按 Ctrl+C 可停止服务。
echo ================================================
venv\Scripts\python main.py --mode api
pause
goto menu

:start_webui
echo.
echo ================================================
echo 正在启动 WebUI...
echo 访问地址: http://127.0.0.1:7860
echo 按 Ctrl+C 可停止服务。
echo ================================================
venv\Scripts\python main.py --mode webui
pause
goto menu

:end
echo 感谢使用！再见！
pause
exit /b 0