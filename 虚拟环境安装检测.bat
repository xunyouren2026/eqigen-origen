@echo off
chcp 65001 >nul 2>nul
title AI视频生成系统 - 完整依赖安装与诊断脚本
setlocal enabledelayedexpansion

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.9-3.11。
    echo 请从 https://www.python.org/downloads/ 下载并安装。
    echo 修复方法：1. 安装Python 3.9-3.11 2. 将Python添加到系统PATH
    echo.
    pause
    exit /b 1
)
echo [√] Python 已安装

:: 检查虚拟环境
if not exist "venv" (
    echo [信息] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败！
        echo 请确保有权限在当前目录创建文件夹。
        echo 修复方法：1. 以管理员身份运行命令提示符 2. 确保项目路径不包含中文或空格
        echo.
        pause
        exit /b 1
    )
) else (
    echo [信息] 使用现有虚拟环境
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败！
    echo.
    pause
    exit /b 1
)
echo [√] 虚拟环境已激活

echo.
echo [信息] 所有检查完成！
echo.
pause