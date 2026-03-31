@echo off
chcp 65001 >nul 2>nul
title AI视频生成系统 - 全量依赖安装脚本（路径自适应）
setlocal enabledelayedexpansion

:: 切换到脚本所在目录（关键：确保虚拟环境创建在此目录）
cd /d "%~dp0"
echo [信息] 当前工作目录: %cd%
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.9-3.11。
    pause
    exit /b 1
)
echo [√] Python 已安装

:: ==================== 创建/激活虚拟环境 ====================
if not exist "venv" (
    echo [信息] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败！
        pause
        exit /b 1
    )
) else (
    echo [信息] 使用现有虚拟环境
)
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败！
    pause
    exit /b 1
)
echo [信息] 虚拟环境已激活（路径: %VIRTUAL_ENV%）

:: 升级 pip（使用 --no-cache-dir 避免缓存问题）
echo [信息] 升级 pip...
python -m pip install --upgrade pip setuptools wheel --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo [警告] pip 升级失败，将继续尝试安装。

:: ==================== 安装 PyTorch（自动检测 CUDA） ====================
echo [信息] 安装 PyTorch...
python -c "import torch; print(torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [信息] 未检测到 CUDA，安装 CPU 版 PyTorch
    python -m pip install torch torchvision torchaudio --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
) else (
    echo [信息] 检测到 CUDA，安装 GPU 版 PyTorch (CUDA 12.1)
    python -m pip install torch torchvision torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
)
if errorlevel 1 (
    echo [错误] PyTorch 安装失败，请检查网络后重试。
    pause
    exit /b 1
)

:: ==================== 安装必需核心库 ====================
echo [信息] 安装必需核心库...
python -m pip install transformers diffusers accelerate --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo [警告] transformers/diffusers/accelerate 安装失败
python -m pip install opencv-python numpy imageio imageio-ffmpeg --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo [警告] opencv-python 等安装失败
python -m pip install tqdm pyyaml pandas scikit-image --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo [警告] tqdm 等安装失败

:: ==================== 安装可选扩展库 ====================
echo [信息] 安装可选扩展库（加速、WebUI、API 等）...
python -m pip install xformers --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple || echo [警告] xformers 安装失败
python -m pip install gradio --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple || echo [警告] gradio 安装失败
python -m pip install fastapi uvicorn --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple || echo [警告] fastapi/uvicorn 安装失败
python -m pip install huggingface_hub --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple || echo [警告] huggingface_hub 安装失败

:: ==================== 超分功能（Real-ESRGAN） ====================
echo [信息] 安装超分依赖（Real-ESRGAN）...
:: 先升级 setuptools 和 wheel，避免 basicsr 构建问题
python -m pip install --upgrade setuptools wheel --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install basicsr --no-deps --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [警告] basicsr 安装失败，超分功能可能不可用。
) else (
    python -m pip install facexlib gfpgan addict future lmdb scipy scikit-image --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install realesrgan --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 echo [警告] Real-ESRGAN 完整安装失败，超分功能可能不可用。
)

:: ==================== 物理约束（RAFT） ====================
echo [信息] 安装光流库 RAFT（用于物理约束）...
where git >nul 2>&1
if errorlevel 1 (
    echo [警告] 未找到 git，跳过 RAFT 安装。如需使用，请先安装 Git 并重试。
) else (
    if not exist "RAFT" (
        git clone https://github.com/princeton-vl/RAFT.git
        if errorlevel 1 (
            echo [警告] RAFT 克隆失败，跳过安装。
            goto after_raft
        )
    )
    pushd RAFT
    python -m pip install -e . --no-cache-dir
    popd
    if errorlevel 1 echo [警告] RAFT 安装失败，物理后处理将回退到简单模式。
)
:after_raft

:: ==================== 可选增强库 ====================
echo [信息] 安装可选增强库（RIFE、Warp、Taichi、FlashAttention）...

:: RIFE 帧插值
echo [信息] 安装 RIFE 帧插值库...
where git >nul 2>&1
if errorlevel 1 (
    echo [警告] 未找到 git，跳过 RIFE 安装。如需使用，请先安装 Git 并重试。
) else (
    if not exist "RIFE" (
        git clone https://github.com/hzwer/arXiv2020-RIFE.git RIFE
        if errorlevel 1 (
            echo [警告] RIFE 克隆失败，跳过安装。
            goto after_rife
        )
    )
    pushd RIFE
    python -m pip install -e . --no-cache-dir
    popd
    if errorlevel 1 echo [警告] RIFE 安装失败，帧插值将回退到线性混合。
)
:after_rife
:: 提示 RIFE 权重文件（需手动放置）
if exist "RIFE" (
    if not exist "models\rife" mkdir models\rife
    if not exist "models\rife\flownet.pkl" (
        echo [提示] RIFE 权重文件未找到，请手动下载放入 models\rife\flownet.pkl
    )
)

:: Warp 物理模拟
echo [信息] 安装 Warp（刚体物理模拟）...
python -m pip install warp-lang --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo [警告] Warp 安装失败（可能不支持当前系统），物理后处理将回退到简单模式。

:: Taichi 物理模拟
echo [信息] 安装 Taichi（弹性物理模拟）...
python -m pip install taichi --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo [警告] Taichi 安装失败，物理后处理将回退到简单模式。

:: FlashAttention 加速注意力
echo [信息] 安装 FlashAttention（加速注意力计算）...
python -m pip install flash-attn --no-build-isolation --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo [警告] FlashAttention 安装失败，将使用标准注意力（速度较慢）。

:: ==================== 模型文件提示 ====================
echo.
echo ================================================
echo 模型文件说明
echo ================================================
echo 您已经手动准备好了 CLIP、Wav2Vec2 和 LeanVAE 模型，无需重复下载。
echo.
echo 若需要 Wan2.1-1.3B 核心模型（约 5 GB），请手动运行：
echo   python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('./models', exist_ok=True); hf_hub_download(repo_id='Wan-AI/Wan2.1-1.3B', filename='model.safetensors', local_dir='./models')"
echo.
echo 启动方式：
echo   - WebUI: python main.py --mode webui
echo   - API:   python main.py --mode api
echo.
echo ================================================
echo 按任意键退出...
pause >nul
exit /b