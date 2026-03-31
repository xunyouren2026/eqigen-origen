# AI Video Generation Project

## 项目简介

基于时空扩散状态机（TS-DSM）、物理约束、分层注意力的高质量视频生成系统。支持文本到视频、图生视频、多模态输入（图像/音频/视频参考）、镜头语言控制、4K/8K 分块生成、长视频规划等。

## 安装

```bash
pip install -r requirements.txt

一、核心必需库（必须安装）
这些库是项目运行的基础，缺少任何一个都会导致代码无法运行。

1. PyTorch 生态
bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu121
注意：CUDA 12.1 版本，如果你的 CUDA 版本不同，请到 pytorch.org 选择对应命令。

2. 核心深度学习库
bash
pip install transformers==5.3.0 diffusers==0.37.1 accelerate==1.13.0
transformers：加载 CLIP、Wav2Vec2 等预训练模型

diffusers：扩散模型调度器（DDPM、DDIM 等）

accelerate：分布式训练加速

3. 视频/图像处理
bash
pip install opencv-python==4.13.0.92 numpy==2.4.4 imageio==2.37.3 imageio-ffmpeg pillow==12.1.1
opencv-python：图像/视频读写、处理

numpy：数组运算

imageio + imageio-ffmpeg：视频保存

pillow：图像处理

4. WebUI 与 API 服务
bash
pip install gradio==6.10.0 fastapi==0.135.2 uvicorn==0.42.0
gradio：WebUI 界面

fastapi + uvicorn：API 服务

5. 工具库
bash
pip install tqdm pyyaml pandas scipy scikit-image
tqdm：进度条

pyyaml：配置文件解析

pandas：数据处理（WebVid 数据集）

scipy：科学计算

scikit-image：图像质量评估（PSNR、SSIM）

二、可选增强库（根据需求安装）
这些库提供额外功能，不安装也不影响核心生成，但会降低性能或缺失某些特性。

1. 注意力加速（强烈推荐）
bash
pip install flash-attn==2.8.3 xformers==0.0.35
flash-attn：FlashAttention 加速注意力计算

xformers：内存高效的注意力实现

2. 超分功能（Real-ESRGAN）
bash
pip install basicsr==1.4.2 realesrgan==0.3.0
pip install facexlib gfpgan addict future lmdb
用于视频超分辨率

3. 物理模拟（PhysicsCorrector）
bash
pip install taichi==1.7.4 warp-lang==1.12.0
taichi：弹性物理模拟

warp-lang：刚体物理模拟

4. 帧插值（RIFE）
bash
git clone https://github.com/hzwer/arXiv2020-RIFE.git
cd RIFE && pip install -e .
需要额外下载权重文件放入 models/rife/flownet.pkl

5. 光流计算（物理约束训练）
bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT && pip install -e .
6. 视频加载（decord）
bash
pip install decord==0.6.0
高性能视频读取，用于数据集加载

7. 日志与监控
bash
pip install wandb==0.25.1 tensorboard
wandb：在线实验跟踪（可选）

tensorboard：本地日志可视化

8. TensorRT 加速（需 NVIDIA 显卡）
bash
pip install tensorrt==10.16.0.72
需要额外安装 TensorRT 运行时，详见 NVIDIA 官网

9. 其他可选
bash
pip install einops ftfy jsonschema matplotlib mmcv moviepy soundfile
einops：张量操作简化

ftfy：文本修复

jsonschema：JSON 校验

matplotlib：可视化

mmcv：视频处理（可选）

moviepy：视频编辑（可选）

soundfile：音频读写

三、完整一键安装命令（推荐）
将以下内容保存为 install.bat，在项目目录下以管理员身份运行：

batch
@echo off
echo 正在安装 AI 视频生成系统依赖...

:: 激活虚拟环境（如果存在）
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: 升级 pip
python -m pip install --upgrade pip setuptools wheel

:: 安装 PyTorch (CUDA 12.1)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu121

:: 安装核心库
pip install transformers==5.3.0 diffusers==0.37.1 accelerate==1.13.0
pip install opencv-python==4.13.0.92 numpy==2.4.4 imageio==2.37.3 imageio-ffmpeg pillow==12.1.1
pip install gradio==6.10.0 fastapi==0.135.2 uvicorn==0.42.0
pip install tqdm pyyaml pandas scipy scikit-image

:: 安装可选加速库
pip install flash-attn==2.8.3 xformers==0.0.35

:: 安装超分库
pip install basicsr==1.4.2 realesrgan==0.3.0
pip install facexlib gfpgan addict future lmdb

:: 安装物理模拟库
pip install taichi==1.7.4 warp-lang==1.12.0

:: 安装其他工具
pip install decord==0.6.0 einops ftfy jsonschema soundfile

echo 安装完成！
pause
四、注意事项
Python 版本：建议使用 Python 3.10 或 3.11，3.12 可能有兼容性问题。

CUDA 版本：如果使用 GPU，确保 CUDA 版本 ≥ 11.8。

虚拟环境：推荐使用虚拟环境避免依赖冲突：

bash
python -m venv venv
venv\Scripts\activate
模型权重：运行前需要下载预训练模型（约 5-10GB），详见 download_model.py。

五、快速验证安装
安装完成后，运行以下命令测试能否导入核心模块：

bash
python -c "import torch; import diffusers; import transformers; import gradio; print('✅ 所有核心库导入成功')"
如果输出 ✅ 所有核心库导入成功，说明环境配置正确。
```
一、必须手动下载的模型（核心）
1. CLIP（文本+图像编码器）
路径：./models/clip/

文件列表：

text
./models/clip/
├── config.json                    # 模型配置
├── merges.txt                     # BPE 合并规则
├── model.safetensors              # 主模型权重（约 1.7 GB）
├── preprocessor_config.json       # 图像预处理配置
├── special_tokens_map.json        # 特殊 token 映射
├── tokenizer.json                 # 分词器
└── vocab.json                     # 词汇表
来源：Hugging Face openai/clip-vit-large-patch14

作用：文本编码和图像编码，用于条件注入。

下载方式：

python
from huggingface_hub import snapshot_download
snapshot_download("openai/clip-vit-large-patch14", local_dir="./models/clip")
2. LeanVAE（视频压缩模型）
路径：./models/LeanVAE/

文件列表：

text
./models/LeanVAE/
├── model.ckpt                     # 主模型权重（约 2-3 GB）
└── config.yaml                    # 可选配置文件
来源：westlake-repl/LeanVAE

作用：将视频编码到潜空间（时间压缩 4 倍，空间压缩 8 倍），大幅降低计算量。

下载方式：

python
from huggingface_hub import snapshot_download
snapshot_download("westlake-repl/LeanVAE", local_dir="./models/LeanVAE")
3. Wav2Vec2（音频编码器）
路径：./models/wav2vec2-base/

文件列表：

text
./models/wav2vec2-base/
├── config.json                    # 模型配置
├── preprocessor_config.json       # 音频预处理配置
├── pytorch_model.bin              # 主模型权重（约 360 MB）
├── special_tokens_map.json        # 特殊 token 映射
├── tokenizer_config.json          # 分词器配置
└── vocab.json                     # 词汇表
来源：Hugging Face facebook/wav2vec2-base

作用：音频编码，用于多模态输入（音频驱动视频）。

下载方式：

python
from huggingface_hub import snapshot_download
snapshot_download("facebook/wav2vec2-base", local_dir="./models/wav2vec2-base")
二、可选下载的模型（增强功能）
4. 主扩散模型（Wan2.1-1.3B）（强烈推荐）
路径：./models/model.safetensors（或任意位置）

文件大小：约 5 GB

来源：Wan-AI/Wan2.1-1.3B

作用：视频生成的核心模型（DiT 架构），缺少它会导致随机初始化，生成效果极差。

下载方式：

python
from huggingface_hub import hf_hub_download
hf_hub_download("Wan-AI/Wan2.1-1.3B", "model.safetensors", local_dir="./models")
5. RIFE 帧插值模型（可选）
路径：./models/rife/flownet.pkl

文件大小：约 20 MB

来源：hzwer/arXiv2020-RIFE

作用：提高视频帧率，生成更流畅的视频。

下载方式：

bash
wget https://github.com/hzwer/arXiv2020-RIFE/blob/main/flownet.pkl -O ./models/rife/flownet.pkl
6. Real-ESRGAN 超分模型（可选）
路径：自动下载到 ~/.cache/ 或手动指定

文件大小：约 50-100 MB

来源：xinntao/Real-ESRGAN

作用：将低分辨率视频超分到 4K/8K。

下载方式：首次使用时自动从 GitHub 下载。

三、总结：必须下载的核心模型
模型	路径	大小	用途
CLIP	./models/clip/	~1.7 GB	文本+图像编码
LeanVAE	./models/LeanVAE/	~2-3 GB	视频压缩
Wav2Vec2	./models/wav2vec2-base/	~360 MB	音频编码
主扩散模型	./models/model.safetensors	~5 GB	视频生成核心
总计约 9-10 GB，这是运行项目的最小要求。建议先下载这些，验证系统能生成视频后，再根据需要添加增强模型。

四、验证模型是否完整
运行以下 Python 代码检查模型文件是否存在：

python
import os

models = {
    "CLIP": "./models/clip/model.safetensors",
    "LeanVAE": "./models/LeanVAE/model.ckpt",
    "Wav2Vec2": "./models/wav2vec2-base/pytorch_model.bin",
    "主模型": "./models/model.safetensors"
}

for name, path in models.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024**3)
        print(f"✅ {name}: {size:.1f} GB")
    else:
        print(f"❌ {name}: 未找到")
如果主模型缺失，日志中会出现 ⚠️ 未找到任何预训练模型 警告，必须补全才能正常生成视频。
通过网盘分享的文件：eqigen-origen.zip
链接: https://pan.baidu.com/s/1c027v7pWwz0egbcGF4ntRQ 提取码: 4ikf 
实在不行就先下载这个全文件
