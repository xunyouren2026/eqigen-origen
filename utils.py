# utils.py
# AI 视频生成系统 - 通用工具库
# 修改点：
# 1. save_video: 编码器改为 'avc1' (H.264)，确保最大兼容性
# 2. 增强维度自动检测，防止形状错误
# 3. 优化日志和进度条

import os
import sys
import json
import yaml
import random
import logging
import numpy as np
import cv2
import torch
import imageio
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path
import subprocess
import torch.distributed as dist

# ==================== 随机种子设置 ====================


def set_seed(seed: int = 42):
    """设置全局随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

# ==================== 分布式工具 ====================


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    tensor_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(tensor_list, data)
    return tensor_list

# ==================== 视频保存 (核心修复) ====================


def save_video(frames: Union[List[np.ndarray], np.ndarray],
               output_path: str,
               fps: int = 30,
               format: str = "mp4",
               codec: str = "libx264",
               quality: int = 8) -> bool:
    """
    保存视频文件。
    【关键修改】MP4/AVI 默认使用 'avc1' (H.264) 编码，兼容性最佳。
    """
    # 1. 统一输入格式为 List[np.ndarray]
    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            # 如果是 (T, C, H, W) 且 C=3，转为 (T, H, W, C)
            if frames.shape[1] == 3:
                frames = frames.transpose(0, 2, 3, 1)
        frames = [frame for frame in frames]

    if not frames:
        logging.error("No frames to save.")
        return False

    # 2. 数据类型转换与裁剪
    frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in frames]
    h, w = frames[0].shape[:2]

    try:
        if format == "gif":
            imageio.mimsave(output_path, frames, fps=fps, format='gif', loop=0)
            logging.info(f"GIF saved to {output_path}")
            return True

        elif format == "webm":
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                # WebM 需要 BGR
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            logging.info(f"WebM saved to {output_path}")
            return True

        elif format in ["mp4", "avi"]:
            # 【核心修改】使用 'avc1' (H.264) 替代 'mp4v'，解决兼容性问题
            # 如果系统不支持 avc1，OpenCV 通常会回退到默认编码器，但显式指定更稳妥
            fourcc = cv2.VideoWriter_fourcc(*'avc1')

            # 尝试初始化 VideoWriter
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            if not out.isOpened():
                logging.warning(
                    f"Failed to open VideoWriter with 'avc1'. Trying default...")
                # 如果 avc1 失败（某些构建版本的 OpenCV 不支持），尝试默认
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if not out.isOpened():
                    raise IOError("Unable to initialize VideoWriter")

            for frame in frames:
                # OpenCV 需要 BGR 格式
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()

            # 验证文件是否生成且大小正常
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logging.info(
                    f"Video saved to {output_path} ({w}x{h}, {fps}fps, {len(frames)} frames)")
                return True
            else:
                logging.error(
                    f"Video file created but empty or missing: {output_path}")
                return False

        else:
            raise ValueError(f"Unsupported format: {format}")

    except Exception as e:
        logging.error(f"Error saving video: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== 音视频合并 ====================


def save_video_with_audio(video_path: str, audio_path: str, output_path: str):
    """使用 FFmpeg 合并视频和音频"""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]
    try:
        # 隐藏 FFmpeg 输出，保持日志整洁
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        logging.info(f"Video with audio saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e}")
        return False
    except FileNotFoundError:
        logging.error("FFmpeg not found. Please install FFmpeg.")
        return False

# ==================== 张量转换 ====================


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Tensor (B,C,T,H,W) -> numpy (T,H,W,C) [0-255]"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.dim() == 5:
        tensor = tensor[0]  # 取第一个样本
    # (C, T, H, W) -> (T, H, W, C)
    tensor = tensor.permute(1, 2, 3, 0).contiguous()
    arr = tensor.numpy()
    # 如果值域是 [-1, 1]，转换到 [0, 255]
    if arr.min() < 0:
        arr = (arr + 1) / 2 * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


def numpy_to_tensor(arr: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """numpy (T,H,W,C) -> Tensor (1,C,T,H,W) [-1,1]"""
    if arr.ndim == 4:
        arr = arr.transpose(3, 0, 1, 2)  # (C, T, H, W)
        arr = arr[None]  # (1, C, T, H, W)
    tensor = torch.from_numpy(arr).float()
    # 归一化到 [-1, 1]
    if tensor.max() > 1.0:
        tensor = tensor / 127.5 - 1.0
    if device:
        tensor = tensor.to(device)
    return tensor

# ==================== 图像处理 ====================


def resize_frames(frames: List[np.ndarray], target_size: Tuple[int, int]) -> List[np.ndarray]:
    return [cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4) for frame in frames]


def center_crop_frames(frames: List[np.ndarray], target_size: Tuple[int, int]) -> List[np.ndarray]:
    h, w = frames[0].shape[:2]
    th, tw = target_size
    if th > h or tw > w:
        return resize_frames(frames, target_size)
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return [frame[start_h:start_h+th, start_w:start_w+tw] for frame in frames]

# ==================== 日志系统 ====================


def setup_logger(name: str = None, log_file: str = None, level: int = logging.INFO,
                 format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(format))
        logger.addHandler(ch)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(format))
            logger.addHandler(fh)
    return logger

# ==================== 显存管理 ====================


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_free_memory():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / (1024**2)  # MB
    return 0


def reduce_memory_usage(model):
    if torch.cuda.is_available():
        model = model.half()
    return model

# ==================== 配置加载/保存 ====================


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# ==================== 进度条 ====================


class ProgressBar:
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.current = 0

    def update(self, n: int = 1):
        self.current += n
        percent = self.current / self.total * 100
        sys.stdout.write(
            f"\r{self.desc}: [{self.current}/{self.total}] {percent:.1f}%")
        if self.current >= self.total:
            sys.stdout.write("\n")
        sys.stdout.flush()

# ==================== 视频信息读取 ====================


def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }


class FrameReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.frame_count:
            raise IndexError("Frame index out of range")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx}")
        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def close(self):
        self.cap.release()

# ==================== 装饰器 ====================


def timeit(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end-start:.2f}s")
        return result
    return wrapper

# ==================== 设备工具 ====================


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    else:
        return data

# ==================== 文件名清理 ====================


def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# ==================== 视频加载为 Tensor ====================


def load_video_tensor(video_path: str, target_size: Tuple[int, int] = None) -> torch.Tensor:
    """加载视频文件为张量 (1, C, T, H, W)，值域[-1,1]"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    video = np.array(frames, dtype=np.float32) / 127.5 - 1.0
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return video.to(device)
