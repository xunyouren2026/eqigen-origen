# postprocess/postprocess.py
from .superres import upscale_video
from .frame_interpolation import interpolate_frames
from .camera import apply_camera
from .style import apply_style
import numpy as np


def temporal_smooth(frames, sigma=1.0):
    """对视频帧序列进行高斯时域滤波"""
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        print("scipy not installed, temporal smoothing disabled")
        return frames
    frames_np = np.array(frames, dtype=np.float32)
    smoothed = gaussian_filter1d(
        frames_np, sigma=sigma, axis=0, mode='nearest')
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def postprocess_video(frames, config, mode='long'):
    """
    视频后处理
    mode: 'creative' 或 'long'
    """
    if mode == 'creative':
        # 创意模式：只做基本风格化和水印，跳过耗时操作
        if config.get('camera'):
            frames = apply_camera(
                frames, config['camera']['type'], config['camera']['strength'])
        if config.get('style'):
            frames = apply_style(frames, config['style'])
        return frames
    else:
        # 长视频模式：完整后处理
        if config.get('superres', False):
            frames = upscale_video(
                frames, scale=config.get('superres_scale', 4))
        if config.get('interpolate', False):
            frames = interpolate_frames(frames, target_fps=config.get('target_fps', 30),
                                        original_fps=config.get('original_fps', 8))
        if config.get('camera'):
            frames = apply_camera(
                frames, config['camera']['type'], config['camera']['strength'])
        if config.get('style'):
            frames = apply_style(frames, config['style'])
        if config.get('temporal_smooth', False):
            frames = temporal_smooth(
                frames, sigma=config.get('temporal_smooth_sigma', 1.0))
        return frames


def add_watermark(frames, text="AI Video", position=(10, 10), font_scale=1, color=(255, 255, 255)):
    import cv2
    import numpy as np

    # 如果是 numpy 数组，转换为列表
    if isinstance(frames, np.ndarray):
        frames = list(frames)

    for i, f in enumerate(frames):
        # 确保帧是 uint8 类型且值域 0-255
        if f.dtype != np.uint8:
            if f.max() <= 1.0:
                f = (f * 255).clip(0, 255).astype(np.uint8)
            else:
                f = np.clip(f, 0, 255).astype(np.uint8)

        # 确保是 3 通道彩色图像，并且内存连续、可写
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.ndim == 3 and f.shape[2] == 1:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.ndim == 3 and f.shape[2] == 3:
            # 输入可能是 RGB，需要转为 BGR（OpenCV 使用 BGR）
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unexpected frame shape: {f.shape}")

        # 关键：创建副本，确保数组可写且内存连续
        f = np.ascontiguousarray(f).copy()

        # 绘制文本
        cv2.putText(f, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 2)

        # 转回 RGB（因为后续保存期望 RGB 格式）
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames[i] = f

    return frames
