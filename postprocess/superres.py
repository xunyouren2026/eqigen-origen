# postprocess/superres.py
import cv2
import numpy as np
from typing import List, Optional
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch


class RealESRGAN:
    """超分模型包装器，支持多种模型"""

    def __init__(self, scale=4, model_name="RealESRGAN_x4plus", model_path=None):
        """
        model_name: 'RealESRGAN_x4plus', 'animevideo_v3', 'RealCugan'
        """
        self.scale = scale
        self.model_name = model_name
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            # 根据模型名称选择权重
            if self.model_name == "RealESRGAN_x4plus":
                default_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            elif self.model_name == "animevideo_v3":
                default_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            elif self.model_name == "RealCugan":
                # RealCugan 需要单独处理，这里简化，使用 RealESRGAN 结构并加载对应权重
                default_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"  # 示例
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            if model_path is None:
                model_path = default_path
            self.model = RealESRGANer(
                scale=self.scale,
                model_path=model_path,
                model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=self.scale),
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
        except ImportError:
            print("Real-ESRGAN not installed, using simple resize")
            self.model = None

    def __call__(self, frame):
        if self.model:
            output, _ = self.model.enhance(frame, outscale=self.scale)
            return output
        else:
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w*self.scale, h*self.scale), interpolation=cv2.INTER_CUBIC)


def upscale_video(frames: List[np.ndarray], scale=4, model_name="RealESRGAN_x4plus", num_workers=0) -> List[np.ndarray]:
    """
    对视频帧序列进行超分。
    num_workers: 并行进程数，0表示串行，>0表示多进程并行。
    """
    if num_workers <= 0:
        upscaler = RealESRGAN(scale, model_name)
        return [upscaler(f) for f in frames]
    else:
        # 使用多进程并行处理
        # 注意：每个进程会加载一份模型，显存占用会成倍增加，请根据显存调整 num_workers
        def process_frames(frames_chunk):
            upscaler = RealESRGAN(scale, model_name)
            return [upscaler(f) for f in frames_chunk]
        # 将帧分成块
        chunk_size = max(1, len(frames) // num_workers)
        chunks = [frames[i:i+chunk_size]
                  for i in range(0, len(frames), chunk_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_frames, chunk)
                       for chunk in chunks]
            results = []
            for future in as_completed(futures):
                results.extend(future.result())
        return results
