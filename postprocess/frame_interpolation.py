import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os


class RIFEInterpolator:
    """RIFE帧插值器，需要预训练模型权重"""

    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = None
        if model_path is not None and os.path.exists(model_path):
            try:
                # 尝试加载RIFE模型
                import sys
                sys.path.append(os.path.dirname(model_path))
                from rife_model import Model  # 需要用户提供rife_model.py
                self.model = Model()
                self.model.load_state_dict(
                    torch.load(model_path, map_location=device))
                self.model.to(device)
                self.model.eval()
                print(f"RIFE model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load RIFE model: {e}")
                self.model = None
        else:
            print(
                "RIFE model path not provided or not found, using linear interpolation fallback.")

    def interpolate(self, frame1, frame2, timestep=0.5):
        """
        插值两帧之间的中间帧
        frame1, frame2: numpy arrays (H,W,3) uint8
        timestep: 0~1 返回的帧在时间上的位置
        """
        if self.model is None:
            # 降级为线性混合
            return cv2.addWeighted(frame1, 1-timestep, frame2, timestep, 0)

        # 转换为tensor (1,3,H,W) 归一化到[-1,1]
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(
            0).float().to(self.device) / 127.5 - 1.0
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(
            0).float().to(self.device) / 127.5 - 1.0

        # RIFE forward
        with torch.no_grad():
            out = self.model(img1, img2, timestep=timestep)
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out + 1.0) * 127.5
        return np.clip(out, 0, 255).astype(np.uint8)

    def interpolate_frames(self, frames, target_fps, original_fps):
        """插值整个视频序列"""
        if target_fps <= original_fps:
            return frames
        factor = target_fps / original_fps
        new_frames = []
        for i in range(len(frames)-1):
            new_frames.append(frames[i])
            for j in range(1, int(factor)):
                t = j / factor
                interp = self.interpolate(frames[i], frames[i+1], t)
                new_frames.append(interp)
        new_frames.append(frames[-1])
        return new_frames

# ============ 全局函数，供 postprocess/__init__.py 导入 ============


def interpolate_frames(frames, target_fps, original_fps):
    """使用 RIFE 插值器或线性混合对帧序列进行插值"""
    interpolator = RIFEInterpolator()  # 默认无模型，使用线性混合
    return interpolator.interpolate_frames(frames, target_fps, original_fps)
