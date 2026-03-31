import numpy as np
import cv2
import random
import numbers
from typing import Union, Tuple, List, Callable


class ToTensor:
    def __call__(self, video):
        if isinstance(video, np.ndarray):
            if video.ndim == 4 and video.shape[3] in (1, 3):
                video = video.transpose(3, 0, 1, 2)
            elif video.ndim == 3:
                video = video[np.newaxis, ...]
            video = video.copy()
            import torch
            return torch.from_numpy(video).float()
        else:
            raise TypeError("Input must be numpy array")


class Normalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, video):
        if video.ndim == 4 and video.shape[3] in (1, 3):
            video = (video - self.mean) / self.std
        elif video.ndim == 4 and video.shape[0] in (1, 3):
            mean = self.mean.reshape(-1, 1, 1, 1)
            std = self.std.reshape(-1, 1, 1, 1)
            video = (video - mean) / std
        else:
            raise ValueError("Unsupported video shape")
        return video


class Resize:
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, video):
        if video.ndim == 4 and video.shape[3] in (1, 3):
            T, H, W, C = video.shape
            new_h, new_w = self.size
            resized = np.zeros((T, new_h, new_w, C), dtype=video.dtype)
            for t in range(T):
                resized[t] = cv2.resize(
                    video[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized
        elif video.ndim == 4 and video.shape[0] in (1, 3):
            C, T, H, W = video.shape
            new_h, new_w = self.size
            resized = np.zeros((C, T, new_h, new_w), dtype=video.dtype)
            for t in range(T):
                frame = video[:, t, :, :].transpose(1, 2, 0)
                frame = cv2.resize(frame, (new_w, new_h),
                                   interpolation=cv2.INTER_LINEAR)
                resized[:, t, :, :] = frame.transpose(2, 0, 1)
            return resized
        else:
            raise ValueError("Unsupported video shape")


class RandomCrop:
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, video):
        if video.ndim == 4 and video.shape[3] in (1, 3):
            T, H, W, C = video.shape
        elif video.ndim == 4 and video.shape[0] in (1, 3):
            C, T, H, W = video.shape
            video = video.transpose(1, 2, 3, 0)
            T, H, W, C = video.shape
        else:
            raise ValueError("Unsupported video shape")
        new_h, new_w = self.size
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        cropped = video[:, top:top+new_h, left:left+new_w, :]
        return cropped


class CenterCrop:
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, video):
        if video.ndim == 4 and video.shape[3] in (1, 3):
            T, H, W, C = video.shape
        elif video.ndim == 4 and video.shape[0] in (1, 3):
            C, T, H, W = video.shape
            video = video.transpose(1, 2, 3, 0)
            T, H, W, C = video.shape
        else:
            raise ValueError("Unsupported video shape")
        new_h, new_w = self.size
        top = (H - new_h) // 2
        left = (W - new_w) // 2
        cropped = video[:, top:top+new_h, left:left+new_w, :]
        return cropped


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if random.random() < self.p:
            if video.ndim == 4 and video.shape[3] in (1, 3):
                video = video[:, :, ::-1, :]
            elif video.ndim == 4 and video.shape[0] in (1, 3):
                video = video[:, :, :, ::-1]
        return video


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)
        return video
