import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob


class RealVideoDataset(Dataset):
    """从文件夹递归加载视频，每个视频的文件名或同级文本作为caption"""

    def __init__(self, data_root, num_frames=16, image_size=256,
                 frame_interval=1, augmentation=True, split='train', val_split=0.01):
        self.data_root = data_root
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_interval = frame_interval
        self.augmentation = augmentation
        self.split = split

        # 递归查找所有视频文件（支持常见格式）
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
        self.video_paths = []
        for ext in video_extensions:
            self.video_paths.extend(
                glob.glob(os.path.join(data_root, '**', ext), recursive=True))

        # 按split划分
        random.shuffle(self.video_paths)
        val_size = int(len(self.video_paths) * val_split)
        if split == 'train':
            self.video_paths = self.video_paths[val_size:]
        else:
            self.video_paths = self.video_paths[:val_size]

        # 尝试获取对应的caption（同名txt或默认使用文件名）
        self.captions = []
        for path in self.video_paths:
            base = os.path.splitext(path)[0]
            txt_path = base + '.txt'
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                caption = os.path.basename(base)  # 使用文件名作为caption
            self.captions.append(caption)

        print(
            f"RealVideoDataset {split}: {len(self.video_paths)} videos loaded.")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        caption = self.captions[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # 出错时随机返回另一个样本
            return self.__getitem__(random.randint(0, len(self)-1))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        needed = self.num_frames * self.frame_interval
        if total_frames < needed:
            cap.release()
            return self.__getitem__(random.randint(0, len(self)-1))

        start = random.randint(0, total_frames - needed)
        frames = []
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + i * self.frame_interval)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros(
                    (self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame)
        cap.release()

        video = np.array(frames, dtype=np.uint8)  # (T, H, W, C)

        # 数据增强
        if self.augmentation:
            if random.random() < 0.5:
                video = video[:, :, ::-1, :]  # 水平翻转
            if random.random() < 0.5:
                brightness = 1 + random.uniform(-0.2, 0.2)
                contrast = 1 + random.uniform(-0.2, 0.2)
                video = video * brightness
                video = (video - video.mean(axis=(1, 2, 3), keepdims=True)) * \
                    contrast + video.mean(axis=(1, 2, 3), keepdims=True)
                video = np.clip(video, 0, 255).astype(np.uint8)

        # 归一化到 [-1,1]
        video = video.astype(np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(
            0, 3, 1, 2)  # (T, C, H, W) -> 后续处理时batch会转置

        return {'video': video, 'text': caption}
