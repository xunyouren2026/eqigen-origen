import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob


class UCF101Dataset(Dataset):
    """UCF101动作识别数据集，假设文件夹结构为：data_root/action_class/video.avi"""

    def __init__(self, data_root, split_file=None, num_frames=16, image_size=256,
                 frame_interval=1, augmentation=True, split='train'):
        self.data_root = data_root
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_interval = frame_interval
        self.augmentation = augmentation
        self.split = split

        # 获取所有视频路径
        self.video_paths = glob.glob(os.path.join(data_root, '*', '*.avi'))
        self.video_paths.extend(
            glob.glob(os.path.join(data_root, '*', '*.mp4')))

        # 根据split_file划分训练/验证
        if split_file is not None:
            with open(split_file, 'r') as f:
                lines = f.readlines()
            train_videos = set()
            val_videos = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_name = parts[0]
                    if int(parts[1]) == 1:
                        train_videos.add(video_name)
                    else:
                        val_videos.add(video_name)
            # 过滤路径
            filtered = []
            for path in self.video_paths:
                rel = os.path.relpath(path, data_root)
                if split == 'train' and rel in train_videos:
                    filtered.append(path)
                elif split == 'val' and rel in val_videos:
                    filtered.append(path)
            self.video_paths = filtered

        # 从路径中提取动作标签作为caption
        self.labels = [os.path.basename(os.path.dirname(p))
                       for p in self.video_paths]

        print(f"UCF101Dataset {split}: {len(self.video_paths)} videos loaded.")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        caption = self.labels[idx]  # 动作类别作为文本

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
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

        video = np.array(frames, dtype=np.uint8)

        if self.augmentation:
            if random.random() < 0.5:
                video = video[:, :, ::-1, :]  # 水平翻转

        # 归一化
        video = video.astype(np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        return {'video': video, 'text': caption}
