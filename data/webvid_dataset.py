import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import logging


class WebVidDataset(Dataset):
    def __init__(self, metadata_path, video_root, num_frames=16, image_size=256,
                 frame_interval=1, augmentation=True, split='train', max_reference_images=4):
        self.metadata = pd.read_csv(metadata_path)
        self.video_root = video_root
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_interval = frame_interval
        self.augmentation = augmentation
        self.split = split
        self.max_reference_images = max_reference_images

        self.video_list = []
        for idx, row in self.metadata.iterrows():
            video_path = os.path.join(video_root, row['videoid'] + '.mp4')
            if os.path.exists(video_path):
                self.video_list.append((video_path, row['name']))
        logging.info(
            f"Loaded {len(self.video_list)} videos for WebVid {split}")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, caption = self.video_list[idx]
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
        reference_images = []
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + i * self.frame_interval)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros(
                    (self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                if i % (self.num_frames // self.max_reference_images) == 0 and len(reference_images) < self.max_reference_images:
                    reference_images.append(frame.copy())
            frames.append(frame)
        cap.release()

        video = np.array(frames, dtype=np.uint8)
        if self.augmentation:
            if random.random() < 0.5:
                video = video[:, :, ::-1, :]
                reference_images = [img[:, ::-1, :]
                                    for img in reference_images]
            if random.random() < 0.5:
                brightness = 1 + random.uniform(-0.2, 0.2)
                contrast = 1 + random.uniform(-0.2, 0.2)
                video = video * brightness
                video = (video - video.mean(axis=(1, 2, 3), keepdims=True)) * \
                    contrast + video.mean(axis=(1, 2, 3), keepdims=True)
                video = np.clip(video, 0, 255).astype(np.uint8)
                reference_images = [
                    np.clip(img * brightness, 0, 255).astype(np.uint8) for img in reference_images]

        video = video.astype(np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        ref_tensors = []
        for img in reference_images:
            img = img.astype(np.float32) / 127.5 - 1.0
            ref_tensors.append(torch.from_numpy(img).permute(2, 0, 1))

        return {
            'video': video,
            'text': caption,
            'reference_images': ref_tensors
        }
