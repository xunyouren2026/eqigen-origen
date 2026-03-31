import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticVideoDataset(Dataset):
    def __init__(self, num_samples, num_frames, image_size, text_templates, augmentation=True):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.text_templates = text_templates
        self.augmentation = augmentation

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video = np.random.randint(
            0, 255, (self.num_frames, self.image_size, self.image_size, 3), dtype=np.uint8)
        text = np.random.choice(self.text_templates)
        video = video.astype(np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        return {'video': video, 'text': text}
