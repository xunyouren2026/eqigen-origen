import torch
from torch.utils.data import DataLoader, Dataset
from .webvid_dataset import WebVidDataset
from .synthetic_dataset import SyntheticVideoDataset
from .real_dataset import RealVideoDataset
from .ucf101_dataset import UCF101Dataset


def get_dataset(config, split='train'):
    dataset_type = config.dataset_type
    if dataset_type == 'synthetic':
        return SyntheticVideoDataset(
            num_samples=config.batch_size * 100,
            num_frames=config.num_frames,
            image_size=config.image_size,
            text_templates=["a video"],
            augmentation=config.augmentation
        )
    elif dataset_type == 'real':
        return RealVideoDataset(
            data_root=config.data_root,
            num_frames=config.num_frames,
            image_size=config.image_size,
            frame_interval=config.frame_interval,
            augmentation=config.augmentation,
            split=split,
            val_split=config.val_split
        )
    elif dataset_type == 'ucf101':
        return UCF101Dataset(
            data_root=config.data_root,
            split_file=config.split_file,
            num_frames=config.num_frames,
            image_size=config.image_size,
            frame_interval=config.frame_interval,
            augmentation=config.augmentation,
            split=split
        )
    elif dataset_type == 'webvid':
        return WebVidDataset(
            metadata_path=config.webvid_metadata_path,
            video_root=config.webvid_video_root,
            num_frames=config.num_frames,
            image_size=config.image_size,
            frame_interval=config.frame_interval,
            augmentation=config.augmentation,
            split=split,
            max_reference_images=config.model.max_reference_images
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataloader(config, split='train'):
    dataset = get_dataset(config, split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
