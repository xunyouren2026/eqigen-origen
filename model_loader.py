import os
import torch
import logging
from huggingface_hub import hf_hub_download
from models.dit import SpatialTemporalUNet
from models.unet import UNetModel
from models.simple_dit import SimpleDiT   # 新增创意模式模型


def load_pretrained_model(config, device, pretrained_name=None, model_type="dit", cache_dir=None):
    """
    加载预训练权重。
    pretrained_name: 可以是本地路径（如 './models/model.safetensors'）或 Hugging Face 仓库ID（如 'Wan-AI/Wan2.1-1.3B'）
    """
    # 无预训练权重时随机初始化
    if pretrained_name is None:
        logging.info(
            "No pretrained model specified, using random initialization.")
        # 根据模式选择模型
        if config.model.mode == 'creative':
            model = SimpleDiT(config.model)
        else:
            if model_type == "dit":
                model = SpatialTemporalUNet(config.model)
            else:
                model = UNetModel(config.model)
        model.to(device)
        return model

    # 检查是否为本地文件路径
    if os.path.isfile(pretrained_name):
        weight_path = pretrained_name
        logging.info(f"Loading model from local file: {weight_path}")
    else:
        # 否则从 Hugging Face 下载
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        try:
            weight_path = hf_hub_download(
                repo_id=pretrained_name,
                filename="model.safetensors",
                cache_dir=cache_dir,
                resume=True
            )
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise

    # 构建模型结构（根据模式选择）
    if config.model.mode == 'creative':
        model = SimpleDiT(config.model)
    else:
        if model_type == "dit":
            model = SpatialTemporalUNet(config.model)
        else:
            model = UNetModel(config.model)

    # 加载权重
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    logging.info(f"Loaded pretrained model from {weight_path}")
    return model
