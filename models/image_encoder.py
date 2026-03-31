import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", device='cuda'):
        super().__init__()
        # 如果是默认模型名，强制使用本地路径
        if model_name == "openai/clip-vit-large-patch14":
            local_path = "./models/clip"
        else:
            local_path = model_name
        # 强制只使用本地文件，避免联网
        self.model = CLIPVisionModel.from_pretrained(
            local_path, local_files_only=True).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(
            local_path, local_files_only=True)
        self.device = device

    def forward(self, images):
        """
        images: list of tensors, each shape (B, C, H, W) or single tensor (B, C, H, W)
        Returns:
            if list: (B, N, D)
            else: (B, D)
        """
        if isinstance(images, list):
            feats = []
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                out = self.model(pixel_values=img)
                feats.append(out.pooler_output)
            return torch.stack(feats, dim=1)
        else:
            return self.model(pixel_values=images).pooler_output
