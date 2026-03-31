# models/lean_vae.py
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class LeanVAE(nn.Module):
    """LeanVAE 封装，接口与原有 VideoVAE 一致"""

    def __init__(self, model_path="./models/LeanVAE", device="cuda", dtype=torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(
            model_path, torch_dtype=dtype).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # 从 config 中读取 latent_channels（LeanVAE 固定为 4）
        self.latent_channels = self.vae.config.latent_channels   # 应为 4
        self.spatial_compress = 8
        self.temporal_compress = 4   # LeanVAE 时间压缩 4 倍

    def encode(self, x):
        # x: (B, C, T, H, W) 值域[-1,1]
        with torch.no_grad():
            # LeanVAE 接受 5D 输入，内部会做 3D 卷积
            z = self.vae.encode(x.to(self.device).to(
                self.dtype)).latent_dist.sample()
        return z, None, None   # 返回格式 (z, mean, logvar)，后两者不用

    def decode(self, z):
        with torch.no_grad():
            decoded = self.vae.decode(z.to(self.device).to(self.dtype)).sample
        return decoded.float()   # 转回 FP32 以便后续操作
