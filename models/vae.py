# models/vae.py
import os
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Optional, Tuple


class LeanVAE(nn.Module):
    """LeanVAE 封装，使用官方源码直接加载 .ckpt 权重"""

    def __init__(self, model_path: str = "./models/LeanVAE", device: torch.device = None):
        super().__init__()
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model_path = model_path

        # 确保 checkpoint 存在
        ckpt_path = os.path.join(model_path, "model.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"LeanVAE checkpoint not found: {ckpt_path}")

        # 将项目根目录添加到 sys.path，以便导入 LeanVAE 包
        import sys
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        # 简化导入（现在目录结构已整理好）
        try:
            from LeanVAE.models.autoencoder import LeanVAE as _LeanVAE
        except ImportError as e:
            raise ImportError(
                f"无法导入 LeanVAE 源码: {e}\n"
                "请确保 LeanVAE 源码位于项目根目录下的 'LeanVAE' 文件夹中，并已安装所需依赖（pytorch_lightning 等）。"
            )

        # 加载模型（strict=False 避免键名不匹配）
        self.vae = _LeanVAE.load_from_checkpoint(ckpt_path, strict=False)
        self.vae.to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # 获取 latent_channels（可以从 checkpoint 的超参数中获取）
        if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'latent_channels'):
            self.latent_channels = self.vae.config.latent_channels
        else:
            # 默认 4 通道
            self.latent_channels = 4

        # LeanVAE 的压缩参数（固定值）
        self.spatial_compress = 8      # 空间压缩 8 倍
        self.temporal_compress = 4     # 时间压缩 4 倍

        print(f"[LeanVAE] 加载完成，latent_channels={self.latent_channels}, "
              f"spatial_compress={self.spatial_compress}, temporal_compress={self.temporal_compress}")

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        编码视频到潜空间
        x: (B, C, T, H, W) 值域 [-1, 1]
        返回: (z, mean, logvar)，mean 和 logvar 为 None
        """
        with torch.no_grad():
            z = self.vae.encode(x.to(self.device))
        return z, None, None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜空间解码视频
        z: (B, latent_dim, T/4+1, H/8, W/8)
        返回: (B, C, T+1, H, W) 值域 [-1, 1]
        """
        with torch.no_grad():
            decoded = self.vae.decode(z.to(self.device))
        return decoded.float()


class WFVAE(nn.Module):
    """WF-VAE 封装，接口与原有 VideoVAE 一致"""

    def __init__(self, model_path: str = "./models/WF-VAE", device: torch.device = None):
        super().__init__()
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # WF-VAE 使用 diffusers 的 AutoencoderKLWan 或自定义加载
        # 这里假设 WF-VAE 兼容 diffusers 接口
        try:
            from diffusers import AutoencoderKLWan
            self.vae = AutoencoderKLWan.from_pretrained(model_path).to(device)
        except ImportError:
            # 回退到标准 AutoencoderKL（如果 WF-VAE 兼容）
            self.vae = AutoencoderKL.from_pretrained(model_path).to(device)

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # WF-VAE 的压缩参数（8 倍时间压缩，8 倍空间压缩）
        self.latent_channels = self.vae.config.latent_channels
        self.spatial_compress = 8
        self.temporal_compress = 8

        print(f"[WF-VAE] 加载完成，latent_channels={self.latent_channels}, "
              f"spatial_compress={self.spatial_compress}, temporal_compress={self.temporal_compress}")

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码视频到潜空间"""
        with torch.no_grad():
            z = self.vae.encode(x.to(self.device)).latent_dist.sample()
        return z, None, None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜空间解码视频"""
        with torch.no_grad():
            decoded = self.vae.decode(z.to(self.device)).sample
        return decoded.float()


class VideoVAE(nn.Module):
    """
    动态 VAE 包装器，根据配置选择不同的 VAE 实现。
    支持：
        - 'image': 原始图像 VAE（无时间压缩）
        - 'lean': LeanVAE（4 倍时间压缩 + 8 倍空间压缩）
        - 'wf': WF-VAE（8 倍时间压缩 + 8 倍空间压缩）
    """

    def __init__(self, config, device: torch.device = None):
        super().__init__()
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.config = config

        vae_type = getattr(config, 'vae_type', 'image')

        if vae_type == 'lean':
            model_path = getattr(config, 'lean_vae_path', "./models/LeanVAE")
            self.vae = LeanVAE(model_path=model_path, device=device)

        elif vae_type == 'wf':
            model_path = getattr(config, 'wf_vae_path', "./models/WF-VAE")
            self.vae = WFVAE(model_path=model_path, device=device)

        else:  # 'image' 或其他，使用原始图像 VAE
            self.vae = AutoencoderKL.from_pretrained(
                "./models/vae",
                local_files_only=True,
                low_cpu_mem_usage=False
            ).to(device)
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
            self.vae.latent_channels = self.vae.config.latent_channels
            self.vae.spatial_compress = 8
            self.vae.temporal_compress = 4  # 图像 VAE 无时间压缩，接口统一

        # 统一暴露属性，供其他模块使用（如 inferencer.py 中的 self.vae.spatial_compress）
        self.latent_channels = self.vae.latent_channels
        self.spatial_compress = self.vae.spatial_compress
        self.temporal_compress = self.vae.temporal_compress

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码视频到潜空间，返回 (z, mean, logvar)"""
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜空间解码视频"""
        return self.vae.decode(z)
