# models/simple_dit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class SimpleDiTBlock(nn.Module):
    """简化版 DiT Block，仅包含自注意力、条件注入、FFN"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_type: str = 'sliding_window', window_size: int = 512,
                 use_relative_position: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        from models.dit import get_attention
        self.attn = get_attention(
            attn_type, dim, num_heads, window_size=window_size,
            causal=False, use_relative_position=use_relative_position,
            use_token_routing=False,
            use_hierarchical_compression=False,
            use_learned_compressor=False
        )
        self.cond_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        if context is not None:
            x = x + self.cond_proj(context)
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleDiT(nn.Module):
    """简化 DiT 主模型，用于创意模式"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dit_context_dim
        self.depth = config.dit_depth
        self.num_heads = config.dit_num_heads
        self.in_channels = config.vae_latent_channels

        self.attn_type = getattr(config, 'attn_type', 'sliding_window')
        self.window_size = getattr(config, 'window_size', 512)
        self.use_relative_position = getattr(config, 'use_relative_position', False)

        self.input_proj = nn.Linear(self.in_channels, self.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.dim))
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim)
        )
        self.cond_proj = nn.Linear(config.dit_context_dim, self.dim)

        self.blocks = nn.ModuleList([
            SimpleDiTBlock(
                self.dim, self.num_heads, mlp_ratio=config.dit_mlp_ratio,
                attn_type=self.attn_type, window_size=self.window_size,
                use_relative_position=self.use_relative_position
            )
            for _ in range(self.depth)
        ])

        self.output_proj = nn.Linear(self.dim, self.in_channels)

    def _timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                prev_state=None, return_state=False, use_cache=False,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, C, T, H, W = x.shape
        N = T * H * W
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        x = self.input_proj(x_flat)

        pos_embed = self.pos_embed[:, :x.size(1), :]
        x = x + pos_embed

        t_emb = self._timestep_embedding(t, self.dim)
        x = x + t_emb.unsqueeze(1)

        cond_emb = self.cond_proj(cond.mean(dim=1, keepdim=True))
        if cond_emb.shape[1] != x.size(1):
            cond_emb = cond_emb.expand(-1, x.size(1), -1)
        x = x + cond_emb

        for block in self.blocks:
            x = block(x, cond_emb, use_cache=use_cache)

        out = self.output_proj(x)
        out = out.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)

        return out, None, None

    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()