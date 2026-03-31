# models/memory_fusion.py
import torch
import torch.nn as nn


class MemoryFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, use_linear: bool = False):
        super().__init__()
        self.use_linear = use_linear
        if use_linear:
            self.gate = nn.Linear(dim * 2, dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.cross_attn = nn.MultiheadAttention(
                dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, external_mem: torch.Tensor, internal_mem: torch.Tensor) -> torch.Tensor:
        if self.use_linear:
            ext_pool = external_mem.mean(
                dim=1, keepdim=True) if external_mem is not None else torch.zeros_like(x[:, :1, :])
            int_pool = internal_mem.mean(
                dim=1, keepdim=True) if internal_mem is not None else torch.zeros_like(x[:, :1, :])
            combined = torch.cat([ext_pool, int_pool], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            return x + gate * (ext_pool + int_pool)
        else:
            combined = torch.cat([external_mem, internal_mem], dim=1)
            attn_out, _ = self.cross_attn(x, combined, combined)
            return x + attn_out
