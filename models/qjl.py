# models/qjl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QJLCompressor(nn.Module):
    """
    QJL 1‑bit 误差估计器：对残差进行 JL 降维和符号量化，用于注意力内积校正。
    """
    def __init__(self, dim: int, num_heads: int, rank: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.rank = rank

        # 随机投影矩阵 G: (dim, rank)，使用固定随机种子
        torch.manual_seed(123)
        self.register_buffer('G', torch.randn(dim, rank) / math.sqrt(rank))

        # 校正系数 alpha = 1/sqrt(rank)
        self.alpha = 1.0 / math.sqrt(rank)

    def compute_sign(self, k_orig: torch.Tensor, k_quant: torch.Tensor) -> torch.Tensor:
        """
        计算残差的符号投影。
        k_orig: 原始 key 张量 (B, H, T, D)
        k_quant: 量化后的 key 张量 (B, H, T, D)
        返回符号向量 s: (B, H, T, rank)，值为 +1 或 -1
        """
        e = k_orig - k_quant
        # 投影到低维空间
        z = torch.einsum('bhtd,dr->bhtr', e, self.G)   # (B, H, T, rank)
        s = torch.sign(z)                               # (B, H, T, rank)
        return s

    def correct_score(self, q: torch.Tensor, k_quant: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        校正注意力分数。
        q: (B, H, N, D)
        k_quant: (B, H, T, D)
        s: (B, H, T, rank)
        返回校正后的内积: (B, H, N, T)
        """
        # 原始内积
        raw_scores = torch.einsum('bhn d,bht d->bhnt', q, k_quant)   # (B, H, N, T)

        # QJL 校正项： (q @ G) * (s * alpha) 的和
        # 首先计算 q 的投影: q_proj = q @ G  (B, H, N, rank)
        q_proj = torch.einsum('bhn d,dr->bhnr', q, self.G) / math.sqrt(self.rank)
        # 校正项 = (q_proj * s) 在 rank 维度求和，乘以 alpha
        corr = torch.einsum('bhnr,bhtr->bhnt', q_proj, s) * self.alpha
        return raw_scores + corr