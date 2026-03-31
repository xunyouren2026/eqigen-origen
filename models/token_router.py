import torch
import torch.nn as nn


class TokenRouter(nn.Module):
    """
    动态路由模块：根据当前 token 特征，输出对各历史块的权重。
    用于 MoGA 风格的稀疏注意力。
    """
    def __init__(self, dim: int, num_blocks: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.num_blocks = num_blocks
        self.router = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_blocks)
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        query: (B, N, D)
        returns: (B, N, num_blocks) 权重，已 softmax
        """
        return self.router(query).softmax(dim=-1)