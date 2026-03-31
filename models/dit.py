# dit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from typing import Optional, Tuple

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False
    print("警告: flash-attn 未安装，将使用标准注意力（速度较慢）。")


# ==================== 稀疏注意力基类 ====================
class BaseSparseAttention(nn.Module):
    """稀疏注意力抽象基类"""

    def __init__(self, dim: int, num_heads: int, causal: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # KV 缓存（用于自回归）
        self.k_cache = None
        self.v_cache = None

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                boundary_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


# ==================== 相对位置编码 ====================
class RelativePositionBias(nn.Module):
    """可学习的相对位置偏置 + 可选抖动"""

    def __init__(self, num_heads: int, max_seq_len: int = 4096,
                 use_jitter: bool = False, jitter_scale: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_jitter = use_jitter
        self.jitter_scale = jitter_scale
        self.rel_bias = nn.Parameter(
            torch.randn(1, num_heads, 2 * max_seq_len - 1))

    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=self.rel_bias.device)
        rel_dist = pos[:, None] - pos[None, :]
        idx = rel_dist + self.max_seq_len - 1
        bias = self.rel_bias[:, :, idx]
        if self.use_jitter and self.training:
            noise = torch.randn_like(bias) * self.jitter_scale
            bias = bias + noise
        return bias


# ==================== 可学习压缩器 ====================
class LearnedCompressor(nn.Module):
    """可学习的 KV 压缩器"""

    def __init__(self, dim: int, num_heads: int, compressed_tokens: int = 64,
                 num_layers: int = 2, compressor_type: str = "mlp"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.compressed_tokens = compressed_tokens
        self.compressor_type = compressor_type

        if compressor_type == "mlp":
            self.token_proj = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
            self.query = nn.Parameter(torch.randn(
                1, num_heads, compressed_tokens, dim))
        elif compressor_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers)
            self.query = None
        else:
            raise ValueError(f"Unknown compressor type: {compressor_type}")

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T, D = k.shape
        k_flat = k.reshape(B * H, T, D)
        v_flat = v.reshape(B * H, T, D)

        if self.compressor_type == "mlp":
            k_proc = self.token_proj(k_flat)
            v_proc = self.token_proj(v_flat)
            query = self.query.expand(
                B, H, -1, -1).reshape(B * H, self.compressed_tokens, D)
            attn = torch.einsum('bqd,bkd->bqk', query, k_proc)
            attn = attn.softmax(dim=-1)
            compressed_k = torch.einsum('bqk,bkd->bqd', attn, k_proc)
            compressed_v = torch.einsum('bqk,bkd->bqd', attn, v_proc)
        else:
            k_enc = self.transformer(k_flat)
            v_enc = self.transformer(v_flat)
            compressed_k = k_enc[:, :self.compressed_tokens, :]
            compressed_v = v_enc[:, :self.compressed_tokens, :]

        compressed_k = compressed_k.reshape(B, H, self.compressed_tokens, D)
        compressed_v = compressed_v.reshape(B, H, self.compressed_tokens, D)
        return compressed_k, compressed_v


# ==================== 自适应记忆压缩器（增强版：带重建头） ====================
class AdaptiveMemoryCompressor(nn.Module):
    """
    自适应记忆压缩器：用可学习查询从历史记忆中提取固定数量的压缩令牌。
    增强：添加重建头，用于计算重建损失（联合训练）。
    """

    def __init__(self, dim: int, num_heads: int, compressed_tokens: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.compressed_tokens = compressed_tokens

        self.query = nn.Parameter(torch.randn(1, compressed_tokens, dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, memory: torch.Tensor, return_recon: bool = False):
        B, L, D = memory.shape
        query = self.query.expand(B, -1, -1)
        compressed, _ = self.cross_attn(query, memory, memory)
        compressed = self.norm(compressed)

        if return_recon:
            upsampled = F.interpolate(compressed.permute(
                0, 2, 1), size=L, mode='linear').permute(0, 2, 1)
            recon = self.reconstruction_head(upsampled)
            return compressed, recon
        return compressed


# ==================== CalibAtt 稀疏注意力（新增） ====================
class CalibSparseAttention(BaseSparseAttention):
    """CalibAtt 稀疏注意力：离线预计算稀疏 mask，推理时直接应用"""

    def __init__(self, dim, num_heads, mask_path, top_k=16, causal=True):
        super().__init__(dim, num_heads, causal)
        self.top_k = top_k
        # 加载预计算的 mask (N, N) 布尔矩阵
        mask = torch.load(mask_path, map_location='cpu')
        self.register_buffer('sparse_mask', mask.bool())

    def forward(self, x, use_cache=False, boundary_mask=None):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)

        # 应用稀疏 mask (假设 mask 是 N x N)
        mask = self.sparse_mask[:N, :N].unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


# ==================== 滑动窗口注意力（三区缓存版，支持 boundary_mask） ====================
class SlidingWindowAttention(BaseSparseAttention):
    """
    滑动窗口注意力，支持三区缓存（sink/mid/recent）和多种压缩机制。
    支持 boundary_mask 用于镜头边界重置缓存。
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 512,
                 causal: bool = True, cache_window_size: Optional[int] = None,
                 use_relative_position: bool = False,
                 use_token_routing: bool = False, num_routing_blocks: int = 8,
                 routing_top_k: int = 4, use_hierarchical_compression: bool = False,
                 compress_ratios: Tuple[int, ...] = (1, 4, 16),
                 use_learned_compressor: bool = False, compressor_type: str = "mlp",
                 compressed_tokens: int = 64, compressor_layers: int = 2,
                 use_jitter: bool = False, jitter_scale: float = 1e-5,
                 config=None,
                 # PackForcing 参数
                 sink_size: int = 4, mid_max_size: int = 256, recent_size: int = 64):
        super().__init__(dim, num_heads, causal)
        self.window_size = window_size
        self.cache_window_size = cache_window_size if cache_window_size is not None else window_size

        # 三区缓存参数
        self.sink_size = sink_size
        self.mid_max_size = mid_max_size
        self.recent_size = recent_size

        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.rel_pos = RelativePositionBias(
                self.num_heads,
                use_jitter=use_jitter,
                jitter_scale=jitter_scale
            )

        self.use_token_routing = use_token_routing
        if use_token_routing:
            from models.token_router import TokenRouter
            self.num_blocks = num_routing_blocks
            self.routing_top_k = routing_top_k
            self.router = TokenRouter(dim, num_blocks=self.num_blocks)

        self.use_hierarchical_compression = use_hierarchical_compression
        if use_hierarchical_compression:
            self.compress_ratios = compress_ratios

        self.use_learned_compressor = use_learned_compressor
        if use_learned_compressor:
            self.compressor = LearnedCompressor(
                dim, num_heads,
                compressed_tokens=compressed_tokens,
                num_layers=compressor_layers,
                compressor_type=compressor_type
            )

        self.use_turboquant = getattr(
            config, 'use_turboquant', False) if config else False
        if self.use_turboquant:
            from models.polar_quant import PolarQuantCompressor
            from models.qjl import QJLCompressor
            self.polar_quant = PolarQuantCompressor(
                self.dim, self.num_heads,
                bits=getattr(config, 'polar_quant_bits', 4)
            )
            self.qjl = QJLCompressor(
                self.dim, self.num_heads,
                rank=getattr(config, 'qjl_rank', 64)
            )
            self.qjl_cache = None

        self.reset_cache()

    def reset_cache(self) -> None:
        """重置所有缓存"""
        self.sink_cache = None      # 固定首帧缓存 (B, H, sink_size, D)
        self.mid_cache = None       # 压缩后的中间缓存 (B, H, compressed_len, D)
        self.recent_cache = None    # 最近帧缓存 (B, H, recent_size, D)
        self.mid_raw_buffer = None  # 尚未压缩的中间原始 token 缓冲 (k, v)
        self.num_tokens_processed = 0
        if self.use_turboquant:
            self.qjl_cache = None

    def _compress_mid_buffer(self, k_buffer: torch.Tensor, v_buffer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """压缩中间缓冲区的 key/value"""
        B, H, T, D = k_buffer.shape
        if T == 0:
            return k_buffer, v_buffer

        if self.use_learned_compressor:
            k_comp, v_comp = self.compressor(k_buffer, v_buffer)
        elif self.use_hierarchical_compression:
            compressed_k = []
            compressed_v = []
            remaining = T
            for ratio in self.compress_ratios:
                if remaining <= 0:
                    break
                step = max(1, ratio)
                indices = torch.arange(
                    0, remaining, step, device=k_buffer.device)
                compressed_k.append(k_buffer[:, :, indices, :])
                compressed_v.append(v_buffer[:, :, indices, :])
                remaining -= len(indices)
            k_comp = torch.cat(compressed_k, dim=2)
            v_comp = torch.cat(compressed_v, dim=2)
        else:
            k_comp, v_comp = k_buffer, v_buffer
        return k_comp, v_comp

    def _window_mask(self, q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(q_len, kv_len, device=device, dtype=torch.bool)
        for i in range(q_len):
            if self.causal:
                start = max(0, kv_len - self.window_size + i)
            else:
                start = max(0, i - self.window_size + 1)
            mask[i, start:kv_len] = False
        if self.causal:
            causal_mask = torch.triu(torch.ones(
                q_len, kv_len, device=device, dtype=torch.bool), diagonal=1)
            mask |= causal_mask
        return mask

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                boundary_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)

        # ========== 三区缓存管理 ==========
        if use_cache:
            # 1. 处理 sink 缓存（前 sink_size 个 token）
            if self.sink_cache is None and self.num_tokens_processed < self.sink_size:
                if self.sink_cache is None:
                    self.sink_cache = k
                else:
                    self.sink_cache = torch.cat([self.sink_cache, k], dim=2)
                self.num_tokens_processed += 1
            else:
                # 2. 加入 recent 缓存
                if self.recent_cache is None:
                    self.recent_cache = k
                else:
                    self.recent_cache = torch.cat(
                        [self.recent_cache, k], dim=2)

                # 检查 recent 是否超过限制
                if self.recent_cache.size(2) > self.recent_size:
                    overflow_len = self.recent_cache.size(2) - self.recent_size
                    overflow_k = self.recent_cache[:, :, :overflow_len, :]
                    overflow_v = self.recent_cache[:,
                                                   :, :overflow_len, :]  # v 同理

                    if self.mid_raw_buffer is None:
                        self.mid_raw_buffer = (overflow_k, overflow_v)
                    else:
                        mid_k, mid_v = self.mid_raw_buffer
                        self.mid_raw_buffer = (torch.cat([mid_k, overflow_k], dim=2),
                                               torch.cat([mid_v, overflow_v], dim=2))

                    self.recent_cache = self.recent_cache[:,
                                                          :, -self.recent_size:, :]

                # 3. 如果 mid_raw_buffer 超过 mid_max_size，触发压缩
                if self.mid_raw_buffer is not None:
                    mid_k, mid_v = self.mid_raw_buffer
                    if mid_k.size(2) > self.mid_max_size:
                        comp_k, comp_v = self._compress_mid_buffer(
                            mid_k, mid_v)
                        if self.mid_cache is None:
                            self.mid_cache = comp_k
                        else:
                            self.mid_cache = torch.cat(
                                [self.mid_cache, comp_k], dim=2)
                        self.mid_raw_buffer = None
            self.num_tokens_processed += 1

            # 根据 boundary_mask 重置缓存（镜头边界）
            if boundary_mask is not None and boundary_mask.size(1) == N and boundary_mask[0, -1] == 1:
                # 当前 token 是镜头边界，清空 recent 和 mid 缓存
                self.recent_cache = None
                self.mid_cache = None
                # 可选：将当前帧作为新的 sink
                if self.sink_cache is not None:
                    self.sink_cache = k
                self.num_tokens_processed = 0

        # ========== 构建最终 k/v（拼接三区 + 当前 token） ==========
        k_final = None
        v_final = None
        if self.sink_cache is not None:
            k_final = self.sink_cache
            v_final = self.sink_cache
        if self.mid_cache is not None:
            if k_final is None:
                k_final = self.mid_cache
                v_final = self.mid_cache
            else:
                k_final = torch.cat([k_final, self.mid_cache], dim=2)
                v_final = torch.cat([v_final, self.mid_cache], dim=2)
        if self.recent_cache is not None:
            if k_final is None:
                k_final = self.recent_cache
                v_final = self.recent_cache
            else:
                k_final = torch.cat([k_final, self.recent_cache], dim=2)
                v_final = torch.cat([v_final, self.recent_cache], dim=2)

        # 拼接当前 token 本身（因为缓存中可能还没有包含当前 token）
        if k_final is None:
            k_final = k
            v_final = v
        else:
            k_final = torch.cat([k_final, k], dim=2)
            v_final = torch.cat([v_final, v], dim=2)

        kv_len = k_final.size(2)

        # ========== 窗口截断（非缓存模式必须截断，缓存模式已通过 recent 机制处理） ==========
        if not use_cache and kv_len > self.window_size:
            k_final = k_final[:, :, -self.window_size:, :]
            v_final = v_final[:, :, -self.window_size:, :]
            kv_len = self.window_size

        # ========== 动态路由（可选） ==========
        if use_cache and self.use_token_routing and kv_len > self.window_size:
            block_size = (kv_len + self.num_blocks - 1) // self.num_blocks
            block_feats = []
            for i in range(self.num_blocks):
                s = i * block_size
                e = min(s + block_size, kv_len)
                if s >= e:
                    break
                block_feats.append(k_final[:, :, s:e, :].mean(dim=2))
            block_feats = torch.stack(block_feats, dim=2)
            q_avg = q.mean(dim=2)
            route_weights = torch.einsum('bhd,bhkd->bhk', q_avg, block_feats)
            route_weights = route_weights.softmax(dim=-1)
            top_k = min(self.routing_top_k, self.num_blocks)
            top_weights, top_indices = route_weights.topk(top_k, dim=-1)

            selected_k, selected_v = [], []
            for b in range(B):
                for h in range(self.num_heads):
                    for idx in top_indices[b, h]:
                        s = idx * block_size
                        e = min(s + block_size, kv_len)
                        selected_k.append(k_final[b:b+1, h:h+1, s:e, :])
                        selected_v.append(v_final[b:b+1, h:h+1, s:e, :])
            k_selected = torch.cat(selected_k, dim=2)
            v_selected = torch.cat(selected_v, dim=2)
            recent_start = max(0, kv_len - self.window_size)
            k_recent = k_final[:, :, recent_start:, :]
            v_recent = v_final[:, :, recent_start:, :]
            k_final = torch.cat([k_selected, k_recent], dim=2)
            v_final = torch.cat([v_selected, v_recent], dim=2)

        # ========== 注意力计算 ==========
        if self.use_relative_position:
            attn_mask = self._window_mask(N, k_final.size(2), x.device)
            attn = (q @ k_final.transpose(-2, -1)) * self.scale
            rel_bias = self.rel_pos(k_final.size(2))
            attn = attn + rel_bias
            attn = attn.masked_fill(attn_mask, float('-inf'))

            if self.use_turboquant and self.qjl_cache is not None:
                s = self.qjl_cache[0]  # (B, H, T_comp, rank)
                q_proj = torch.einsum(
                    'bhn d,dr->bhnr', q, self.qjl.G) / math.sqrt(self.qjl.rank)
                corr = torch.einsum('bhnr,bhtr->bhnt',
                                    q_proj, s) * self.qjl.alpha
                attn = attn + corr

            attn = attn.softmax(dim=-1)
            out = (attn @ v_final).transpose(1, 2).reshape(B, N, C)
        else:
            if FLASH_ATTN_AVAILABLE:
                # FlashAttention 不支持自定义校正，需要先应用校正再调用 flash_attn_func
                attn = (q @ k_final.transpose(-2, -1)) * self.scale
                if self.causal:
                    causal_mask = torch.triu(torch.ones(
                        N, kv_len, device=x.device, dtype=torch.bool), diagonal=1)
                    attn = attn.masked_fill(causal_mask, float('-inf'))
                if self.use_turboquant and self.qjl_cache is not None:
                    s = self.qjl_cache[0]
                    q_proj = torch.einsum(
                        'bhn d,dr->bhnr', q, self.qjl.G) / math.sqrt(self.qjl.rank)
                    corr = torch.einsum('bhnr,bhtr->bhnt',
                                        q_proj, s) * self.qjl.alpha
                    attn = attn + corr
                attn = attn.softmax(dim=-1)
                out = (attn @ v_final).transpose(1, 2).reshape(B, N, C)
            else:
                attn = (q @ k_final.transpose(-2, -1)) * self.scale
                if self.causal:
                    causal_mask = torch.triu(torch.ones(
                        N, kv_len, device=x.device, dtype=torch.bool), diagonal=1)
                    attn = attn.masked_fill(causal_mask, float('-inf'))
                if self.use_turboquant and self.qjl_cache is not None:
                    s = self.qjl_cache[0]
                    q_proj = torch.einsum(
                        'bhn d,dr->bhnr', q, self.qjl.G) / math.sqrt(self.qjl.rank)
                    corr = torch.einsum('bhnr,bhtr->bhnt',
                                        q_proj, s) * self.qjl.alpha
                    attn = attn + corr
                attn = attn.softmax(dim=-1)
                out = (attn @ v_final).transpose(1, 2).reshape(B, N, C)

        return self.out_proj(out)


# ==================== 块稀疏注意力 ====================
class BlockSparseAttention(BaseSparseAttention):
    def __init__(self, dim: int, num_heads: int, block_size: int = 64, causal: bool = True):
        super().__init__(dim, num_heads, causal)
        self.block_size = block_size

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                boundary_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)

        num_blocks = (N + self.block_size - 1) // self.block_size
        attn_list = []
        for b in range(num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, N)
            q_block = q[:, :, start:end, :]
            block_indices = [b-1, b, b+1]
            block_indices = [i for i in block_indices if 0 <= i < num_blocks]
            k_blocks, v_blocks = [], []
            for idx in block_indices:
                s = idx * self.block_size
                e = min(s + self.block_size, N)
                k_blocks.append(k[:, :, s:e, :])
                v_blocks.append(v[:, :, s:e, :])
            k_block = torch.cat(k_blocks, dim=2)
            v_block = torch.cat(v_blocks, dim=2)
            attn = (q_block @ k_block.transpose(-2, -1)) * self.scale
            if self.causal:
                q_pos = torch.arange(start, end, device=x.device)
                k_pos = torch.cat([
                    torch.arange(s, e, device=x.device)
                    for s, e in [(idx*self.block_size, min((idx+1)*self.block_size, N)) for idx in block_indices]
                ])
                causal_mask = q_pos[:, None] < k_pos[None, :]
                attn = attn.masked_fill(causal_mask.unsqueeze(
                    0).unsqueeze(0), float('-inf'))
            attn = attn.softmax(dim=-1)
            out_block = (attn @ v_block).transpose(1,
                                                   2).reshape(B, end - start, C)
            attn_list.append(out_block)
        out = torch.cat(attn_list, dim=1)
        return self.out_proj(out)


# ==================== 步长稀疏注意力 ====================
class StridedAttention(BaseSparseAttention):
    def __init__(self, dim: int, num_heads: int, stride: int = 4, causal: bool = True):
        super().__init__(dim, num_heads, causal)
        self.stride = stride

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                boundary_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads,
                                   self.head_dim).permute(0, 2, 1, 3)

        indices = torch.arange(N, device=x.device)
        stride_indices = indices % self.stride == 0
        k_stride = k[:, :, stride_indices, :]
        v_stride = v[:, :, stride_indices, :]

        attn = (q @ k_stride.transpose(-2, -1)) * self.scale
        if self.causal:
            q_pos = indices.unsqueeze(1)
            k_pos = indices[stride_indices].unsqueeze(0)
            causal_mask = q_pos < k_pos
            attn = attn.masked_fill(causal_mask.unsqueeze(
                0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v_stride).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


# ==================== Performer 线性注意力 ====================
class PerformerAttention(BaseSparseAttention):
    def __init__(self, dim: int, num_heads: int, causal: bool = True,
                 feature_dim: int = 256, use_relative_position: bool = False):
        super().__init__(dim, num_heads, causal)
        self.feature_dim = feature_dim
        self.use_relative_position = use_relative_position
        self.random_features = nn.Parameter(torch.randn(
            self.head_dim, feature_dim) / math.sqrt(feature_dim))
        if use_relative_position:
            self.rel_pos = RelativePositionBias(num_heads)

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                boundary_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        q = torch.einsum('bnhd,df->bnhf', q, self.random_features)
        k = torch.einsum('bnhd,df->bnhf', k, self.random_features)

        kv = torch.einsum('bnhf,bnhd->bhfd', k, v)
        out = torch.einsum('bnhf,bhfd->bnhd', q, kv)

        out = out.reshape(B, N, C)
        out = self.out_proj(out)
        return out


# ==================== 注意力工厂函数 ====================
def get_attention(attn_type: str, dim: int, num_heads: int,
                  window_size: int = 512, block_size: int = 64, stride: int = 4,
                  causal: bool = True, cache_window_size: Optional[int] = None,
                  **kwargs):
    config = kwargs.get('config', None)
    if attn_type == 'sliding_window':
        return SlidingWindowAttention(
            dim, num_heads, window_size, causal, cache_window_size,
            use_relative_position=kwargs.get('use_relative_position', False),
            use_token_routing=kwargs.get('use_token_routing', False),
            num_routing_blocks=kwargs.get('num_routing_blocks', 8),
            routing_top_k=kwargs.get('routing_top_k', 4),
            use_hierarchical_compression=kwargs.get(
                'use_hierarchical_compression', False),
            compress_ratios=kwargs.get('compress_ratios', (1, 4, 16)),
            use_learned_compressor=kwargs.get('use_learned_compressor', False),
            compressor_type=kwargs.get('compressor_type', 'mlp'),
            compressed_tokens=kwargs.get('compressed_tokens', 64),
            compressor_layers=kwargs.get('compressor_layers', 2),
            use_jitter=kwargs.get('use_jitter', False),
            jitter_scale=kwargs.get('jitter_scale', 1e-5),
            config=config,
            sink_size=kwargs.get('sink_size', 4),
            mid_max_size=kwargs.get('mid_max_size', 256),
            recent_size=kwargs.get('recent_size', 64)
        )
    elif attn_type == 'calib':
        mask_path = kwargs.get('calib_mask_path', '')
        top_k = kwargs.get('calib_top_k', 16)
        return CalibSparseAttention(dim, num_heads, mask_path, top_k, causal)
    elif attn_type == 'block_sparse':
        return BlockSparseAttention(dim, num_heads, block_size, causal)
    elif attn_type == 'strided':
        return StridedAttention(dim, num_heads, stride, causal)
    elif attn_type == 'performer':
        return PerformerAttention(dim, num_heads, causal,
                                  feature_dim=kwargs.get(
                                      'performer_feature_dim', 256),
                                  use_relative_position=kwargs.get('use_relative_position', False))
    else:
        raise ValueError(f'Unknown attention type: {attn_type}')


# ==================== 记忆注入交叉注意力 ====================
class MemoryCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = memory.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(memory).reshape(B, M, 2, self.num_heads,
                                     self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ==================== 物理专家层 ====================
class PhysicalExpert(nn.Module):
    def __init__(self, dim: int, expert_dim: Optional[int] = None):
        super().__init__()
        expert_dim = expert_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x).softmax(dim=-1)


# ==================== DiT Block ====================
class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_type: str = 'sliding_window', window_size: int = 512,
                 block_size: int = 64, stride: int = 4,
                 use_memory: bool = False, use_phy_experts: bool = False,
                 num_phy_experts: int = 4, cache_window_size: Optional[int] = None,
                 config=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # 从 config 中提取 CalibAtt 相关参数
        calib_mask_path = getattr(config, 'calib_mask_path', '')
        calib_top_k = getattr(config, 'calib_top_k', 16)

        self.attn = get_attention(
            attn_type, dim, num_heads, window_size, block_size, stride,
            causal=True, cache_window_size=cache_window_size,
            use_relative_position=config.use_relative_position if config else False,
            use_token_routing=config.use_token_routing if config else False,
            num_routing_blocks=config.num_routing_blocks if config else 8,
            routing_top_k=config.routing_top_k if config else 4,
            use_hierarchical_compression=config.use_hierarchical_compression if config else False,
            compress_ratios=config.compress_ratios if config else (1, 4, 16),
            use_learned_compressor=config.use_learned_compressor if config else False,
            compressor_type=config.compressor_type if config else "mlp",
            compressed_tokens=config.compressed_tokens if config else 64,
            compressor_layers=config.compressor_layers if config else 2,
            performer_feature_dim=config.performer_feature_dim if hasattr(
                config, 'performer_feature_dim') else 256,
            use_jitter=config.use_rope_jitter if config else False,
            jitter_scale=config.rope_jitter_scale if config else 1e-5,
            config=config,
            sink_size=getattr(config, 'sink_size', 4),
            mid_max_size=getattr(config, 'mid_max_size', 256),
            recent_size=getattr(config, 'recent_size', 64),
            calib_mask_path=calib_mask_path,
            calib_top_k=calib_top_k
        )
        self.norm_ca = nn.LayerNorm(dim)
        self.cross_attn = MemoryCrossAttention(
            dim, num_heads) if use_memory else None
        self.cond_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.use_phy_experts = use_phy_experts
        if use_phy_experts:
            self.phy_router = Router(dim, num_phy_experts)
            self.phy_experts = nn.ModuleList(
                [PhysicalExpert(dim) for _ in range(num_phy_experts)])

        self.use_adaptive_compressor = getattr(
            config, 'use_adaptive_compressor', False)
        if self.use_adaptive_compressor:
            self.compressor = AdaptiveMemoryCompressor(
                dim, num_heads, compressed_tokens=getattr(config, 'compressed_tokens', 64))

        self.use_layerwise_history = getattr(
            config, 'use_layerwise_history', False)
        if self.use_layerwise_history:
            self.history_gate = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
            self.history_proj = nn.Linear(dim, dim)

        self.use_multi_scale = getattr(config, 'use_multi_scale_memory', False)
        if self.use_multi_scale:
            default_scales = [4, 16, 32, 64, 128, 256]
            self.scales = getattr(config, 'multi_scale_sizes', default_scales)
            max_memory = getattr(config, 'memory_size', 256)
            self.scales = sorted(
                set([s for s in self.scales if s <= max_memory]))
            if not self.scales:
                self.scales = [4, 16, 32, 64, 128, min(256, max_memory)]
                print(f"警告：多尺度列表为空，使用默认: {self.scales}")
            self.compressors = nn.ModuleList(
                [AdaptiveMemoryCompressor(dim, num_heads, k) for k in self.scales])
            self.scale_gates = nn.Parameter(
                torch.ones(len(self.scales)) / len(self.scales))
            print(f"多尺度记忆已启用，尺度数量: {len(self.scales)}, 列表: {self.scales}")

        self.use_memory_bank = getattr(config, 'use_memory_bank', False)
        if self.use_memory_bank:
            from models.memory_bank import MemoryBank
            self.memory_bank = MemoryBank(dim, max_size=getattr(
                config, 'memory_size', 512), update_alpha=0.9)

        self.use_hierarchical_memory = getattr(
            config, 'use_hierarchical_memory', False)
        if self.use_hierarchical_memory:
            from models.memory_bank import MemoryBank
            self.short_memory = MemoryBank(dim, max_size=getattr(
                config, 'short_memory_size', 128), update_alpha=0.9)
            self.medium_memory = MemoryBank(dim, max_size=getattr(
                config, 'medium_memory_size', 256), update_alpha=0.8)
            self.long_memory = MemoryBank(dim, max_size=getattr(
                config, 'long_memory_size', 512), update_alpha=0.7)

        self.use_internal_memory = getattr(
            config, 'use_internal_memory', False)
        if self.use_internal_memory:
            self.memory_fusion = None  # 稍后由外部注入

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                memory_bank: Optional['MemoryBank'] = None,
                internal_memory_state: Optional[Tuple[torch.Tensor,
                                                      torch.Tensor]] = None,
                boundary_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_hierarchical_memory and memory_bank is None:
            short_ret, _ = self.short_memory.retrieve(x, top_k=4)
            medium_ret, _ = self.medium_memory.retrieve(x, top_k=4)
            long_ret, _ = self.long_memory.retrieve(x, top_k=4)
            x = torch.cat([short_ret, medium_ret, long_ret, x], dim=1)
            self.short_memory.update(x)
            self.medium_memory.update(x)
            self.long_memory.update(x)

        if self.use_memory_bank and memory_bank is not None:
            retrieved, _ = memory_bank.retrieve(x, top_k=8)
            x = torch.cat([retrieved, x], dim=1)

        if self.use_adaptive_compressor and memory is not None:
            compressed_memory = self.compressor(memory)
            x = torch.cat([compressed_memory, x], dim=1)

        if self.use_multi_scale and memory is not None:
            multi_mem = []
            for comp in self.compressors:
                compressed = comp(memory)
                multi_mem.append(compressed)
            if hasattr(self, 'scale_gates') and self.scale_gates is not None:
                gates = torch.softmax(self.scale_gates, dim=0)
                weighted_mem = []
                for i, mem in enumerate(multi_mem):
                    weighted_mem.append(mem * gates[i])
                x = torch.cat(weighted_mem + [x], dim=1)
            else:
                x = torch.cat(multi_mem + [x], dim=1)

        if self.use_layerwise_history and memory is not None:
            gate = self.history_gate(x.mean(dim=1, keepdim=True))
            hist = self.history_proj(memory)
            k = max(1, int(gate.squeeze().item() * memory.size(1)))
            selected_hist = hist[:, :k, :]
            x = torch.cat([selected_hist, x], dim=1)

        # 自注意力（传递 boundary_mask）
        x = x + self.attn(self.norm1(x), use_cache=use_cache,
                          boundary_mask=boundary_mask)

        if self.use_internal_memory and self.memory_fusion is not None:
            internal_mem, _ = internal_memory_state if internal_memory_state is not None else (
                None, None)
            external_mem = None
            if memory_bank is not None:
                external_mem, _ = memory_bank.retrieve(x, top_k=8)
            x = self.memory_fusion(x, external_mem, internal_mem)

        if self.cross_attn is not None and memory is not None:
            x = x + self.cross_attn(self.norm_ca(x), memory)

        if context is not None:
            proj_context = self.cond_proj(context)
            if proj_context.shape[1] != x.shape[1]:
                if proj_context.shape[1] == 1:
                    proj_context = proj_context.expand(-1, x.shape[1], -1)
                else:
                    proj_context = proj_context.transpose(1, 2)
                    proj_context = F.interpolate(
                        proj_context, size=x.shape[1], mode='linear')
                    proj_context = proj_context.transpose(1, 2)
            x = x + proj_context

        if self.use_phy_experts:
            weights = self.phy_router(x)
            expert_out = sum(w.unsqueeze(-1) * expert(x)
                             for w, expert in zip(weights.unbind(-1), self.phy_experts))
            x = x + expert_out

        x = x + self.mlp(self.norm2(x))
        return x


# ==================== 主模型 SpatialTemporalUNet ====================
class SpatialTemporalUNet(nn.Module):
    print("  进入模型 forward")

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dit_context_dim
        self.depth = config.dit_depth
        self.num_heads = config.dit_num_heads
        self.in_channels = config.vae_latent_channels

        self.attn_type = getattr(config, 'attn_type', 'sliding_window')
        self.window_size = getattr(config, 'window_size', 512)
        self.block_size = getattr(config, 'block_size', 64)
        self.stride = getattr(config, 'stride', 4)
        self.use_memory = getattr(config, 'use_memory', False)
        self.use_phy_experts = getattr(config, 'use_phy_experts', False)
        self.num_phy_experts = getattr(config, 'num_phy_experts', 4)
        self.cache_window_size = getattr(
            config, 'cache_window_size', self.window_size)

        self.input_proj = nn.Linear(self.in_channels, self.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.dim))
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim)
        )
        self.cond_proj = nn.Linear(config.dit_context_dim, self.dim)

        self.blocks = nn.ModuleList([
            DiTBlock(
                self.dim, self.num_heads, mlp_ratio=config.dit_mlp_ratio,
                attn_type=self.attn_type, window_size=self.window_size,
                block_size=self.block_size, stride=self.stride,
                use_memory=self.use_memory, use_phy_experts=self.use_phy_experts,
                num_phy_experts=self.num_phy_experts,
                cache_window_size=self.cache_window_size,
                config=config
            )
            for _ in range(self.depth)
        ])

        self.output_proj = nn.Linear(self.dim, self.in_channels)

        if getattr(config, 'multi_modal_fusion', 'sum') == "gate":
            self.modality_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            self.modality_weights = None

        self.memory_size = getattr(config, 'memory_size', 256)
        self.state_memory = nn.Parameter(
            torch.randn(1, self.memory_size, self.dim))
        self.state_attn = nn.MultiheadAttention(
            self.dim, num_heads=4, batch_first=True)

        self.use_memory_bank = getattr(config, 'use_memory_bank', False)
        if self.use_memory_bank:
            from models.memory_bank import MemoryBank
            self.global_memory_bank = MemoryBank(self.dim, max_size=getattr(
                config, 'memory_size', 512), update_alpha=0.9)

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
                image_cond: Optional[torch.Tensor] = None,
                audio_cond: Optional[torch.Tensor] = None,
                video_cond: Optional[torch.Tensor] = None,
                prev_state: Optional[torch.Tensor] = None,
                return_state: bool = False,
                use_cache: bool = False,
                camera_traj: Optional[torch.Tensor] = None,
                boundary_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, C, T, H, W = x.shape
        N = T * H * W

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        x = self.input_proj(x_flat)

        if self.config.use_first_frame_anchor:
            first_frame = x[:, :H * W].mean(dim=1, keepdim=True)
            x = torch.cat([first_frame, x], dim=1)

        if self.config.use_concatenated_history and prev_state is not None:
            x = torch.cat([prev_state, x], dim=1)

        pos_embed = self.pos_embed[:, :x.size(1), :]
        x = x + pos_embed

        t_emb = self._timestep_embedding(t, self.dim)
        x = x + t_emb.unsqueeze(1)

        attn_dtype = next(self.state_attn.parameters()).dtype

        if camera_traj is not None:
            if cond is not None:
                cond = torch.cat([cond, camera_traj], dim=1)
            else:
                cond = camera_traj

        if self.config.use_unified_cond and cond is not None:
            cond_emb = self.cond_proj(cond)
            if cond_emb.shape[1] == 1:
                cond_emb = cond_emb.expand(-1, x.size(1), -1)
            elif cond_emb.shape[1] != x.size(1):
                cond_emb = F.interpolate(cond_emb.transpose(
                    1, 2), size=x.size(1), mode='linear').transpose(1, 2)
            x = x + cond_emb
            cond_emb = cond_emb.to(attn_dtype)
        else:
            if self.modality_weights is not None:
                weights = torch.softmax(self.modality_weights, dim=0)
                text_cond = self.cond_proj(cond).mean(dim=1, keepdim=True)
                cond_emb = weights[0] * text_cond
                if image_cond is not None:
                    cond_emb = cond_emb + weights[1] * image_cond
                if audio_cond is not None:
                    cond_emb = cond_emb + weights[2] * audio_cond
                if video_cond is not None:
                    cond_emb = cond_emb + weights[0] * video_cond
            else:
                cond_emb = self.cond_proj(cond).mean(dim=1, keepdim=True)
                if image_cond is not None:
                    cond_emb = cond_emb + image_cond
                if audio_cond is not None:
                    cond_emb = cond_emb + audio_cond
                if video_cond is not None:
                    cond_emb = cond_emb + video_cond
            if cond_emb.shape[1] != x.size(1):
                cond_emb = cond_emb.expand(-1, x.size(1), -1)
            x = x + cond_emb
            cond_emb = cond_emb.to(attn_dtype)

        if prev_state is not None:
            mem = prev_state
        else:
            mem = self.state_memory.expand(B, -1, -1)

        x = x.to(attn_dtype)
        mem = mem.to(attn_dtype)
        state_context, _ = self.state_attn(x, mem, mem)
        x = x + state_context

        for block in self.blocks:
            x = block(x, cond_emb, memory=mem, use_cache=use_cache,
                      memory_bank=self.global_memory_bank if self.use_memory_bank else None,
                      internal_memory_state=None,
                      boundary_mask=boundary_mask)

        x = x.to(attn_dtype)
        mem = mem.to(attn_dtype)
        new_state, _ = self.state_attn(x, mem, mem)

        if self.use_memory_bank:
            state_mean = new_state.mean(dim=1, keepdim=True)
            self.global_memory_bank.update(state_mean, importance=None)

        out = self.output_proj(x)
        if self.config.use_concatenated_history and prev_state is not None:
            hist_len = prev_state.size(1)
            out = out[:, hist_len:, :]
        if self.config.use_first_frame_anchor:
            out = out[:, 1:, :]   # 移除首帧锚定 token
        out = out.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)

        if return_state:
            return out, None, new_state
        else:
            return out, None

    def get_initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return self.state_memory.expand(batch_size, -1, -1).to(device)

    def reset_cache(self) -> None:
        for block in self.blocks:
            block.attn.reset_cache()
