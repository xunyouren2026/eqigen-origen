# models/polar_quant.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class PolarQuantCompressor(nn.Module):
    """
    极坐标量化压缩器：将输入向量 (B, H, T, D) 压缩为 (B, H, T, D)（精度降低，数量不变）
    流程：随机旋转 → 极坐标分解 → 量化半径和角度 → 逆变换
    """
    def __init__(self, dim: int, num_heads: int, bits: int = 4, max_r: float = None):
        """
        dim: 向量维度（必须为偶数，以便配对）
        num_heads: 注意力头数
        bits: 总比特数，将分配给半径和角度。建议 bits=4，其中 2 bits 给半径，2 bits 给角度（可调）
        max_r: 半径的最大值，若为 None 则动态从数据中估计
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.bits = bits
        self.r_bits = bits // 2
        self.theta_bits = bits - self.r_bits

        # 随机旋转矩阵：正交矩阵，形状 (dim, dim)，用于预处理向量
        # 使用固定随机种子保证可复现
        torch.manual_seed(42)
        R = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(R)   # 正交矩阵
        self.register_buffer('rotation', Q)

        # 半径量化范围：默认最大半径 10.0，可通过统计更新
        self.max_r = max_r if max_r is not None else 10.0
        # 半径量化步长
        self.r_step = self.max_r / (2**self.r_bits - 1)

        # 角度量化：角度范围 [0, 2π)，均匀分为 2**theta_bits 个区间
        self.theta_step = 2 * math.pi / (2**self.theta_bits)

        # 确保维度为偶数
        assert dim % 2 == 0, "Dimension must be even for polar decomposition."
        self.n_pairs = dim // 2

    def _to_polar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将输入张量 x 形状 (..., D) 转换为极坐标表示。
        将每两个连续维度视为一个二维向量，返回半径 r 和角度 theta。
        输入: (..., D)
        输出: r: (..., D/2), theta: (..., D/2)
        """
        # 将最后维度分成两两一组
        x = x.reshape(*x.shape[:-1], -1, 2)   # (..., D/2, 2)
        r = torch.norm(x, dim=-1)              # (..., D/2)
        theta = torch.atan2(x[..., 1], x[..., 0])  # (..., D/2)
        # 将角度范围从 [-π, π) 映射到 [0, 2π)
        theta = theta + math.pi
        return r, theta

    def _from_polar(self, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        从极坐标还原为笛卡尔坐标。
        r: (..., D/2), theta: (..., D/2)
        返回: (..., D)
        """
        x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)  # (..., D/2, 2)
        return x.reshape(*x.shape[:-2], -1)  # (..., D)

    def _quantize_radius(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对半径进行均匀量化。
        返回: quantized_r (int indices), dequantized_r (float)
        """
        # 限制半径范围
        r_clipped = torch.clamp(r, 0, self.max_r)
        # 量化
        idx = (r_clipped / self.r_step).long()
        idx = torch.clamp(idx, 0, 2**self.r_bits - 1)
        dequant = idx.float() * self.r_step
        return idx, dequant

    def _quantize_theta(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对角度进行均匀量化。
        返回: quantized_theta (int indices), dequantized_theta (float)
        """
        # 确保 theta 在 [0, 2π)
        theta = theta % (2 * math.pi)
        idx = (theta / self.theta_step).long()
        idx = torch.clamp(idx, 0, 2**self.theta_bits - 1)
        dequant = idx.float() * self.theta_step
        return idx, dequant

    def compress(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        压缩 key 和 value。
        k, v: (B, H, T, D)
        返回压缩后的 k_comp, v_comp: (B, H, T, D)（精度降低，数量不变）
        """
        B, H, T, D = k.shape

        # 随机旋转
        k_rot = torch.einsum('bhtd,de->bhte', k, self.rotation)   # (B,H,T,D)
        v_rot = torch.einsum('bhtd,de->bhte', v, self.rotation)

        # 极坐标分解
        r_k, theta_k = self._to_polar(k_rot)   # (B,H,T,D/2)
        r_v, theta_v = self._to_polar(v_rot)

        # 量化半径和角度
        r_k_idx, r_k_q = self._quantize_radius(r_k)
        theta_k_idx, theta_k_q = self._quantize_theta(theta_k)
        r_v_idx, r_v_q = self._quantize_radius(r_v)
        theta_v_idx, theta_v_q = self._quantize_theta(theta_v)

        # 重建旋转后的 k_quant, v_quant
        k_quant_rot = self._from_polar(r_k_q, theta_k_q)
        v_quant_rot = self._from_polar(r_v_q, theta_v_q)

        # 逆旋转
        k_quant = torch.einsum('bhtd,de->bhte', k_quant_rot, self.rotation.t())
        v_quant = torch.einsum('bhtd,de->bhte', v_quant_rot, self.rotation.t())

        return k_quant, v_quant

    def decompress(self, k_comp: torch.Tensor, v_comp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解压缩：直接返回压缩后的值（已近似）
        """
        return k_comp, v_comp