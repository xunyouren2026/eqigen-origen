# teacache.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable


class TeaCache:
    """
    TeaCache 加速器，用于扩散模型采样。
    基于论文 "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model" (CVPR 2025)
    参考实现: https://github.com/LiewFeng/TeaCache
    """

    def __init__(
        self,
        threshold: float = 0.2,
        model: torch.nn.Module = None,
        get_input_feat: Optional[Callable] = None,
        device: torch.device = None,
        use_cfg_separation: bool = True
    ):
        """
        Args:
            threshold: 相似度阈值（L1 距离），小于此值则复用缓存。推荐 0.1-0.4
            model: 模型实例（用于获取特征）
            get_input_feat: 自定义获取输入特征的函数
            device: 设备
            use_cfg_separation: 是否为 CFG 的正负分支维护独立缓存
        """
        self.threshold = threshold
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.get_input_feat = get_input_feat or self._default_get_feat
        self.use_cfg_separation = use_cfg_separation

        # 缓存状态
        self.cache = None           # 上一步的特征（无条件分支）
        self.cache_t = None         # 上一步的时间步
        self.cache_noise_pred = None  # 上一步的噪声预测结果（无条件）

        # CFG 独立缓存（条件分支）
        self.cache_cond = None
        self.cache_t_cond = None
        self.cache_noise_pred_cond = None

    def _default_get_feat(self, latents: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        默认获取模型第一层归一化后的输入特征。
        取 latents 在通道维度的均值作为特征（降维，方便比较）
        """
        feat = latents.mean(dim=1, keepdim=True).flatten(1)  # (B, T*H*W)
        return feat

    def reset(self):
        """重置缓存（开始新序列时调用）"""
        self.cache = None
        self.cache_t = None
        self.cache_noise_pred = None
        self.cache_cond = None
        self.cache_t_cond = None
        self.cache_noise_pred_cond = None

    def step(self, latents: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
             model_fn: Callable) -> Tuple[torch.Tensor, bool]:
        """
        执行一步采样（用于无条件分支或 CFG 未分离时）

        Args:
            latents: 当前潜变量 (B, C, T, H, W)
            t: 当前时间步 (B,)
            cond: 条件嵌入 (B, L, D)
            model_fn: 模型前向函数，接收 (latents, t, cond) 返回 noise_pred

        Returns:
            noise_pred: 噪声预测结果
            used_cache: 是否使用了缓存（布尔）
        """
        cur_feat = self.get_input_feat(latents, t, cond)

        use_cache = False
        if self.cache is not None and torch.abs(self.cache_t - t).item() == 1:
            # 计算 L1 距离
            dist = torch.linalg.norm(cur_feat - self.cache, ord=1).item()
            # 归一化距离（可选，提高稳定性）
            norm = torch.linalg.norm(self.cache, ord=1).item() + 1e-6
            rel_dist = dist / norm

            if rel_dist < self.threshold:
                use_cache = True
                noise_pred = self.cache_noise_pred
            else:
                self.cache = cur_feat
                self.cache_t = t
                noise_pred = model_fn(latents, t, cond)
                self.cache_noise_pred = noise_pred
        else:
            self.cache = cur_feat
            self.cache_t = t
            noise_pred = model_fn(latents, t, cond)
            self.cache_noise_pred = noise_pred

        return noise_pred, use_cache

    def step_with_cfg(self, latents: torch.Tensor, t: torch.Tensor,
                      cond: torch.Tensor, uncond: torch.Tensor,
                      model_fn: Callable) -> Tuple[torch.Tensor, bool, bool]:
        """
        执行一步采样（支持 CFG 分离缓存，用于条件和无条件分支独立缓存）

        Args:
            latents: 当前潜变量
            t: 当前时间步
            cond: 条件嵌入
            uncond: 无条件嵌入
            model_fn: 模型前向函数

        Returns:
            noise_pred: 融合后的噪声预测
            used_cache_cond: 条件分支是否使用了缓存
            used_cache_uncond: 无条件分支是否使用了缓存
        """
        # 获取当前特征（条件分支和无条件分支共享输入特征）
        cur_feat = self.get_input_feat(latents, t, cond)

        # ===== 条件分支缓存 =====
        use_cache_cond = False
        if self.cache_cond is not None and torch.abs(self.cache_t_cond - t).item() == 1:
            norm = torch.linalg.norm(self.cache_cond, ord=1).item() + 1e-6
            rel_dist = torch.linalg.norm(
                cur_feat - self.cache_cond, ord=1).item() / norm
            if rel_dist < self.threshold:
                use_cache_cond = True
                noise_pred_cond = self.cache_noise_pred_cond
            else:
                self.cache_cond = cur_feat
                self.cache_t_cond = t
                noise_pred_cond = model_fn(latents, t, cond)
                self.cache_noise_pred_cond = noise_pred_cond
        else:
            self.cache_cond = cur_feat
            self.cache_t_cond = t
            noise_pred_cond = model_fn(latents, t, cond)
            self.cache_noise_pred_cond = noise_pred_cond

        # ===== 无条件分支缓存 =====
        use_cache_uncond = False
        if self.cache is not None and torch.abs(self.cache_t - t).item() == 1:
            norm = torch.linalg.norm(self.cache, ord=1).item() + 1e-6
            rel_dist = torch.linalg.norm(
                cur_feat - self.cache, ord=1).item() / norm
            if rel_dist < self.threshold:
                use_cache_uncond = True
                noise_pred_uncond = self.cache_noise_pred
            else:
                self.cache = cur_feat
                self.cache_t = t
                noise_pred_uncond = model_fn(latents, t, uncond)
                self.cache_noise_pred = noise_pred_uncond
        else:
            self.cache = cur_feat
            self.cache_t = t
            noise_pred_uncond = model_fn(latents, t, uncond)
            self.cache_noise_pred = noise_pred_uncond

        return noise_pred_cond, noise_pred_uncond, use_cache_cond, use_cache_uncond


class TeaCacheWithCoefficients(TeaCache):
    """
    带多项式系数缩放的 TeaCache，用于更精确的相似度判断。
    参考 vLLM-Omni 实现中的系数缩放策略[citation:1][citation:6]
    """

    def __init__(
        self,
        threshold: float = 0.2,
        coefficients: list = None,
        model: torch.nn.Module = None,
        get_input_feat: Optional[Callable] = None,
        device: torch.device = None
    ):
        super().__init__(threshold, model, get_input_feat, device)
        # 多项式系数：默认使用与 Wan2.2 类似的系数[citation:2]
        self.coefficients = coefficients or [1.0, 0.0, 0.0]
        self.accumulated = 0.0

    def _rescale(self, rel_l1: float) -> float:
        """使用多项式系数缩放 L1 距离"""
        result = 0.0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (rel_l1 ** i)
        return result

    def step(self, latents: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
             model_fn: Callable) -> Tuple[torch.Tensor, bool]:
        cur_feat = self.get_input_feat(latents, t, cond)

        use_cache = False
        if self.cache is not None and torch.abs(self.cache_t - t).item() == 1:
            norm = torch.linalg.norm(self.cache, ord=1).item() + 1e-6
            rel_l1 = torch.linalg.norm(
                cur_feat - self.cache, ord=1).item() / norm
            self.accumulated += self._rescale(rel_l1)

            if self.accumulated < self.threshold:
                use_cache = True
                noise_pred = self.cache_noise_pred
            else:
                self.accumulated = 0.0
                self.cache = cur_feat
                self.cache_t = t
                noise_pred = model_fn(latents, t, cond)
                self.cache_noise_pred = noise_pred
        else:
            self.accumulated = 0.0
            self.cache = cur_feat
            self.cache_t = t
            noise_pred = model_fn(latents, t, cond)
            self.cache_noise_pred = noise_pred

        return noise_pred, use_cache

    def reset(self):
        super().reset()
        self.accumulated = 0.0
