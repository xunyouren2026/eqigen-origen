# utils/auto_config_adapter.py
import torch
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AutoConfigAdapter:
    """
    智能配置适配器
    根据当前 GPU 显存和架构，动态调整 config 中的关键性能参数，
    在保证不 OOM 的前提下最大化生成速度。
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.vram_total_gb = 0.0
        self.vram_reserved_gb = 2.5  # 基础预留，稍后动态调整
        self.gpu_name = "CPU"
        self.arch_name = "Unknown"
        self.compute_capability = (0, 0)
        self.supports_fp8 = False

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.vram_total_gb = props.total_memory / (1024 ** 3)
            self.gpu_name = props.name
            self.compute_capability = props.major, props.minor

            # 获取当前 PyTorch 已分配的显存（近似）
            allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
            # 预留额外安全余量（可通过环境变量调整）
            reserved_extra = float(os.environ.get('VRAM_RESERVED_EXTRA', 1.0))
            self.vram_reserved_gb = 2.5 + allocated_mem + reserved_extra

            # 识别架构
            if props.major == 9:
                self.arch_name = "Hopper"
            elif props.major == 8 and props.minor >= 9:
                self.arch_name = "Ada Lovelace"
            elif props.major == 8:
                self.arch_name = "Ampere"
            elif props.major == 7:
                self.arch_name = "Volta/Turing"
            else:
                self.arch_name = "Legacy"

            # FP8 支持检测 (需要 Ada 或 Hopper)
            self.supports_fp8 = (props.major == 9) or (
                props.major == 8 and props.minor >= 9)

    def detect_profile(self) -> str:
        """检测硬件档位"""
        if not torch.cuda.is_available():
            return "cpu_safe"

        usable_vram = self.vram_total_gb - self.vram_reserved_gb

        if usable_vram >= 20.0:
            return "flagship_ultra"   # 4090, A100, H100
        elif usable_vram >= 13.0:
            return "high_perf"        # 4080, 3090, 4070Ti
        elif usable_vram >= 9.0:
            return "balanced"         # 4070, 3060 12G, 4060Ti 16G
        elif usable_vram >= 6.0:
            return "entry_mid"        # 3060 8G, 2060, 3050
        else:
            return "entry_low"        # <6G

    def apply_optimizations(self, config) -> None:
        """应用优化到 config 对象"""
        profile = self.detect_profile()

        logger.info("="*60)
        logger.info(f"🚀 [AutoConfig] 硬件检测报告")
        logger.info(f"   GPU: {self.gpu_name}")
        logger.info(
            f"   架构：{self.arch_name} (Compute {self.compute_capability})")
        logger.info(f"   总显存：{self.vram_total_gb:.1f} GB")
        logger.info(
            f"   可用显存 (预估): {self.vram_total_gb - self.vram_reserved_gb:.1f} GB")
        logger.info(f"   FP8 支持：{self.supports_fp8}")
        logger.info(f"   匹配策略：[{profile}]")
        logger.info("-"*60)

        # 默认值 (安全底线)
        adjustments = {
            'max_block_frames': 48,
            'memory_size': 256,
            'tile_batch_size': 1,
            'vae_type': 'image',
            'use_turboquant': False,
            'teacache_threshold': 0.25,
            'use_internal_memory': False,
            'use_multi_scale_memory': False
        }

        if profile == "flagship_ultra":
            adjustments.update({
                'max_block_frames': 192,
                'memory_size': 1024,
                'tile_batch_size': 8,
                'vae_type': 'lean',
                'use_turboquant': self.supports_fp8,
                'teacache_threshold': 0.15,
                'use_internal_memory': True,
                'use_multi_scale_memory': True
            })
            logger.info("✅ 已启用旗舰级优化：LeanVAE + 192 帧块 + 1024 记忆容量 + FP8")

        elif profile == "high_perf":
            adjustments.update({
                'max_block_frames': 128,
                'memory_size': 512,
                'tile_batch_size': 4,
                'vae_type': 'lean',
                'use_turboquant': self.supports_fp8,
                'teacache_threshold': 0.18,
                'use_internal_memory': True,
                'use_multi_scale_memory': True
            })
            logger.info("✅ 已启用高性能优化：LeanVAE + 128 帧块 + 512 记忆容量")

        elif profile == "balanced":
            adjustments.update({
                'max_block_frames': 96,
                'memory_size': 512,
                'tile_batch_size': 2,
                'vae_type': 'lean',
                'use_turboquant': False,
                'teacache_threshold': 0.20,
                'use_internal_memory': True,
                'use_multi_scale_memory': False
            })
            logger.info("✅ 已启用均衡优化：LeanVAE + 96 帧块 + 适度记忆")

        elif profile in ["entry_mid", "entry_low"]:
            adjustments.update({
                'max_block_frames': 48 if profile == "entry_low" else 64,
                'memory_size': 128 if profile == "entry_low" else 256,
                'tile_batch_size': 1,
                'vae_type': 'image',
                'use_turboquant': False,
                'teacache_threshold': 0.30,
                'use_internal_memory': False,
                'use_multi_scale_memory': False
            })
            logger.warning("⚠️ 显存紧张，已切换至安全模式：普通 VAE + 小块生成 + 关闭高级记忆")

        # 检查 LeanVAE 权重是否存在，若不存在则回退到 image
        if adjustments['vae_type'] == 'lean':
            lean_path = getattr(
                config.model, 'lean_vae_path', './models/LeanVAE')
            if not os.path.exists(lean_path):
                logger.warning(
                    f"[AutoConfig] LeanVAE not found at {lean_path}, falling back to 'image' VAE.")
                adjustments['vae_type'] = 'image'

        # 执行应用
        applied_count = 0
        for key, value in adjustments.items():
            if hasattr(config.model, key):
                old_val = getattr(config.model, key)
                if old_val != value:
                    setattr(config.model, key, value)
                    logger.debug(f"   调整 {key}: {old_val} -> {value}")
                    applied_count += 1
            else:
                logger.warning(f"   配置项 {key} 不存在于 config.model 中")

        # 特殊处理：如果显存很小，强制关闭一些重型功能
        if self.vram_total_gb < 12.0:
            config.model.use_hierarchical_memory = False
            config.model.use_learned_compressor = False
            config.model.use_token_routing = False
            logger.info("   强制关闭分级记忆和复杂路由以节省显存")

        logger.info(f"🎉 [AutoConfig] 成功应用 {applied_count} 项动态优化配置！")
        logger.info("="*60)
