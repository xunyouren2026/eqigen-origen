# trainer.py
# AI 视频生成系统 - 训练器核心模块
# 功能：支持渐进式训练 (至 256 帧)、物理约束损失 (NS-Diff)、高级帧损坏、多阶段蒸馏
# 状态：✅ 已修复物理损失 Batch 索引错误，已启用 256 帧渐进式训练
# 新增：✅ 物理损失向量化 (批量处理)   ✅ DPO 训练集成 (可选)

import os
import math
import logging
import numpy as np
import random
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, LCMScheduler

# 可选依赖
try:
    import wandb
except ImportError:
    wandb = None

try:
    from torchvision.models import vgg16
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False

# 尝试导入 RAFT 光流模型（用于物理约束）
try:
    from raft import RAFT
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    logging.warning("RAFT not found. Physics loss will use simple gradient.")

# 项目内部模块
from models.ema import EMA
from metrics import compute_fvd, compute_psnr, compute_ssim, compute_clip_score
from utils import save_video, get_rank, get_world_size, to_device
from models.enhanced_physics_scorer import EnhancedPhysicsScorer
from inferencer import Inferencer

# ==================== 高级帧损坏模块 (23 种模式) ====================
from enum import Enum


class CorruptionType(Enum):
    EXPOSURE = 1
    GAMMA = 2
    CONTRAST = 3
    SATURATION = 4
    GAUSSIAN_NOISE = 5
    POISSON_NOISE = 6
    SALT_PEPPER_NOISE = 7
    GAUSSIAN_BLUR = 8
    MOTION_BLUR = 9
    MEDIAN_BLUR = 10
    JPEG_ARTIFACT = 11
    DOWNSAMPLE_UPSAMPLE = 12
    RANDOM_BLOCK = 13
    RANDOM_LINE = 14
    RANDOM_ERASE = 15
    COLOR_JITTER = 16
    GRAYSCALE = 17
    CHANNEL_SWAP = 18
    FRAME_REPEAT = 19
    FRAME_DROP = 20
    FRAME_SWAP = 21
    TEMPORAL_SHUFFLE = 22
    MOTION_BLUR_TEMPORAL = 23


class FrameCorruption:
    @staticmethod
    def exposure(frame, strength=0.5):
        factor = 1 + (torch.rand(1).item() - 0.5) * strength * 2
        return torch.clamp(frame * factor, -1, 1)

    @staticmethod
    def gamma(frame, strength=0.5):
        gamma = 1 + (torch.rand(1).item() - 0.5) * strength
        sign = 1 if frame.min() >= 0 else -1
        return sign * torch.abs(frame) ** gamma

    @staticmethod
    def contrast(frame, strength=0.5):
        mean = frame.mean()
        factor = 1 + (torch.rand(1).item() - 0.5) * strength
        return mean + (frame - mean) * factor

    @staticmethod
    def saturation(frame, strength=0.5):
        gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        sat = 1 + (torch.rand(1).item() - 0.5) * strength
        return frame * sat + gray * (1 - sat)

    @staticmethod
    def gaussian_noise(frame, strength=0.5):
        sigma = strength * 0.2
        return frame + torch.randn_like(frame) * sigma

    @staticmethod
    def poisson_noise(frame, strength=0.5):
        scale = 1.0
        frame_scaled = (frame + 1) / 2
        noisy = torch.poisson(frame_scaled * scale) / scale
        return noisy * 2 - 1

    @staticmethod
    def salt_pepper_noise(frame, strength=0.5):
        prob = strength * 0.1
        mask = torch.rand_like(frame) < prob
        frame = torch.where(mask, torch.rand_like(frame) * 2 - 1, frame)
        return frame

    @staticmethod
    def gaussian_blur(frame, strength=0.5):
        kernel_size = int(strength * 10) // 2 * 2 + 1
        kernel_size = max(3, kernel_size)
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                            device=frame.device) / (kernel_size*kernel_size)
        frame = frame.unsqueeze(0)
        blurred = F.conv2d(
            frame, kernel, padding=kernel_size // 2, groups=frame.shape[1])
        return blurred.squeeze(0)

    @staticmethod
    def motion_blur(frame, strength=0.5):
        kernel_size = int(strength * 10) + 1
        kernel = torch.zeros(kernel_size, kernel_size, device=frame.device)
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        frame = frame.unsqueeze(0)
        blurred = F.conv2d(
            frame, kernel, padding=kernel_size // 2, groups=frame.shape[1])
        return blurred.squeeze(0)

    @staticmethod
    def median_blur(frame, strength=0.5):
        kernel_size = int(strength * 5) + 1
        kernel_size = kernel_size if kernel_size % 2 else kernel_size+1
        return F.avg_pool2d(frame.unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2).squeeze(0)

    @staticmethod
    def jpeg_artifact(frame, strength=0.5):
        scale = 0.5 + strength * 0.5
        h, w = frame.shape[-2:]
        small = F.interpolate(frame.unsqueeze(
            0), scale_factor=scale, mode='bilinear')
        up = F.interpolate(small, size=(h, w), mode='bilinear')
        return up.squeeze(0)

    @staticmethod
    def downsample_upsample(frame, strength=0.5):
        scale = 0.2 + strength * 0.6
        h, w = frame.shape[-2:]
        small = F.interpolate(frame.unsqueeze(
            0), scale_factor=scale, mode='bilinear')
        up = F.interpolate(small, size=(h, w), mode='bilinear')
        return up.squeeze(0)

    @staticmethod
    def random_block(frame, strength=0.5):
        h, w = frame.shape[-2:]
        block_size = int(min(h, w) * (0.05 + strength * 0.15))
        x = torch.randint(0, w - block_size, (1,)).item()
        y = torch.randint(0, h - block_size, (1,)).item()
        frame[:, y:y+block_size, x:x+block_size] = -1.0
        return frame

    @staticmethod
    def random_line(frame, strength=0.5):
        h, w = frame.shape[-2:]
        if torch.rand(1).item() < 0.5:
            y = torch.randint(0, h, (1,)).item()
            frame[:, y, :] = -1.0
        else:
            x = torch.randint(0, w, (1,)).item()
            frame[:, :, x] = -1.0
        return frame

    @staticmethod
    def random_erase(frame, strength=0.5):
        h, w = frame.shape[-2:]
        block_size = int(min(h, w) * (0.05 + strength * 0.15))
        x = torch.randint(0, w - block_size, (1,)).item()
        y = torch.randint(0, h - block_size, (1,)).item()
        mean_val = frame.mean()
        frame[:, y:y+block_size, x:x+block_size] = mean_val
        return frame

    @staticmethod
    def color_jitter(frame, strength=0.5):
        shift = (torch.rand(3, device=frame.device) - 0.5) * strength * 0.5
        frame = frame + shift.view(3, 1, 1)
        return torch.clamp(frame, -1, 1)

    @staticmethod
    def grayscale(frame, strength=0.5):
        gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        return gray.unsqueeze(0).repeat(3, 1, 1)

    @staticmethod
    def channel_swap(frame, strength=0.5):
        perm = torch.randperm(3)
        return frame[perm]

# ==================== 物理约束模块（增强版） ====================


class PhysicsConstraint(nn.Module):
    """
    物理约束扩散（NS-Diff 完整实现）
    使用 RAFT 提取光流，计算速度场，施加纳维 - 斯托克斯方程约束
    """

    def __init__(self, device, use_raft=True):
        super().__init__()
        self.device = device
        self.use_raft = use_raft and RAFT_AVAILABLE
        if self.use_raft:
            self.raft = RAFT()
            checkpoint = torch.load(
                'models/raft-things.pth', map_location=device)
            self.raft.load_state_dict(checkpoint)
            self.raft.eval()
            for p in self.raft.parameters():
                p.requires_grad = False
        self.gravity = 9.8
        self.viscosity = 0.01

    def compute_velocity_field(self, frames):
        if not self.use_raft:
            diff = frames[:, :, 1:] - frames[:, :, :-1]
            return diff
        B, C, T, H, W = frames.shape
        frames_255 = ((frames + 1) / 2 * 255).clamp(0, 255)
        flows = []
        for b in range(B):
            for t in range(T-1):
                img1 = frames_255[b, :, t].permute(1, 2, 0).cpu().numpy()
                img2 = frames_255[b, :, t+1].permute(1, 2, 0).cpu().numpy()
                img1_t = torch.from_numpy(img1).float().permute(
                    2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                img2_t = torch.from_numpy(img2).float().permute(
                    2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                with torch.no_grad():
                    flow_low, flow_up = self.raft(img1_t, img2_t, iters=20)
                flows.append(flow_up.squeeze(0))
        flows = torch.stack(flows, dim=1)
        vel = torch.zeros(B, T-1, 3, H, W, device=self.device)
        vel[:, :, :2] = flows
        return vel.permute(0, 2, 1, 3, 4)

    def navier_stokes_loss(self, velocity_field):
        B, C, T, H, W = velocity_field.shape
        u = velocity_field[:, 0]
        v = velocity_field[:, 1] if C > 1 else torch.zeros_like(u)
        du_dt = torch.gradient(u, dim=1)[0]
        dv_dt = torch.gradient(v, dim=1)[0]
        du_dx = torch.gradient(u, dim=2)[0]
        du_dy = torch.gradient(u, dim=3)[0]
        dv_dx = torch.gradient(v, dim=2)[0]
        dv_dy = torch.gradient(v, dim=3)[0]
        div = du_dx + du_dy
        conv_u = u * du_dx + v * du_dy
        conv_v = u * dv_dx + v * dv_dy
        laplacian_u = self.laplacian_2d(u)
        laplacian_v = self.laplacian_2d(v)
        ns_u = du_dt + conv_u - self.viscosity * laplacian_u
        ns_v = dv_dt + conv_v - self.viscosity * laplacian_v
        ns_loss = (ns_u.pow(2).mean() + ns_v.pow(2).mean())
        incompressible_loss = div.abs().mean()
        return ns_loss + 0.1 * incompressible_loss

    def laplacian_2d(self, x):
        kernel = torch.tensor([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]],
                              dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        return F.conv2d(x.view(-1, 1, x.shape[-2], x.shape[-1]), kernel, padding=1).view(x.shape)

    def rigid_body_loss(self, trajectory):
        if trajectory.shape[1] < 3:
            return torch.tensor(0.0, device=self.device)
        acc = trajectory[:, 2:] - 2 * trajectory[:, 1:-1] + trajectory[:, :-2]
        jerk = acc[:, 2:] - 2 * acc[:, 1:-1] + acc[:, :-2]
        return jerk.abs().mean()

    def forward(self, frames, trajectory=None):
        if frames.shape[2] < 2:
            return torch.tensor(0.0, device=self.device)
        velocity = self.compute_velocity_field(frames)
        ns_loss = self.navier_stokes_loss(velocity)
        rigid_loss = self.rigid_body_loss(
            trajectory) if trajectory is not None else 0.0
        return ns_loss + 0.01 * rigid_loss

# ==================== 关系注意力模块 ====================


class RelationalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_entities=8):
        super().__init__()
        self.num_heads = num_heads
        self.num_entities = num_entities
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos = nn.Parameter(torch.randn(
            1, num_heads, num_entities, num_entities))

    def forward(self, x, entity_mask=None):
        B, N, C = x.shape
        tokens_per_entity = N // self.num_entities
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self.rel_pos.repeat(
            B, 1, tokens_per_entity, tokens_per_entity)
        attn = attn + rel_bias
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class RelationalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context, entity_mask=None):
        B, N, C = x.shape
        _, M, _ = context.shape
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(
            B, M, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# ==================== 分层注意力 ====================


class HierarchicalAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, temporal_only=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SpatialTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.spatial_attn = HierarchicalAttention(dim, num_heads, window_size)
        self.temporal_attn = HierarchicalAttention(dim, num_heads, window_size)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.route_weights = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x, cond):
        spatial_out = self.spatial_attn(self.norm1(x))
        temporal_out = self.temporal_attn(self.norm1(x))
        w = torch.softmax(self.route_weights, dim=0)
        x = x + w[0] * spatial_out + w[1] * temporal_out
        x = x + self.cross_attn(self.norm2(x), cond, cond)[0]
        x = x + self.ffn(self.norm3(x))
        return x

# ==================== 蒸馏模块 ====================


class FlowHead(nn.Module):
    def __init__(self, dim, in_channels, out_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, dim), nn.SiLU(), nn.Linear(dim, out_channels))

    def forward(self, x):
        return self.head(x)


class TransferMatchingDistillation(nn.Module):
    def __init__(self, teacher_model, student_model, num_steps=4, sigma=0.01):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.num_steps = num_steps
        self.sigma = sigma

    def forward(self, x0, t, cond):
        with torch.no_grad():
            teacher_out = self.teacher(x0, t, cond)[0]
        student_out = self.student(x0, t, cond)[0]
        return F.mse_loss(student_out, teacher_out)


class DualStreamDistillation(nn.Module):
    def __init__(self, student, discriminator):
        super().__init__()
        self.student = student
        self.discriminator = discriminator
        self.dmd_loss = nn.MSELoss()
        self.adv_loss = nn.BCEWithLogitsLoss()

    def forward(self, x0, t, cond, target):
        pred = self.student(x0, t, cond)[0]
        dmd = self.dmd_loss(pred, target)
        fake_pred = self.discriminator(pred)
        real_pred = self.discriminator(target)
        adv = self.adv_loss(fake_pred, torch.ones_like(fake_pred)) + \
            self.adv_loss(real_pred, torch.zeros_like(real_pred))
        return dmd, adv

# ==================== 主训练器 ====================


class Trainer:
    def __init__(self, config, model, vae, scheduler, text_encoder, image_encoder, audio_encoder, video_encoder, lens_controller, device):
        self.config = config
        self.device = device
        self.global_step = 0
        self.best_fvd = float('inf')
        self.epoch = 0

        # 模式标识
        self.mode = config.model.mode   # 'creative' 或 'long'

        # 分布式设置
        self.is_distributed = config.train.distributed and torch.cuda.device_count() > 1
        self.local_rank = config.train.local_rank if self.is_distributed else 0
        self.world_size = config.train.world_size if self.is_distributed else 1

        # 模型
        self.model = model.to(device)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.image_encoder = image_encoder.to(
            device) if image_encoder is not None else None
        self.audio_encoder = audio_encoder.to(
            device) if audio_encoder is not None else None
        self.video_encoder = video_encoder.to(
            device) if video_encoder is not None else None
        self.lens_controller = lens_controller.to(
            device) if lens_controller is not None else None
        self.scheduler = scheduler

        # 分布式包装
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[
                             self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
            self.vae = DDP(self.vae, device_ids=[
                           self.local_rank], output_device=self.local_rank, find_unused_parameters=False)

        # 优化器
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.vae.parameters()),
            lr=config.train.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度
        warmup_steps = config.train.warmup_steps
        total_steps = config.train.epochs * \
            (config.data.batch_size * self.world_size)
        if config.train.lr_scheduler == 'cosine':
            self.scheduler_lr = CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup_steps)
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_steps)
                self.scheduler_lr = SequentialLR(
                    self.optimizer, [warmup_scheduler, self.scheduler_lr], milestones=[warmup_steps])
        else:
            self.scheduler_lr = None

        self.scaler = GradScaler() if config.train.mixed_precision else None
        self.ema = EMA(
            model, decay=config.train.ema_decay) if config.train.use_ema else None

        # 物理约束模块
        self.physics = PhysicsConstraint(
            device, use_raft=config.train.get('use_raft_physics', False))

        # 蒸馏模块
        self.distillation = None
        if config.train.get('use_distillation', False):
            self.distillation = TransferMatchingDistillation(
                self.model, self.model, num_steps=config.train.get('distill_steps', 4))

        # 双流蒸馏
        self.dual_stream = None
        if config.train.get('use_dual_stream', False):
            from models.discriminator import VideoDiscriminator
            self.discriminator = VideoDiscriminator().to(device)
            self.dual_stream = DualStreamDistillation(
                self.model, self.discriminator)

        # 日志
        if get_rank() == 0:
            self.writer = SummaryWriter(config.train.log_dir)
            if config.train.get('use_wandb', False) and wandb is not None:
                wandb.init(project="ai-video-generation", config=config)
        else:
            self.writer = None

        self.gradient_accumulation_steps = config.train.gradient_accumulation_steps
        self.use_gradient_checkpointing = config.train.get(
            'use_gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
            self.model.train()
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        # 恢复检查点
        if config.train.resume_from:
            self.load_checkpoint(config.train.resume_from)

        # 训练阶段
        self.stage = 0
        self._update_stage()

        # 多模态融合权重
        if config.model.multi_modal_fusion == "gate":
            self.modality_weights = nn.Parameter(
                torch.tensor([1.0, 1.0, 1.0])).to(device)
        else:
            self.modality_weights = None

        # VGG 感知损失
        if VGG_AVAILABLE:
            self.vgg = vgg16(pretrained=True).features.to(device).eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        else:
            self.vgg = None

        # ========== 渐进式训练状态 (已升级至 256 帧) ==========
        self.current_progressive_stage = 0
        self.last_num_frames = self.config.data.num_frames

        # ========== Diagonal Distillation 需要的总步数 ==========
        self.total_steps = 1  # 占位，将在 train 中设置

        # ========== DPO 训练相关 ==========
        self.dpo_pairs = None
        self.use_dpo = config.train.get('use_dpo', False)
        self.dpo_weight = config.train.get('dpo_weight', 0.1)
        if self.use_dpo:
            self.dpo_step_interval = config.train.get(
                'dpo_step_interval', 10)  # 每多少步采样一次 DPO

    def _update_stage(self):
        if self.epoch < self.config.train.get('stage1_epochs', 10):
            self.stage = 0
        elif self.epoch < self.config.train.get('stage2_epochs', 30):
            self.stage = 1
        elif self.epoch < self.config.train.get('stage3_epochs', 80):
            self.stage = 2
        else:
            self.stage = 3

    # ========== 历史帧损坏增强（23 种模式） ==========
    def corrupt_history(self, latents, p=0.2, corruption_types=None, num_modes=2):
        B, C, T, H, W = latents.shape
        corrupted = latents.clone()

        if corruption_types is None:
            corruption_types = list(CorruptionType)

        single_frame_modes = [m for m in corruption_types if m not in
                              [CorruptionType.FRAME_REPEAT, CorruptionType.FRAME_DROP,
                               CorruptionType.FRAME_SWAP, CorruptionType.TEMPORAL_SHUFFLE,
                               CorruptionType.MOTION_BLUR_TEMPORAL]]

        for b in range(B):
            # 整体时序损坏
            if torch.rand(1).item() < p * 0.5:
                temporal_modes = [CorruptionType.FRAME_REPEAT, CorruptionType.FRAME_DROP,
                                  CorruptionType.FRAME_SWAP, CorruptionType.TEMPORAL_SHUFFLE,
                                  CorruptionType.MOTION_BLUR_TEMPORAL]
                chosen = np.random.choice(temporal_modes, 1)[0]
                if chosen == CorruptionType.FRAME_REPEAT:
                    src = torch.randint(1, T, (1,)).item()
                    dst = torch.randint(
                        src+1, T, (1,)).item() if src+1 < T else src
                    corrupted[b, :, dst] = corrupted[b, :, src]
                elif chosen == CorruptionType.FRAME_DROP:
                    idx = torch.randint(1, T, (1,)).item()
                    corrupted[b, :, idx] = corrupted[b, :, idx-1]
                elif chosen == CorruptionType.FRAME_SWAP:
                    idx1 = torch.randint(1, T, (1,)).item()
                    idx2 = torch.randint(1, T, (1,)).item()
                    if idx1 != idx2:
                        corrupted[b, :, idx1], corrupted[b, :,
                                                         idx2] = corrupted[b, :, idx2], corrupted[b, :, idx1]
                elif chosen == CorruptionType.TEMPORAL_SHUFFLE:
                    seg_len = torch.randint(2, min(5, T), (1,)).item()
                    start = torch.randint(1, T - seg_len, (1,)).item()
                    seg = corrupted[b, :, start:start+seg_len]
                    perm = torch.randperm(seg_len)
                    corrupted[b, :, start:start+seg_len] = seg[:, perm]
                elif chosen == CorruptionType.MOTION_BLUR_TEMPORAL:
                    blur_len = torch.randint(2, min(4, T), (1,)).item()
                    for t in range(blur_len, T):
                        corrupted[b, :, t] = torch.mean(
                            corrupted[b, :, t-blur_len:t], dim=1)

            for t in range(1, T):
                if torch.rand(1).item() < p:
                    k = torch.randint(1, num_modes + 1, (1,)).item()
                    selected_modes = np.random.choice(
                        single_frame_modes, k, replace=False)
                    frame = corrupted[b, :, t]
                    for mode in selected_modes:
                        strength = torch.rand(1).item() * 0.5
                        frame = self._apply_corruption(frame, mode, strength)
                    corrupted[b, :, t] = frame
        return corrupted

    def _apply_corruption(self, frame, mode, strength):
        if mode == CorruptionType.EXPOSURE:
            return FrameCorruption.exposure(frame, strength)
        elif mode == CorruptionType.GAMMA:
            return FrameCorruption.gamma(frame, strength)
        elif mode == CorruptionType.CONTRAST:
            return FrameCorruption.contrast(frame, strength)
        elif mode == CorruptionType.SATURATION:
            return FrameCorruption.saturation(frame, strength)
        elif mode == CorruptionType.GAUSSIAN_NOISE:
            return FrameCorruption.gaussian_noise(frame, strength)
        elif mode == CorruptionType.POISSON_NOISE:
            return FrameCorruption.poisson_noise(frame, strength)
        elif mode == CorruptionType.SALT_PEPPER_NOISE:
            return FrameCorruption.salt_pepper_noise(frame, strength)
        elif mode == CorruptionType.GAUSSIAN_BLUR:
            return FrameCorruption.gaussian_blur(frame, strength)
        elif mode == CorruptionType.MOTION_BLUR:
            return FrameCorruption.motion_blur(frame, strength)
        elif mode == CorruptionType.MEDIAN_BLUR:
            return FrameCorruption.median_blur(frame, strength)
        elif mode == CorruptionType.JPEG_ARTIFACT:
            return FrameCorruption.jpeg_artifact(frame, strength)
        elif mode == CorruptionType.DOWNSAMPLE_UPSAMPLE:
            return FrameCorruption.downsample_upsample(frame, strength)
        elif mode == CorruptionType.RANDOM_BLOCK:
            return FrameCorruption.random_block(frame, strength)
        elif mode == CorruptionType.RANDOM_LINE:
            return FrameCorruption.random_line(frame, strength)
        elif mode == CorruptionType.RANDOM_ERASE:
            return FrameCorruption.random_erase(frame, strength)
        elif mode == CorruptionType.COLOR_JITTER:
            return FrameCorruption.color_jitter(frame, strength)
        elif mode == CorruptionType.GRAYSCALE:
            return FrameCorruption.grayscale(frame, strength)
        elif mode == CorruptionType.CHANNEL_SWAP:
            return FrameCorruption.channel_swap(frame, strength)
        else:
            return frame

    # ========== 统一条件序列构建 ==========
    def build_cond_seq(self, text_emb, img_cond, audio_cond, video_cond, lens_cond, traj_cond):
        cond_list = []
        cond_list.append(text_emb.mean(dim=1, keepdim=True))
        if img_cond is not None:
            cond_list.append(img_cond)
        if audio_cond is not None:
            cond_list.append(audio_cond)
        if video_cond is not None:
            cond_list.append(video_cond)
        if lens_cond is not None:
            cond_list.append(lens_cond)
        if traj_cond is not None:
            cond_list.append(traj_cond)
        if len(cond_list) == 0:
            return None
        return torch.cat(cond_list, dim=1)

    def _compute_importance(self, memory):
        diff = torch.diff(memory, dim=1).norm(dim=-1)
        importance = torch.cat([diff[:, :1], diff, diff[:, -1:]], dim=1)
        importance = importance / (importance.mean(dim=1, keepdim=True) + 1e-6)
        return importance.detach()

    def train_step(self, batch):
        videos = batch['video'].to(self.device).permute(0, 2, 1, 3, 4)
        texts = batch['text']
        ref_images = batch.get('reference_images', [])
        ref_videos = batch.get('reference_videos', [])
        ref_audios = batch.get('reference_audios', [])
        lens_script = batch.get('lens_script', None)
        entity_mask = batch.get('entity_mask', None)
        trajectory = batch.get('trajectory', None)

        # 长视频模式才进行阶段裁剪
        if self.mode == 'long' and self.stage == 0:
            videos = videos[:, :, :1, :, :]

        _, _, T, H, W = videos.shape
        spatial_compress = self.vae.spatial_compress
        temporal_compress = self.vae.temporal_compress
        pad_h = (spatial_compress - (H % spatial_compress)) % spatial_compress
        pad_w = (spatial_compress - (W % spatial_compress)) % spatial_compress
        pad_t = (temporal_compress - (T %
                 temporal_compress)) % temporal_compress
        if pad_h > 0 or pad_w > 0 or pad_t > 0:
            videos = F.pad(videos, (0, pad_w, 0, pad_h, 0, pad_t, 0, 0))

        with torch.no_grad():
            latents, mean, logvar = self.vae.encode(videos)
        kl_loss = -0.5 * torch.sum(1 + logvar -
                                   mean.pow(2) - logvar.exp()) / videos.numel()

        # ========== Diagonal Distillation 精细控制（基于帧数和训练进度） ==========
        if self.mode == 'long' and self.distillation is not None and self.stage >= 2:
            current_num_frames = videos.shape[2]
            frame_factor = 1.0 - current_num_frames / self.config.data.num_frames
            progress = min(1.0, self.global_step / (self.total_steps * 0.8))
            max_t_factor = frame_factor * (1 - progress * 0.8)
            max_t = int(self.scheduler.num_timesteps * max_t_factor)
            max_t = max(1, min(self.scheduler.num_timesteps, max_t))
            t = torch.randint(
                0, max_t, (latents.shape[0],), device=self.device)
        else:
            t = torch.randint(0, self.scheduler.num_timesteps,
                              (latents.shape[0],), device=self.device)

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.q_sample(latents, t, noise)

        # 高级帧损坏（仅长视频模式且配置启用）
        if self.mode == 'long' and self.config.model.get('use_frame_corruption', False):
            noisy_latents = self.corrupt_history(
                noisy_latents,
                p=self.config.model.get('corruption_prob', 0.15),
                corruption_types=self.config.model.get(
                    'corruption_modes', None),
                num_modes=self.config.model.get('corruption_num_modes', 2)
            )

        text_emb = self.text_encoder(texts)

        img_cond = None
        if self.image_encoder and ref_images:
            img_list = []
            for img in ref_images[:self.config.model.max_reference_images]:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img_list.append(img.to(self.device))
            img_feats = self.image_encoder(img_list)
            img_cond = img_feats.mean(dim=1, keepdim=True)

        audio_cond = None
        if self.audio_encoder and ref_audios:
            aud_list = []
            for aud in ref_audios[:self.config.model.max_reference_audios]:
                if aud.dim() == 1:
                    aud = aud.unsqueeze(0)
                aud = aud.to(self.device)
                aud_feat = self.audio_encoder(aud)
                aud_list.append(aud_feat.mean(dim=1, keepdim=True))
            if aud_list:
                audio_cond = torch.stack(aud_list, dim=1).mean(dim=1)

        video_cond = None
        if self.video_encoder and ref_videos:
            vid_list = []
            for vid in ref_videos[:self.config.model.max_reference_videos]:
                if vid.dim() == 4:
                    vid = vid.unsqueeze(0)
                vid = vid.to(self.device)
                vid_feat = self.video_encoder(vid)
                vid_list.append(vid_feat.unsqueeze(1))
            if vid_list:
                video_cond = torch.stack(vid_list, dim=1).mean(dim=1)

        lens_cond = None
        if lens_script is not None and self.lens_controller is not None:
            lens_cond = self.lens_controller(lens_script)
            lens_cond = lens_cond.mean(dim=1, keepdim=True)

        if self.config.model.use_unified_cond:
            cond_seq = self.build_cond_seq(
                text_emb, img_cond, audio_cond, video_cond, lens_cond, None)
            cond_emb = cond_seq
        else:
            if self.modality_weights is not None:
                weights = torch.softmax(self.modality_weights, dim=0)
                cond_emb = weights[0] * text_emb.mean(dim=1, keepdim=True)
                if img_cond is not None:
                    cond_emb = cond_emb + weights[1] * img_cond
                if audio_cond is not None:
                    cond_emb = cond_emb + weights[2] * audio_cond
                if video_cond is not None:
                    cond_emb = cond_emb + weights[0] * video_cond
                if lens_cond is not None:
                    cond_emb = cond_emb + lens_cond
            else:
                cond_emb = text_emb.mean(dim=1, keepdim=True)
                if img_cond is not None:
                    cond_emb = cond_emb + img_cond
                if audio_cond is not None:
                    cond_emb = cond_emb + audio_cond
                if video_cond is not None:
                    cond_emb = cond_emb + video_cond
                if lens_cond is not None:
                    cond_emb = cond_emb + lens_cond

        noise_pred, flow_pred, new_state = self.model(
            noisy_latents, t, cond_emb,
            prev_state=None, entity_mask=entity_mask, return_state=True
        )

        # 加权 MSE（首帧锚定增强）
        if self.config.train.get('first_frame_weight', 1.0) > 1.0:
            loss_all = F.mse_loss(noise_pred, noise, reduction='none')
            weights = torch.ones_like(loss_all)
            weights[:, :, 0, :, :] *= self.config.train.first_frame_weight
            mse_loss = (loss_all * weights).mean()
        else:
            mse_loss = F.mse_loss(noise_pred, noise)

        # ========== 物理损失向量化 ==========
        physics_loss = torch.tensor(0.0, device=self.device)
        if self.mode == 'long' and self.stage >= 1 and videos.shape[2] > 1:
            # 一次性对整个 batch 进行解码，而不是逐样本循环
            pred_latents = self.scheduler.step(
                noise_pred, t, noisy_latents).prev_sample
            pred_frames = self.vae.decode(pred_latents).clamp(-1, 1)
            physics_loss = self.physics(pred_frames, trajectory)

        # 蒸馏损失（仅长视频模式）
        distill_loss = 0.0
        if self.mode == 'long' and self.distillation is not None and self.stage >= 2:
            distill_loss = self.distillation(noisy_latents, t, cond_emb)

        # DMD + 对抗损失（仅长视频模式）
        dmd_loss, adv_loss = 0.0, 0.0
        if self.mode == 'long' and self.dual_stream is not None and self.stage >= 2:
            dmd_loss, adv_loss = self.dual_stream(
                noisy_latents, t, cond_emb, noise)

        # 重建损失（仅长视频模式）
        recon_loss = 0.0
        if self.mode == 'long' and self.config.model.get('use_adaptive_compressor', False) and self.config.train.get('recon_weight', 0.0) > 0:
            if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                first_block = self.model.blocks[0]
                if hasattr(first_block, 'compressor') and first_block.compressor is not None:
                    memory = self.model.state_memory.expand(
                        latents.size(0), -1, -1)
                    importance = self._compute_importance(memory)
                    compressed, recon = first_block.compressor(
                        memory, return_recon=True)
                    recon_loss = F.mse_loss(recon, memory, reduction='none')
                    weighted_recon_loss = (
                        recon_loss * importance.unsqueeze(-1)).mean()
                    recon_loss = weighted_recon_loss

        # 感知损失（仅长视频模式）
        perceptual_loss = 0.0
        if self.mode == 'long' and self.stage >= 1 and VGG_AVAILABLE and self.config.train.get('perceptual_weight', 0.0) > 0:
            with torch.no_grad():
                # 使用批量预测（只需第一个样本，但保持张量形状）
                pred_latents = self.scheduler.step(
                    noise_pred, t, noisy_latents).prev_sample
                pred_frames = self.vae.decode(pred_latents).clamp(-1, 1)
                real_frames = videos
                # 取第一个帧作为图像感知损失（简化）
                pred_frame = pred_frames[:, :, 0, :, :]
                real_frame = real_frames[:, :, 0, :, :]
                pred_224 = F.interpolate(pred_frame, size=(
                    224, 224), mode='bilinear', align_corners=False)
                real_224 = F.interpolate(real_frame, size=(
                    224, 224), mode='bilinear', align_corners=False)

                def get_features(x):
                    features = []
                    for layer in self.vgg:
                        x = layer(x)
                        if isinstance(layer, nn.ReLU):
                            features.append(x)
                            if len(features) == 4:
                                break
                    return features
                pred_feats = get_features(pred_224)
                real_feats = get_features(real_224)
                perceptual_loss = sum(F.mse_loss(p, r) for p, r in zip(
                    pred_feats, real_feats)) / len(pred_feats)

        # 总损失计算（根据模式）
        if self.mode == 'creative':
            total_loss = mse_loss + self.config.model.vae_kl_weight * kl_loss
        else:
            total_loss = (mse_loss +
                          self.config.model.vae_kl_weight * kl_loss +
                          self.config.train.get('physics_weight', 0.1) * physics_loss +
                          self.config.train.get('distill_weight', 0.1) * distill_loss +
                          self.config.train.get('dmd_weight', 0.1) * dmd_loss +
                          self.config.train.get('adv_weight', 0.1) * adv_loss +
                          self.config.train.get('recon_weight', 0.0) * recon_loss +
                          self.config.train.get('perceptual_weight', 0.1) * perceptual_loss)

        # DPO 损失（如果启用且数据存在）
        dpo_loss = torch.tensor(0.0, device=self.device)
        if self.use_dpo and self.dpo_pairs is not None and self.global_step % self.dpo_step_interval == 0:
            # 随机采样一个 DPO 对
            dpo_batch = random.choice(self.dpo_pairs)
            dpo_loss = self.train_dpo_step(dpo_batch)
            total_loss = total_loss + self.dpo_weight * dpo_loss

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if self.config.train.get('grad_clip', 0) > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.grad_clip)

        return total_loss.item()

    def train(self, train_loader, val_loader):
        # [8, 16, 32, 64, 128, 256]
        progressive_frames = self.config.train.progressive_frames
        progressive_epochs = self.config.train.progressive_epochs
        current_stage = self.current_progressive_stage

        # 设置总步数，供 Diagonal Distillation 使用
        self.total_steps = len(train_loader) * self.config.train.epochs

        # 加载 DPO 数据（如果启用）
        if self.use_dpo:
            dpo_path = os.path.join(self.config.train.save_dir, "dpo_pairs.pt")
            if os.path.exists(dpo_path):
                self.dpo_pairs = torch.load(dpo_path, map_location='cpu')
                print(
                    f"Loaded {len(self.dpo_pairs)} DPO pairs from {dpo_path}")
            else:
                print(
                    f"DPO data not found at {dpo_path}, DPO training disabled.")
                self.use_dpo = False

        for epoch in range(self.epoch, self.config.train.epochs):
            self.epoch = epoch
            self._update_stage()

            # ========== 渐进式训练（仅长视频模式） ==========
            if self.mode == 'long':
                if current_stage < len(progressive_epochs) and epoch >= progressive_epochs[current_stage]:
                    current_stage += 1
                    if current_stage < len(progressive_frames):
                        new_num_frames = progressive_frames[current_stage]
                        if new_num_frames != self.last_num_frames:
                            print(
                                f"渐进式训练：切换至阶段 {current_stage}，帧数 {self.last_num_frames} -> {new_num_frames}")
                            if hasattr(train_loader.dataset, 'num_frames'):
                                train_loader.dataset.num_frames = new_num_frames
                            if hasattr(val_loader.dataset, 'num_frames'):
                                val_loader.dataset.num_frames = new_num_frames
                            self.last_num_frames = new_num_frames
                            self.current_progressive_stage = current_stage

            if hasattr(train_loader.dataset, 'set_stage'):
                train_loader.dataset.set_stage(self.stage)
            if hasattr(val_loader.dataset, 'set_stage'):
                val_loader.dataset.set_stage(self.stage)

            self.model.train()
            self.vae.train()
            if self.dual_stream is not None:
                self.discriminator.train()

            total_loss = 0.0
            pbar = tqdm(
                train_loader, desc=f"Stage {self.stage} Epoch {epoch+1}", disable=get_rank() != 0)

            for step, batch in enumerate(pbar):
                loss = self.train_step(batch)
                total_loss += loss

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler_lr:
                        self.scheduler_lr.step()

                    if self.ema:
                        self.ema.update()

                    self.global_step += 1

                    if self.writer and self.global_step % self.config.train.log_interval == 0:
                        self.writer.add_scalar('loss', loss, self.global_step)
                        self.writer.add_scalar('lr', self.scheduler_lr.get_last_lr()[
                                               0] if self.scheduler_lr else 0, self.global_step)
                        if self.config.train.get('use_wandb', False) and wandb is not None:
                            wandb.log({
                                'loss': loss,
                                'lr': self.scheduler_lr.get_last_lr()[0] if self.scheduler_lr else 0,
                                'stage': self.stage
                            }, step=self.global_step)

                    pbar.set_postfix(loss=loss)

            avg_loss = total_loss / len(train_loader)
            if self.writer:
                self.writer.add_scalar('epoch_loss', avg_loss, epoch)

            if (epoch + 1) % self.config.train.eval_interval == 0:
                metrics = self.validate(val_loader)
                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f'val/{k}', v, epoch)
                if metrics['fvd'] < self.best_fvd:
                    self.best_fvd = metrics['fvd']
                    self.save_checkpoint(epoch, is_best=True)
                self.save_checkpoint(epoch)

            if (epoch + 1) % 5 == 0 and get_rank() == 0:
                self.sample_video(epoch)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        self.vae.eval()
        total_metrics = {'fvd': 0.0, 'psnr': 0.0,
                         'ssim': 0.0, 'clip_score': 0.0}
        count = 0
        for batch in val_loader:
            videos = batch['video'].to(self.device).permute(0, 2, 1, 3, 4)
            texts = batch['text']
            shape = (videos.shape[0], self.config.model.vae_latent_channels,
                     videos.shape[2] // self.vae.temporal_compress,
                     videos.shape[3] // self.vae.spatial_compress,
                     videos.shape[4] // self.vae.spatial_compress)
            model_eval = self.ema.model if self.ema else self.model
            cond = self.text_encoder(texts)
            cond_emb = cond.mean(dim=1, keepdim=True)
            latents = torch.randn(shape, device=self.device)
            for t in self.scheduler.scheduler.timesteps:
                t_tensor = torch.full(
                    (shape[0],), t, device=self.device, dtype=torch.long)
                noise_pred, _, _ = model_eval(
                    latents, t_tensor, cond_emb, prev_state=None)
                latents = self.scheduler.step(
                    noise_pred, t, latents).prev_sample
            gen_videos = self.vae.decode(latents).cpu().numpy()
            real_videos = videos.cpu().numpy()
            total_metrics['fvd'] += compute_fvd(gen_videos, real_videos)
            total_metrics['psnr'] += compute_psnr(gen_videos, real_videos)
            total_metrics['ssim'] += compute_ssim(gen_videos, real_videos)
            total_metrics['clip_score'] += compute_clip_score(
                gen_videos, texts)
            count += 1
        for k in total_metrics:
            total_metrics[k] /= count
        return total_metrics

    def sample_video(self, epoch):
        prompt = "a cat playing with a ball"
        cond = self.text_encoder([prompt])
        cond_emb = cond.mean(dim=1, keepdim=True)
        target_res = self.config.data.image_size
        target_frames = self.config.data.num_frames
        t_comp = self.vae.temporal_compress
        s_comp = self.vae.spatial_compress
        shape = (1, self.config.model.vae_latent_channels,
                 (target_frames + t_comp - 1) // t_comp,
                 target_res // s_comp,
                 target_res // s_comp)
        model_sample = self.ema.model if self.ema else self.model
        latents = torch.randn(shape, device=self.device)
        for t in self.scheduler.scheduler.timesteps:
            t_tensor = torch.full(
                (1,), t, device=self.device, dtype=torch.long)
            noise_pred, _, _ = model_sample(
                latents, t_tensor, cond_emb, prev_state=None)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        video = self.vae.decode(latents).cpu()
        video = (video + 1) / 2 * 255
        video = video.squeeze(0).permute(1, 2, 3, 0).numpy().astype(np.uint8)
        video = video[:target_frames, :target_res, :target_res, :]
        save_video(video, os.path.join(
            self.config.train.save_dir, f"sample_{epoch}.mp4"), fps=8)

    def _compute_log_prob(self, latents, noise, t, cond):
        noise_pred, _, _ = self.model(latents, t, cond)
        mse = F.mse_loss(noise_pred, noise, reduction='none').mean(
            dim=(1, 2, 3, 4))
        return -mse

    def train_dpo_step(self, batch):
        pos_video = batch['positive'].to(self.device)
        neg_video = batch['negative'].to(self.device)
        prompt = batch['prompt']
        text_emb = self.text_encoder(prompt)
        cond_emb = text_emb.mean(dim=1, keepdim=True)
        t = torch.randint(0, self.scheduler.num_timesteps,
                          (pos_video.size(0),), device=self.device)
        noise = torch.randn_like(pos_video)
        noisy_pos = self.scheduler.q_sample(pos_video, t, noise)
        noisy_neg = self.scheduler.q_sample(neg_video, t, noise)
        pred_pos, _, _ = self.model(noisy_pos, t, cond_emb)
        pred_neg, _, _ = self.model(noisy_neg, t, cond_emb)
        log_prob_pos = -F.mse_loss(pred_pos, noise,
                                   reduction='none').mean(dim=(1, 2, 3, 4))
        log_prob_neg = -F.mse_loss(pred_neg, noise,
                                   reduction='none').mean(dim=(1, 2, 3, 4))
        loss = -F.logsigmoid(log_prob_pos - log_prob_neg).mean()
        return loss

    def _load_video_tensor(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (self.config.data.image_size, self.config.data.image_size))
            frames.append(frame)
        cap.release()
        video = np.array(frames, dtype=np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)
        return video.to(self.device)

    def generate_dpo_data(self, inferencer, num_samples=1000, num_candidates=8, save_path="dpo_pairs.pt"):
        scorer = EnhancedPhysicsScorer(
            self.device, use_raft=self.config.train.get('use_raft_physics', False))
        prompts = [
            "a ball rolling down a slope", "water pouring into a glass", "a bouncing ball",
            "a pendulum swinging", "a flag waving in wind", "a person walking",
            "a car driving on a road", "a falling leaf"
        ]
        prompts = prompts * (num_samples // len(prompts) + 1)
        prompts = prompts[:num_samples]
        pairs = []
        for idx, prompt in enumerate(prompts):
            candidates = []
            for _ in range(num_candidates):
                video_path, _ = inferencer.generate(
                    prompt, duration=2.0, fps=8, return_state=True)
                video_tensor = self._load_video_tensor(video_path)
                candidates.append(video_tensor)
            scores = [scorer.score(vid, prompt) for vid in candidates]
            best_idx = torch.argmax(torch.tensor(scores)).item()
            worst_idx = torch.argmin(torch.tensor(scores)).item()
            pairs.append({
                "prompt": prompt,
                "positive": candidates[best_idx],
                "negative": candidates[worst_idx]
            })
            if (idx+1) % 10 == 0:
                print(f"Generated {idx+1}/{len(prompts)} pairs")
        torch.save(pairs, save_path)
        print(f"DPO data saved to {save_path}")
        return pairs

    def save_checkpoint(self, epoch, is_best=False):
        if get_rank() != 0:
            return
        state = {
            'epoch': epoch,
            'model': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'vae': self.vae.module.state_dict() if self.is_distributed else self.vae.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler_lr.state_dict() if self.scheduler_lr else None,
            'ema': self.ema.shadow if self.ema else None,
            'global_step': self.global_step,
            'progressive_stage': self.current_progressive_stage,
            'last_num_frames': self.last_num_frames,
        }
        if self.image_encoder is not None:
            state['image_encoder'] = self.image_encoder.state_dict()
        if self.audio_encoder is not None:
            state['audio_encoder'] = self.audio_encoder.state_dict()
        if self.video_encoder is not None:
            state['video_encoder'] = self.video_encoder.state_dict()
        if self.lens_controller is not None:
            state['lens_controller'] = self.lens_controller.state_dict()
        path = os.path.join(self.config.train.save_dir,
                            f"checkpoint_epoch_{epoch}.pt")
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(
                self.config.train.save_dir, "best_model.pt")
            torch.save(state, best_path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        if self.is_distributed:
            self.model.module.load_state_dict(state['model'])
            self.vae.module.load_state_dict(state['vae'])
        else:
            self.model.load_state_dict(state['model'])
            self.vae.load_state_dict(state['vae'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler_lr and state['scheduler']:
            self.scheduler_lr.load_state_dict(state['scheduler'])
        if self.ema and state['ema']:
            self.ema.shadow = state['ema']
        if self.image_encoder is not None and 'image_encoder' in state:
            self.image_encoder.load_state_dict(state['image_encoder'])
        if self.audio_encoder is not None and 'audio_encoder' in state:
            self.audio_encoder.load_state_dict(state['audio_encoder'])
        if self.video_encoder is not None and 'video_encoder' in state:
            self.video_encoder.load_state_dict(state['video_encoder'])
        if self.lens_controller is not None and 'lens_controller' in state:
            self.lens_controller.load_state_dict(state['lens_controller'])
        self.global_step = state['global_step']
        self.epoch = state['epoch'] + 1
        self.current_progressive_stage = state.get('progressive_stage', 0)
        self.last_num_frames = state.get(
            'last_num_frames', self.config.data.num_frames)
        logging.info(
            f"Loaded checkpoint from {path}, progressive stage {self.current_progressive_stage}, last num_frames {self.last_num_frames}")
