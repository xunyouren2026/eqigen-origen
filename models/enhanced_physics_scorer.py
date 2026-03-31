import torch
import torch.nn as nn
import torch.nn.functional as F
from .physics_constraint import PhysicsConstraint


class EnhancedPhysicsScorer(nn.Module):
    """
    多指标物理评分器：融合光流一致性、刚体运动、碰撞检测、弹性势能等
    输出 0~1 分数，越高表示物理越合理
    """

    def __init__(self, device='cuda', use_raft=True):
        super().__init__()
        self.device = device
        self.physics = PhysicsConstraint(device, use_raft=use_raft)
        # 注册权重参数（可学习，但这里固定）
        self.register_buffer('weights', torch.tensor(
            [1.0, 0.5, 0.2, 0.1]))  # NS, 刚体, 碰撞, 弹性
        # 平滑系数，避免分数极端
        self.smooth = 1e-6

    def score(self, video, trajectory=None, prompt=None):
        """
        video: (C, T, H, W) 或 (B, C, T, H, W)
        trajectory: 可选，刚体轨迹 (B, T, 3) 或 (T,3)
        返回 0~1 物理合理性分数（标量）
        """
        # 统一batch维度
        if video.dim() == 4:
            video = video.unsqueeze(0)
        B, C, T, H, W = video.shape
        if trajectory is not None and trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)

        # 计算各项物理损失
        # 注意：PhysicsConstraint.forward 返回标量损失（越小越物理）
        phys_loss = self.physics(video, trajectory)   # 标量

        # 将损失转换为分数（0~1），使用 logistic 映射
        # 损失范围通常在 0~10 之间，这里使用指数衰减
        score = 1.0 / (1.0 + phys_loss.item() + self.smooth)

        return score

    def score_batch(self, videos, trajectories=None, prompts=None):
        """
        批量评分
        videos: (B, C, T, H, W)
        trajectories: (B, T, 3) 或 None
        prompts: 列表，可选
        """
        scores = []
        for i in range(videos.size(0)):
            traj = trajectories[i] if trajectories is not None else None
            s = self.score(videos[i], traj, prompts[i] if prompts else None)
            scores.append(s)
        return torch.tensor(scores, device=videos.device)

    def get_component_losses(self, video, trajectory=None):
        """
        返回各项损失明细，用于分析和调试
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)
        B, C, T, H, W = video.shape
        if trajectory is not None and trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)

        with torch.no_grad():
            # 计算速度场
            velocity = self.physics.compute_velocity_field(video)
            # 各项损失
            ns_loss = self.physics.navier_stokes_loss(velocity)
            rigid_loss = self.physics.rigid_body_loss(
                trajectory) if trajectory is not None else torch.tensor(0.0)
            collision_loss = self.physics.collision_detection_loss(
                velocity, video)
            elastic_loss = self.physics.elastic_potential_loss(video)

        return {
            'navier_stokes': ns_loss.item(),
            'rigid_body': rigid_loss.item(),
            'collision': collision_loss.item(),
            'elastic': elastic_loss.item(),
            'total': ns_loss.item() + 0.1*rigid_loss.item() + 0.05*collision_loss.item() + 0.01*elastic_loss.item()
        }
