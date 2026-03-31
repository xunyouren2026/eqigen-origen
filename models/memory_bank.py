# models/memory_bank.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class MemoryBank(nn.Module):
    """
    长期记忆库模块
    功能：
    1. 存储历史视频片段的关键特征 (Key-Value 机制)
    2. 支持滑动更新 (FIFO + 重要性加权)
    3. 支持多尺度检索 (短期/中期/长期)
    4. 预留对比学习接口 (Contrastive Learning)

    优化配置：
    - max_size: 默认提升至 1024，支持更长视频的一致性
    """

    def __init__(self, dim: int, max_size: int = 1024, update_alpha: float = 0.9,
                 use_learned_retrieval: bool = False,
                 use_importance_pred: bool = False,
                 use_contrastive: bool = False,
                 contrastive_temp: float = 0.07):
        """
        :param dim: 特征维度 (通常等于 model.dit_context_dim)
        :param max_size: 记忆库最大容量 (默认 1024)
        :param update_alpha: 更新时的动量系数 (0.9 表示新记忆占 10%)
        :param use_contrastive: 是否启用对比学习损失计算
        """
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        self.update_alpha = update_alpha
        self.use_learned_retrieval = use_learned_retrieval
        self.use_importance_pred = use_importance_pred
        self.use_contrastive = use_contrastive
        self.contrastive_temp = contrastive_temp

        # 注册缓冲区 (不随梯度更新，但随模型保存)
        # memory: (1, max_size, dim)
        self.register_buffer('memory', torch.zeros(1, max_size, dim))
        # age: 记录每个槽位的“年龄”，用于淘汰最旧的记忆
        self.register_buffer('age', torch.zeros(1, max_size))
        # importance: 记录每个槽位的重要性评分
        self.register_buffer('importance', torch.zeros(1, max_size))

        self.step = 0

        # 可选：可学习检索头
        if use_learned_retrieval:
            self.retrieval_mlp = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, 1)
            )

        # 可选：重要性预测头
        if use_importance_pred:
            self.importance_mlp = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()
            )

        # 可选：对比学习投影头
        if use_contrastive:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, dim)
            )

    def update(self, features: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """
        更新记忆库
        :param features: 当前帧/块的特征 (B, N, D)
        :param importance: 外部传入的重要性权重 (可选)
        """
        B, N, D = features.shape
        # 聚合当前特征 (取平均作为新记忆候选)
        new_mem = features.mean(dim=1, keepdim=True)  # (B, 1, D)

        # 计算重要性
        if importance is None and self.use_importance_pred:
            imp = self.importance_mlp(new_mem).squeeze(-1)  # (B, 1)
        else:
            imp = importance.mean(dim=1, keepdim=True) if importance is not None else torch.ones(
                B, 1, device=features.device)

        # 寻找最旧且最不重要的槽位进行替换
        # 策略：Age / Importance 越大，越容易被替换
        weighted_age = self.age[0] / (self.importance[0] + 1e-6)
        oldest_idx = weighted_age.argmax().item()

        # 如果预测重要性极低，可选择跳过更新 (节省计算，可选)
        if self.use_importance_pred and imp.mean().item() < 0.2:
            self.age += 1  # 仍然增加其他槽位的年龄
            return

        # 执行滑动更新：New = Alpha * Old + (1-Alpha) * Candidate
        self.memory[0, oldest_idx] = (
            self.update_alpha * self.memory[0, oldest_idx] +
            (1 - self.update_alpha) * new_mem.squeeze(0)
        )

        # 重置被更新槽位的年龄和重要性
        self.age[0, oldest_idx] = 0
        self.importance[0, oldest_idx] = imp.squeeze(
        ).item() if torch.is_tensor(imp) else imp

        # 其他槽位年龄 +1
        self.age += 1
        self.step += 1

    def retrieve(self, query: torch.Tensor, top_k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索记忆
        :param query: 查询特征 (B, N, D)
        :param top_k: 返回前 K 个相关记忆
        :return: (retrieved_features, similarity_scores)
        """
        B, N, D = query.shape
        q = query.mean(dim=1, keepdim=True)  # (B, 1, D)
        mem = self.memory.expand(B, -1, -1)   # (B, max_size, D)

        if self.use_learned_retrieval:
            q_expand = q.expand(-1, self.max_size, -1)
            combined = torch.cat([q_expand, mem], dim=-1)
            sim = self.retrieval_mlp(combined).squeeze(-1)
        else:
            # 默认使用余弦相似度
            sim = F.cosine_similarity(q, mem, dim=-1)

        top_sim, top_idx = sim.topk(min(top_k, self.max_size), dim=-1)

        #  Gather 对应的记忆向量
        retrieved = torch.gather(
            mem, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))

        return retrieved, top_sim

    def contrastive_loss(self, query: torch.Tensor, positive: torch.Tensor,
                         negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算对比学习损失 (InfoNCE Loss)
        用于训练时拉近同一视频不同片段的记忆特征
        :param query: 锚点特征 (B, N, D)
        :param positive: 正样本特征 (B, N, D)
        :param negatives: 负样本特征 (B, M, D) (可选，若不提供则使用记忆库中的其他项)
        """
        if not self.use_contrastive or negatives is None:
            return torch.tensor(0.0, device=query.device)

        # 投影到对比学习空间
        q_proj = self.contrastive_proj(query.mean(dim=1))
        p_proj = self.contrastive_proj(positive.mean(dim=1))
        n_proj = self.contrastive_proj(negatives.mean(dim=1))

        # 计算相似度
        pos_sim = torch.cosine_similarity(
            q_proj, p_proj, dim=-1) / self.contrastive_temp

        # 负样本相似度矩阵 (B, Num_Negatives)
        neg_sims = torch.cosine_similarity(
            q_proj.unsqueeze(1), n_proj.unsqueeze(0), dim=-1) / self.contrastive_temp

        # 构建 Logits 和 Labels
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(logits.size(
            0), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits, labels)

    def reset(self):
        """清空记忆库"""
        self.memory.zero_()
        self.age.zero_()
        self.importance.zero_()
        self.step = 0
        print("[MemoryBank] Memory reset.")
