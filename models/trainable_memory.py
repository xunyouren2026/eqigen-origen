# models/trainable_memory.py
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LightweightMemory(nn.Module):
    def __init__(self, num_slots: int, dim: int, rank: int = 16,
                 top_k: int = 8, controller_type: str = 'gru'):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.rank = rank
        self.top_k = top_k
        self.controller_type = controller_type

        self.base_memory = nn.Parameter(torch.randn(num_slots, dim) * 0.01)
        self.lora_A = nn.Parameter(torch.randn(num_slots, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, dim) * 0.01)

        if controller_type == 'gru':
            self.controller = nn.GRUCell(dim, dim)
        else:
            self.controller = nn.LSTMCell(dim, dim)

        self.write_gate = nn.Linear(dim, 1)
        self.erase_gate = nn.Linear(dim, 1)

    @property
    def memory(self):
        return self.base_memory + torch.matmul(self.lora_A, self.lora_B)

    def read(self, query: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = top_k if top_k is not None else self.top_k
        mem = self.memory
        q = query.mean(dim=1, keepdim=True)  # (B,1,D)
        scores = torch.matmul(q, mem.T)      # (B,1,num_slots)
        top_scores, top_indices = torch.topk(scores.squeeze(1), k, dim=-1)
        retrieved = torch.gather(mem.unsqueeze(0).expand(query.size(0), -1, -1),
                                 dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, self.dim))
        return retrieved, top_scores, top_indices

    def write(self, hidden: torch.Tensor, value: torch.Tensor, indices: Optional[torch.Tensor] = None):
        if indices is None:
            indices = torch.arange(self.num_slots, device=hidden.device).unsqueeze(
                0).expand(hidden.size(0), -1)

        write_prob = torch.sigmoid(self.write_gate(hidden))   # (B,1)
        erase_prob = torch.sigmoid(self.erase_gate(hidden))   # (B,1)

        for b in range(hidden.size(0)):
            for i in range(indices.size(1)):
                idx = indices[b, i].item()
                self.base_memory[idx] = (
                    1 - write_prob[b, 0]) * self.base_memory[idx] + write_prob[b, 0] * value[b, 0]

    def forward(self, query: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        retrieved, _, idx = self.read(query)
        if hidden is None:
            hidden = torch.zeros_like(query[:, 0])
        if self.controller_type == 'gru':
            hidden = self.controller(retrieved.sum(dim=1), hidden)
        else:
            h, c = hidden
            h, c = self.controller(retrieved.sum(dim=1), (h, c))
            hidden = (h, c)
        self.write(hidden, query.mean(dim=1, keepdim=True), indices=idx)
        return retrieved, hidden

    def reset(self):
        self.base_memory.data = torch.randn_like(self.base_memory) * 0.01
        self.lora_A.data = torch.randn_like(self.lora_A) * 0.01
        self.lora_B.data = torch.randn_like(self.lora_B) * 0.01
