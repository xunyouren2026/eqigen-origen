import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register_buffer('step', torch.tensor(0))
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1 - decay) * param.data + \
                    decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()
