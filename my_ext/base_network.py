from typing import Optional
from copy import deepcopy

import torch
from torch import nn, Tensor


class BaseNetwork(nn.Module):
    def __init__(self, model: nn.Module, criterion: nn.Module, swa_enable=False):
        super().__init__()
        self.model = model
        self.criterion = criterion

        self.swa_enable = swa_enable
        if self.swa_enable:
            self.averaged_model = deepcopy(model)  # type: Optional[nn.Module]
            self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long))
            for p in self.averaged_model.parameters():
                p.requires_grad_(False)
        else:
            self.averaged_model = None  # type: Optional[nn.Module]
            self.register_buffer('n_averaged', None)

    def train(self, mode=True):
        self.training = mode
        self.model.train(mode)
        self.criterion.train(mode)
        if self.swa_enable:
            self.averaged_model.train(False)

    def __getattr__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            return getattr(self.model, item)
