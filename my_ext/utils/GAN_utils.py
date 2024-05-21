from typing import Union

from torch import nn
from torch.optim import Optimizer

__all__ = ['toggle_grad']


def toggle_grad(net_or_opt: Union[nn.Module, Optimizer], requires_grad=True):
    if isinstance(net_or_opt, nn.Module):
        for p in net_or_opt.parameters():
            p.requires_grad_(requires_grad)
    else:
        for param_group in net_or_opt.param_groups:
            for p in param_group['params']:
                p.requires_grad = requires_grad
    return
