import torch

from my_ext.utils.registry import Registry

OPTIMIZERS = Registry(ignore_case=True)

OPTIMIZERS.register('sgd', torch.optim.SGD)
OPTIMIZERS.register('adam', torch.optim.Adam)
OPTIMIZERS.register('adamax', torch.optim.Adamax)
OPTIMIZERS.register('RMSprop', torch.optim.RMSprop)
OPTIMIZERS.register('adamw', torch.optim.AdamW)

from . import RAdam, ranger, Adan

from .build import make, options, freeze_modules

__all__ = ['OPTIMIZERS', 'make', 'options']
