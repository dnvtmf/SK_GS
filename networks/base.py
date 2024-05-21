from typing import Type, Dict, Optional
import torch
from torch import nn, Tensor

from my_ext import Registry
import logging

__all__ = ['NeRF_Network', 'NERF_NETWORKS']
NERF_NETWORKS = Registry()  # type: Registry[Type[NeRF_Network]]


class NeRF_Network(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logging.warning(f"{self.__class__.__name__} unused parameters: {list(kwargs.keys())}")

    def set_from_dataset(self, dataset):
        pass

    def render(self, rays_o: Tensor, rays_d: Tensor, background: Tensor = None, **kwargs):
        return NotImplemented

    def loss(self, inputs, outputs, targets):
        # type: (Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]) -> Dict[str, Tensor]
        raise NotImplementedError
