from typing import Union, Tuple

import torch
from torch import nn, Tensor

from my_ext import utils
from ..misc import normalize
from .base import Light


class PointLight(Light):

    def __init__(
        self,
        ambient_color: Union[float, Tuple[float, float, float], Tensor] = (1.0, 1.0, 1.0),
        diffuse_color: Union[float, Tuple[float, float, float], Tensor] = (1.0, 1.0, 1.0),
        specular_color: Union[float, Tuple[float, float, float], Tensor] = (1.0, 1.0, 1.0),
        location: Union[float, Tuple[float, float, float], Tensor] = (0, 1., 0),
        device=None,
        trainable=False
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch dimension N.
        """
        super().__init__()
        if isinstance(ambient_color, Tensor):
            ambient_color = ambient_color.to(device)
        else:
            ambient_color = torch.tensor(utils.n_tuple(ambient_color, 3), dtype=torch.float, device=device)
        if trainable:
            self.ambient_color = nn.Parameter(ambient_color)
        else:
            self.register_buffer('ambient_color', ambient_color)

        if isinstance(diffuse_color, Tensor):
            diffuse_color = diffuse_color.to(device)
        else:
            diffuse_color = torch.tensor(utils.n_tuple(diffuse_color, 3), dtype=torch.float, device=device)
        if trainable:
            self.diffuse_color = nn.Parameter(diffuse_color)
        else:
            self.register_buffer('diffuse_color', diffuse_color)

        if isinstance(specular_color, Tensor):
            specular_color = specular_color.to(device)
        else:
            specular_color = torch.tensor(utils.n_tuple(specular_color, 3), dtype=torch.float, device=device)
        if trainable:
            self.specular_color = nn.Parameter(specular_color)
        else:
            self.register_buffer('specular_color', specular_color)

        if isinstance(location, Tensor):
            location = location.to(device)
        else:
            location = torch.tensor(utils.n_tuple(location, 3), dtype=torch.float, device=device)
        self.register_buffer('location', location)
        self.location: Tensor

    def forward(self, points: Tensor = None):
        direction = normalize(self.location - points)
        return direction, self.ambient_color, self.diffuse_color, self.specular_color
