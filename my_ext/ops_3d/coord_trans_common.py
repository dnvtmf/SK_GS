from typing import Union, Sequence
import math

import torch
import numpy as np
from torch import Tensor


def _rotate(angle: Union[float, Tensor], device=None, a=0, b=1) -> Tensor:
    if isinstance(angle, Tensor):
        s, c = torch.sin(angle), torch.cos(angle)
        T = torch.eye(4, dtype=s.dtype, device=s.device if device is None else device)
        T = T.expand(list(s.shape) + [4, 4]).contiguous()
        T[..., a, a] = c
        T[..., a, b] = -s
        T[..., b, a] = s
        T[..., b, b] = c
    else:
        s, c = math.sin(angle), math.cos(angle)
        T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        T[a][a] = c
        T[b][b] = c
        T[a][b] = -s
        T[b][a] = s
        T = torch.tensor(T, device=device)
    return T


def rotate_x(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 1, 2)


def rotate_y(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 0, 2)


def rotate_z(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 0, 1)


def rotate(x: float = None, y: float = None, z: float = None, device=None):
    R = torch.eye(4, device=device)
    if x is not None:
        R = R @ rotate_x(x, device)
    if y is not None:
        R = R @ rotate_y(y, device)
    if z is not None:
        R = R @ rotate_z(z, device)
    return R


def scale(s: float, device=None):
    return torch.tensor([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)


def fovx_to_fovy(fovx, aspect=1.) -> Union[np.ndarray, Tensor]:
    if isinstance(fovx, Tensor):
        return torch.arctan(torch.tan(fovx * 0.5) / aspect) * 2.0
    else:
        return np.arctan(np.tan(fovx * 0.5) / aspect) * 2.0


def focal_to_fov(focal: Union[float, Tensor, np.ndarray], *size: Union[float, Sequence[float]]):
    """focal length of fov"""
    if len(size) == 1:
        size = size[0]
    if isinstance(size, Sequence):
        if isinstance(focal, Tensor):
            return torch.stack([2 * torch.arctan(0.5 * s / focal) for s in size], dim=-1)
        else:
            return np.stack([2 * np.arctan(0.5 * s / focal) for s in size], axis=-1)
    else:
        t = 0.5 * size / focal
        return 2 * (torch.arctan(t) if isinstance(t, Tensor) else np.arctan(t))


def fov_to_focal(fov: Union[float, Tensor, np.ndarray], size: Union[float, Tensor, np.ndarray]):
    """FoV to focal length"""
    return size / (2 * (torch.tan(fov * 0.5) if isinstance(fov, Tensor) else np.tan(fov * 0.5)))
