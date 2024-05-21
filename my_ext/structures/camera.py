from typing import Union, Sequence, Any, Tuple

import numpy as np
import torch
from torch import Tensor

from .base_3D import Structure3D
from .base_2D import Structure2D
from my_ext.utils import geometry, to_tensor


class CameraIntrinsics(Structure2D):
    """相机内参 (齐次）
    支持多组
    """

    def __init__(self, f, c, size=(1, 1), infos=None):
        # type: (Union[np.ndarray, Tensor, Sequence], Union[np.ndarray, Tensor, Sequence], Sequence, Any)->None
        """

        Args:
            f: (fx, fy) fx=1/dx, fy=1/dy
            c: (cx, cy)
            size: 图像大小 (H, W)
            infos: 其他信息
        """
        super().__init__(infos)
        self.f = to_tensor(f)  # type: Tensor
        self.c = to_tensor(c)  # type: Tensor
        assert self.f.shape == self.c.shape and self.f.shape[-1] == 2
        self.size = tuple(size)

    @classmethod
    def from_K(cls, K: Union[np.ndarray, Tensor], size=(1, 1), *args, **kwargs):
        if isinstance(K, np.ndarray):
            f = np.stack([K[..., 0, 0], K[..., 1, 1]], axis=-1)
        else:
            f = torch.stack([K[..., 0, 0], K[..., 1, 1]], dim=-1)
        return cls(f, K[..., :2, 2], size, *args, **kwargs)

    @property
    def K(self) -> Tensor:
        K_ = self.f.new_zeros((*self.f.shape[:-2], 3, 3))
        K_[..., 0, 0] = self.f[..., 0]
        K_[..., 1, 1] = self.f[..., 1]
        K_[..., :2, 2] = self.c
        # K_[0, 2] = self.c[0]
        # K_[1, 2] = self.c[1]
        K_[..., 2, 2] = 1.
        return K_

    @property
    def K_homo(self) -> Tensor:
        K_ = self.f.new_zeros((*self.f.shape[:-2], 3, 4))
        K_[..., 0, 0] = self.f[..., 0]
        K_[..., 1, 1] = self.f[..., 1]
        K_[..., :2, 2] = self.c
        K_[..., 2, 2] = 1.
        return K_

    def clone(self):
        return CameraIntrinsics(self.f.clone(), self.c.clone(), self.size, self.infos)

    def cuda(self):
        self.f = self.f.cuda()
        self.c = self.c.cuda()
        for k, v in self.infos.items():
            if hasattr(v, 'cuda'):
                self.infos[k] = v.cuda()
        return self

    def to(self, *args, **kwargs):
        self.f = self.f.to(*args, **kwargs)
        self.c = self.c.to(*args, **kwargs)
        for k, v in self.infos.items():
            if hasattr(v, 'to'):
                self.infos[k] = v.to(*args, **kwargs)
        return self

    def crop_(self, x: int, y: int, w: int, h: int, *args, **kwargs):
        self.c[..., 0] -= x
        self.c[..., 1] -= y
        self.size = (w, h)
        return self

    def resize_(self, size: Tuple[int, int], *args, **kwargs):
        scale = self.f.new_tensor([size[0] / self.size[0], size[1] / self.size[1]])
        self.c *= scale
        self.f *= scale
        self.size = size
        return self

    def pad_(self, padding, *args, **kwargs):
        self.c[..., 0] += padding[0]
        self.c[..., 1] += padding[1]
        self.size = (self.size[0] + padding[0] + padding[2], self.size[1] + padding[1] + padding[3])
        return self

    def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
        if vertical:
            self.f[..., 1] = -self.f[..., 1]
            self.c[..., 1] = self.size[1] - self.c[..., 1]
        if horizontal:
            self.f[..., 0] = -self.f[..., 0]
            self.c[..., 0] = self.size[0] - self.c[..., 0]
        return self

    def camera_to_pixel(self, points: Tensor):
        """ convert 3D points to 2D pixel coordinates
        Args:
            points: shape: (..., 3) or (..., 4)
        """
        assert points.shape[-1] == 3 or points.shape[-1] == 4, f'Error shape of points: {points.shape}'
        return torch.addcmul(self.c, points[..., :2] / points[..., 2:3], self.f)

    def pixel_to_camera(self, uv: Tensor, depth: Tensor = None, dim: int = None):
        """ convert 2D pixel coordinates to 3D points
        Args:
            uv: The corrdinates of 2D pixels. shape: (..., 2)
            depth: The value of z axis. shape: (..., 1) or (...)
            dim: e.g., uv shape: (B, N, 2), camera_c: (B, 2), ==> dim=-2 or dim=1
        """
        if dim is None:
            uv = (uv - self.c) / self.f
        else:
            uv = (uv - self.c.unsqueeze(dim)) / self.f.unsqueeze(dim)
        points = torch.cat([uv, uv.new_ones(*uv.shape[:-1], 1)], dim=-1)  # shape: (..., 3)
        if depth is not None:
            if depth.ndim == points.ndim:
                points *= depth
            elif depth.ndim == points.ndim - 1:
                points *= depth[..., None]
            else:
                raise RuntimeError(f"Error shape of depth")
        return points

    @classmethod
    def make_one(cls, cameras):
        f = torch.stack([c.f for c in cameras], dim=0)
        c = torch.stack([c.c for c in cameras], dim=0)
        for i in range(1, len(cameras)):
            assert cameras[i].size == cameras[0].size
        return cls(f, c, cameras[0].size)

    def split(self, dim=0):
        assert self.f.ndim > 1
        return [CameraIntrinsics(f, c, self.size) for f, c in zip(self.f.unbind(dim=dim), self.c.unbind(dim=dim))]

    def __getitem__(self, item):
        return CameraIntrinsics(self.f[item], self.c[item], self.size, self.infos)

    def __repr__(self):
        s = f"{self.__class__.__name__}(size={self.size}, "
        if self.f.ndim == 1:
            s += f"f=({self.f[0].item():.3f}, {self.f[1].item():.3f}), " \
                 f"c=({self.c[0].item():.3f}, {self.c[1].item():.3f})"
        else:
            s += f"batch_shape={list(self.f.shape[:-1])})"
        return s


class CameraExtrinsics(Structure3D):
    """相机外参"""

    def __init__(self, T: Tensor, infos=None):
        """

        Args:
            T: 齐次的相机外参(左乘)，shape: (4, 4), i.e., P_c = T @ P_w
            infos:
        """
        super().__init__(infos)
        self.T = T
        assert self.T.shape == (4, 4), self.T.shape

    @property
    def R(self) -> Tensor:
        return self.T[:3, :3]

    @property
    def t(self) -> Tensor:
        return self.T[:3, 3]

    def cuda(self):
        self.T = self.T.cuda()
        return self

    def to(self, *args, **kwargs):
        self.T = self.T.to(*args, **kwargs)
        return self

    @classmethod
    def from_quanernion(cls, quanernion: Union[np.ndarray, Sequence[float], Tensor], t=None, infos=None):
        """

        Args:
            quanernion: 表示旋转的四元数(w, x, y, z)
            t: 平移向量, shape: 3
            infos:

        Returns:

        """
        w, x, y, z = quanernion
        R = geometry.quanernion_to_matrix(w, x, y, z)
        return cls.from_matrix(R, t, infos)

    @classmethod
    def from_matrix(cls, rotation, translation=None, infos=None):
        """

        Args:
            rotation: 旋转矩阵，shape: (3, 3)
            translation: 平移向量, shape: 3
            infos:

        Returns:

        """
        rotation = to_tensor(rotation)
        T = rotation.new_zeros(4, 4)
        T[:3, :3] = rotation
        if translation is not None:
            T[:3, 3] = to_tensor(translation)
        T[3, 3] = 1
        return cls(T, infos)

    def affine_(self, scale, shift):
        self.T = self.T @ self.T.new_tensor([[scale, 0, 0, shift[0]],
                                             [0, scale, 0, shift[1]],
                                             [0, 0, scale, shift[2]],
                                             [0, 0, 0, 1]]).T
        return self

    def camera_to_world(self, points: Tensor) -> Tensor:
        """ convert camera coordinates to world coordinates

        Args:
            points: shape: (..., 3) or (..., 4)
        Return:
            world coordinates
        """
        if points.shape[-1] == 3:
            return (points - self.T[:3, 3]) @ self.T[:3, :3]
        elif points.shape[-1] == 4:
            return (points[..., :3] - self.T[:3, 3]) @ self.T[:3, :3]
            # return points @ self.T.T.inverse()
        else:
            raise ValueError(f'Error shape for points: {points.shape}')

    def world_to_camera(self, points: Tensor):
        """ convert world coordinates to camera coordinates

        Args:
            points: shape: (..., 3) or (..., 4)
        Return:
            camera coordinates
        """
        if points.shape[-1] == 3:
            return points @ self.T[:3, :3].T + self.T[:3, 3]
        elif points.shape[-1] == 4:
            return points @ self.T.T
        else:
            raise ValueError(f'Error shape for points: {points.shape}')

    def __repr__(self):
        return f"{self.__class__.__name__}()"
