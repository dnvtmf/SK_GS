from typing import Union, Any, Sequence
import math
from copy import deepcopy

import torch
from torch import Tensor

from my_ext.utils import n_tuple


class Structure3D:
    def __init__(self, info=None):
        self.infos = {} if info is None else info

    def add_info(self, **kwargs):
        self.infos.update(kwargs)
        return self

    def get_info(self, key=None):
        return self.infos[key] if key is not None else self.infos

    def pop_info(self, key=None):
        return self.infos.popitem() if key is None else self.infos.pop(key)

    def get_info_with_default(self, key, default=None):
        return self.infos.setdefault(key, default)

    def has_info(self, key):
        return key in self.infos

    def __len__(self):
        return NotImplemented

    def __getitem__(self, item):
        return NotImplemented

    def clone(self):
        return deepcopy(self)

    def pan_zoom_(self, scaling=1., offset=0.0, *args, **kwargs):
        # type: (Union[float, Sequence[float]], Union[float, Sequence[float]], Any, Any) -> Structure3D
        """First move the object by given offset, then scaling the object"""
        raise NotImplementedError

    def pan_zoom(self, scaling=1., offset=0.0, *args, **kwargs):
        # type: (Union[float, Sequence[float]], Union[float, Sequence[float]], Any, Any) -> Structure3D
        """First move the object by given offset, then scaling the object"""
        return self.clone().pan_zoom_(scaling=scaling, offset=offset, *args, **kwargs)

    def flip_(self, x_axis=False, y_axis=False, z_axis=False, center=(0., 0., 0.), *args, **kwargs):
        return NotImplemented

    def flip(self, x_axis=False, y_axis=False, z_axis=False, center=(0., 0., 0.), *args, **kwargs):
        return self.clone().flip_(x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, center=center, *args, **kwargs)

    def crop_(self, x: int, y: int, w: int, h: int, clamp=True, *args, **kwargs):
        return NotImplemented

    def crop(self, x: int, y: int, w: int, h: int, clamp=True, *args, **kwargs):
        return self.clone().crop_(x, y, w, h, clamp=clamp, *args, **kwargs)

    def pad_(self, padding, *args, **kwargs):
        return NotImplemented

    def pad(self, padding, *args, **kwargs):
        return self.clone().pad_(padding, *args, **kwargs)

    @staticmethod
    def get_affine_matrix_3d(
        rotation=0.,
        scaling=1.,
        translation=0.,
        center=(0, 0, 0),
        flip=False,
        **kwargs
    ) -> Tensor:
        """
        适用于右乘的变换矩阵

        1. move the origin to center
        2. rotate <angle> degrees
        3. shear  <--not implemented-->
        4. scale
        5. flip the axis
        6. move the origin back and translate
        """
        scaling = n_tuple(scaling, 3)
        translation = n_tuple(translation, 3)
        rotation = n_tuple(rotation, 3)
        flip = n_tuple(flip, 3)
        T = torch.eye(4, dtype=torch.float)
        ## step-1 move points to center
        T[3, 0] = -center[0]
        T[3, 1] = -center[1]
        T[3, 2] = -center[2]
        ## step-2 rotate around (0, 0, 0)
        # rotate around x-axis, i.e., change yz
        if rotation[0] != 0:
            rot_sin = math.sin(rotation[0])
            rot_cos = math.cos(rotation[0])
            rot_mat_x = torch.tensor([[1, 0, 0, 0], [0, rot_cos, -rot_sin, 0], [0, rot_sin, rot_cos, 0], [0, 0, 0, 1]])
            T = T @ rot_mat_x
        # rotate around y-axis, i.e., change xz
        if rotation[1] != 0:
            rot_sin = math.sin(rotation[1])
            rot_cos = math.cos(rotation[1])
            rot_mat_y = torch.tensor([[rot_cos, 0, -rot_sin, 0], [0, 1, 0, 0], [rot_sin, 0, rot_cos, 0], [0, 0, 0, 1]])
            T = T @ rot_mat_y
        # rotate around z-axis, i.e., change xy
        if rotation[2] != 0:
            rot_sin = math.sin(rotation[2])
            rot_cos = math.cos(rotation[2])
            rot_mat_z = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T = T @ rot_mat_z
        ## step-3 shear
        pass
        ## step-4: scale the points and step-5: flip
        T = T @ T.new_tensor([[-scaling[0] if flip[0] else scaling[0], 0, 0, 0],
                              [0, -scaling[1] if flip[1] else scaling[1], 0, 0],
                              [0, 0, -scaling[2] if flip[2] else scaling[2], 0],
                              [0, 0, 0, 1]])
        # T[:, 0] *= -scaling[0] if flip[0] else scaling[0]
        # T[:, 1] *= -scaling[1] if flip[1] else scaling[1]
        # T[:, 2] *= -scaling[2] if flip[2] else scaling[2]
        ## step-6: move the center back, and apply translate
        T = T @ T.new_tensor([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [translation[0] + center[0], translation[1] + center[1], translation[2] + center[2], 1]])
        return T

    def affine_transform_(self, affine_matrix, *args, **kwargs):
        raise NotImplementedError

    def affine_transform(self, affine_matrix, *args, **kwargs):
        return self.clone().affine_transform_(affine_matrix, *args, **kwargs)
