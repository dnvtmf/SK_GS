from typing import Tuple
import copy
import math

import numpy as np


class Structure2D:
    size: Tuple[int, int]

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
        return copy.deepcopy(self)

    def resize_(self, size: (int, int), *args, **kwargs):
        return NotImplemented

    def resize(self, size: (int, int), *args, **kwargs):
        return self.clone().resize_(size, *args, **kwargs)

    def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
        return NotImplemented

    def flip(self, horizontal=False, vertical=False, *args, **kwargs):
        return self.clone().flip_(horizontal=horizontal, vertical=vertical, *args, **kwargs)

    def crop_(self, x: int, y: int, w: int, h: int, clamp=True, *args, **kwargs):
        return NotImplemented

    def crop(self, x: int, y: int, w: int, h: int, clamp=True, *args, **kwargs):
        return self.clone().crop_(x, y, w, h, clamp=clamp, *args, **kwargs)

    def _get_padding_param(self, padding):
        """
        padding: int: left_pad = top_pad = right_pad = bottom_pad
                (int, int): left_pad = right_pad, top_pad = bottom_pad

                (int, int, int, int): left_pad, top_pad, right_pad, bottom_pad
        :return: (int, int, int, int)
        """
        if isinstance(padding, (int, float, bool, str)):
            left_pad = top_pad = right_pad = bottom_pad = int(padding)
        else:
            left_pad = int(padding[0])
            top_pad = int(padding[1])
            if len(padding) == 2:
                right_pad = left_pad
                bottom_pad = top_pad
            elif len(padding) == 4:
                right_pad = int(padding[2])
                bottom_pad = int(padding[3])
            else:
                raise ValueError(f"Padding must be an int or a 2, or 4 element tuple, not {len(padding)}")
        return left_pad, top_pad, right_pad, bottom_pad

    def pad_(self, padding, *args, **kwargs):
        return NotImplemented

    def pad(self, padding, *args, **kwargs):
        return self.clone().pad_(padding, *args, **kwargs)

    def to(self, device):
        return NotImplemented

    def clip_to_image(self):
        return NotImplemented

    def get_affine_matrix(self, angle=(0., 0.), translate=(0., 0.), scale=(1., 1.), shear=(0., 0.), center=None,
                          output_size=None):
        """
        center is default as the center of images
        1. move the origin to center
        2. rotate the image <angle> degrees
        3. shear the image
        4. scale
        5. move the origin back when output size is None, or move the center to the output center
        6. translate
        """
        if center is None:
            cx, cy = (self.size[0] * 0.5, self.size[1] * 0.5)
        else:
            cx, cy = center
        R = np.eye(3)
        angle = math.radians(angle)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        sx, sy = math.tan(math.radians(shear[0])), math.tan(math.radians(shear[1]))
        a, b = scale[0] * (cos_a - sin_a * sy), scale[1] * (cos_a * sx - sin_a)
        c, d = scale[0] * (sin_a + cos_a * sy), scale[1] * (sin_a * sx + cos_a)
        offset_x, offset_y = (cx, cy) if output_size is None else (output_size[0] // 2, output_size[1] // 2)
        R[0] = [a, b, offset_x + translate[0] - a * cx - b * cy]
        R[1] = [c, d, offset_y + translate[1] - c * cx - d * cy]
        return R

    def affine(self, affine_matrix, output_size, **kwargs):
        return NotImplemented

    def draw(self, img: np.ndarray = None, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
