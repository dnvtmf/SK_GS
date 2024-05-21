#!/usr/bin/env python3
from typing import Iterable

from my_ext.structures import Structure2D, Structure3D


class Restoration(Structure2D):
    def __init__(self, size, backward=None, infos=None):
        super().__init__(infos)
        self.size = size
        self.backward = [] if backward is None else backward

    def get_transforms(self):
        return list(reversed(self.backward))

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, item):
        return self.infos[item]

    def clone(self):
        return Restoration(self.size, self.backward.copy(), self.infos.copy())

    def resize_(self, size: (int, int), *args, **kwargs):
        self.backward.append(('resize_', self.size, kwargs))
        self.size = size
        return self

    def resize(self, size: (int, int), *args, **kwargs):
        return self.clone().resize_(size, *args, **kwargs)

    def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
        self.backward.append(('flip_', horizontal, vertical, kwargs))
        return self

    def flip(self, horizontal=False, vertical=False, *args, **kwargs):
        return self.clone().flip_(horizontal, vertical, *args, **kwargs)

    def crop_(self, x: int, y: int, w: int, h: int, *args, **kwargs):
        self.backward.append(('pad_', (x, y, self.size[0] - w - x, self.size[1] - h - y), kwargs))
        self.size = (w, h)
        return self

    def crop(self, x: int, y: int, w: int, h: int, *args, **kwargs):
        return self.clone().crop_(x, y, w, h, *args, **kwargs)

    def pad_(self, padding, *args, **kwargs):
        padding = self._get_padding_param(padding)
        self.backward.append(('crop_', padding[0], padding[1], self.size[0], self.size[1], kwargs))
        self.size = (self.size[0] + padding[0] + padding[2], self.size[1] + padding[1] + padding[3])
        return self

    def pad(self, padding, *args, **kwargs):
        return self.clone().pad_(padding, *args, **kwargs)

    def to(self, device):
        pass

    def affine(self, affine_matrix, output_size, **kwargs):
        return NotImplemented

    def __repr__(self):
        return f"Restoration(size={self.size}, info={self.infos})"

    def apply(self, other: Structure2D):
        for args in self.get_transforms():
            name = args[0]  # type: str
            kwargs = args[-1]  # type: dict
            args = args[1:-1]
            if hasattr(other, name):
                other = getattr(other, name)(*args, **kwargs)
            else:
                print(type(other), 'No such attribute', name)
        return other

    def __call__(self, other: Structure2D):
        return self.apply(other)

    def draw(self, img=None, *args, **kwargs):
        return img


class Restoration3D(Structure3D):
    def __init__(self, backward=None, infos=None):
        super().__init__(infos)
        self.backward = [] if backward is None else backward

    def get_transforms(self):
        return list(reversed(self.backward))

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, item):
        return self.infos[item]

    def clone(self):
        return Restoration(self.backward.copy(), self.infos.copy())

    def pan_zoom_(self, scaling=1., offset=0.0, *args, **kwargs):
        if isinstance(offset, Iterable):
            if isinstance(scaling, Iterable):
                offset = [-x * s for x, s in zip(offset, scaling)]
            else:
                offset = [-x * scaling for x in offset]
        else:
            if isinstance(scaling, Iterable):
                offset = [-offset * s for s in scaling]
            else:
                offset = -offset * scaling
        scaling = [1. / s for s in scaling] if isinstance(scaling, Iterable) else scaling
        self.backward.append(('pan_zoom_', scaling, offset, kwargs))
        return self

    # def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
    #     self.backward.append(('flip_', horizontal, vertical, kwargs))
    #     return self
    #
    # def crop_(self, x: int, y: int, w: int, h: int, *args, **kwargs):
    #     self.backward.append(('pad_', (x, y, self.size[0] - w - x, self.size[1] - h - y), kwargs))
    #     self.size = (w, h)
    #     return self
    #
    # def pad_(self, padding, *args, **kwargs):
    #     self.backward.append(('crop_', padding[0], padding[1], self.size[0], self.size[1], kwargs))
    #     return self

    def to(self, device):
        pass

    def affine(self, affine_matrix, output_size, **kwargs):
        return NotImplemented

    def __repr__(self):
        return f"Restoration(info={self.infos})"

    def apply(self, other: Structure3D):
        other = other.clone()
        for args in self.get_transforms():
            name = args[0]  # type: str
            kwargs = args[-1]  # type: dict
            args = args[1:-1]
            if hasattr(other, name):
                other = getattr(other, name)(*args, **kwargs)
            else:
                print(type(other), 'No such attribute', name)
        return other

    def __call__(self, other: Structure3D):
        return self.apply(other)

    def draw(self, img=None, *args, **kwargs):
        return img
