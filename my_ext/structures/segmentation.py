import cv2
import numpy as np
import torch

from my_ext.structures import Structure2D, Restoration
from my_ext.utils import get_colors


class Segmentation(Structure2D):
    IgnoreIndex = 255

    def __init__(self, mask, image_size=None, infos=None):
        super().__init__(infos)
        if mask is None:
            assert image_size is not None
            self.mask = np.zeros(image_size[1], image_size[0])
        else:
            if isinstance(mask, np.ndarray):
                self.mask = mask
            elif isinstance(mask, torch.Tensor):
                self.mask = mask.detach().cpu().numpy()  # type: np.ndarray
            else:
                raise TypeError(f'Unsupported type {type(mask)} for Segmentation')
            if image_size is not None:
                assert self.mask.shape == (image_size[1], image_size[0]), \
                    f'Error shape: (w, h): {(mask.shape[1], mask.shape[0])} != {image_size}'
        self.mask: np.ndarray

    @staticmethod
    def load_from_image(img_path):
        mask = cv2.imread(str(img_path), -1)
        assert mask is not None and mask.ndim == 2
        return Segmentation(mask)

    @property
    def size(self):
        h, w = self.mask.shape
        return w, h

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self

    def clone(self):
        return Segmentation(self.mask.copy(), infos=self.infos.copy())

    def resize_(self, size, *args, **kwargs):
        self.mask = cv2.resize(self.mask, size, interpolation=cv2.INTER_NEAREST)
        return self

    def resize(self, size, *args, **kwargs):
        return Segmentation(self.mask, infos=self.infos).resize_(size, *args, **kwargs)

    def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
        if horizontal:
            self.mask = np.flip(self.mask, axis=1)
        if vertical:
            self.mask = np.flip(self.mask, axis=0)
        return self

    def flip(self, horizontal=False, vertical=False, *args, **kwargs):
        return Segmentation(self.mask, infos=self.infos).flip_(horizontal, vertical, *args, **kwargs)

    def crop_(self, x, y, w, h, *args, **kwargs):
        self.mask = self.mask[y:y + h, x:x + w]
        return self

    def crop(self, x, y, w, h, *args, **kwargs):
        return Segmentation(self.mask[y:y + h, x:x + w], infos=self.infos)

    def pad_(self, padding, seg_fill=None, **kwargs):
        seg_fill = self.IgnoreIndex if seg_fill is None else seg_fill
        lp, tp, rp, bp = self._get_padding_param(padding)
        self.mask = cv2.copyMakeBorder(self.mask, tp, bp, lp, rp, borderType=cv2.BORDER_CONSTANT, value=seg_fill)
        return self

    def pad(self, padding, seg_fill=None, **kwargs):
        return Segmentation(self.mask, infos=self.infos).pad_(padding, seg_fill, **kwargs)

    def affine(self, affine_matrix, output_size, seg_fill=None, **kwargs):
        seg_fill = self.IgnoreIndex if seg_fill is None else seg_fill
        if isinstance(affine_matrix, torch.Tensor):
            affine_matrix = affine_matrix.cpu().numpy()
        seg = cv2.warpAffine(self.mask, affine_matrix[:2], output_size, flags=cv2.INTER_NEAREST, borderValue=seg_fill)
        return Segmentation(seg, output_size, infos=self.infos)

    def restore(self, rt: Restoration):
        result = self
        for t in rt.get_transforms():
            result = getattr(result, t[0])(*t[1:-1], **t[-1])
        return result

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size}{self.infos if self.infos else ""})'

    def draw(self, img: np.ndarray = None, seg_colors=None, seg_alpha=0.5, num_classes=None, *args, **kwargs):
        if img is None:
            img = np.zeros((*self.size, 3), dtype=np.uint8)
            seg_alpha = 1.0

        mask = self.mask.astype(np.uint8)
        N = np.max(mask) if num_classes is None else num_classes
        if N == 0:
            return img
        if seg_colors is None:
            cs = np.array(get_colors(N) + [[255, 255, 255]] * (256 - N), dtype=np.uint8)
        else:
            cs = np.array(seg_colors + [[255, 255, 255]] * 256, dtype=np.uint8).reshape(-1, 3)
        assert mask.shape == img.shape[:2], f'Shape Error: {mask.shape} != {img.shape[:2]}!'
        img = cv2.addWeighted(img, 1., cs[mask], seg_alpha, 0)
        return img
