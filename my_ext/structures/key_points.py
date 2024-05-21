import cv2
import numpy as np
import torch

from my_ext.structures import Structure2D, Restoration
from my_ext.utils import get_colors

__all__ = ['KeyPoints']


class KeyPoints(Structure2D):
    NUM_KEYPOINTS = None
    NAMES = []
    FLIP_INDEXES = None
    SKELETON = []

    def __init__(self, kps, image_size, masks=None, infos=None):
        """
        Args:
            kps:
            image_size:
            masks: 1: visible, -1: invisible; 0: not annotated
        """
        super().__init__(infos)
        if kps is None:
            self.kps = torch.zeros((0, self.num_keypoints(), 2))
        elif isinstance(kps, np.ndarray):
            self.kps = torch.from_numpy(kps).float()
        elif isinstance(kps, torch.Tensor):
            self.kps = kps
        else:
            self.kps = torch.tensor(kps, dtype=torch.float)
            # raise TypeError(f'Unsupported type {type(kps)} for KeyPoints')
        assert self.kps.ndim == 3 and self.kps.size(1) == self.num_keypoints() and self.kps.size(2) in [2, 3], \
            f"Error Shape of kps: {self.kps.shape}, not NxKx[2,3]"

        if masks is None:
            if self.kps.size(2) == 2:
                self.scores = self.kps.new_ones(self.kps.shape[:2], dtype=torch.float)
            else:
                self.scores = self.kps[:, :, 2].to(dtype=torch.float)
                self.kps = self.kps[:, :, :2].contiguous()
        elif isinstance(kps, np.ndarray):
            self.scores = torch.from_numpy(masks).to(dtype=torch.float, device=self.kps.device)
        else:
            assert isinstance(masks, torch.Tensor)
            self.scores = masks.to(torch.float)
        assert self.scores.shape == self.kps.shape[:2], f"masks shape {self.scores.shape} != {self.kps.shape[:2]}"

        self.size = image_size

    def __len__(self):
        return self.kps.size(0)

    def __getitem__(self, item):
        return KeyPoints(self.kps[item], self.size, masks=self.scores[item])

    def clone(self):
        return KeyPoints(self.kps.clone(), self.size, self.scores.clone(), self.infos.copy())

    @staticmethod
    def setting(names, flip_pairs, skeleton):
        KeyPoints.NAMES = names
        KeyPoints.NUM_KEYPOINTS = len(names)
        KeyPoints.SKELETON = skeleton
        KeyPoints.FLIP_INDEXES = list(range(KeyPoints.NUM_KEYPOINTS))
        for a, b in flip_pairs:
            KeyPoints.FLIP_INDEXES[a] = b
            KeyPoints.FLIP_INDEXES[b] = a

    @staticmethod
    def num_keypoints():
        if KeyPoints.NUM_KEYPOINTS is None:
            raise ValueError("Please setting KeyPoints First")
        return KeyPoints.NUM_KEYPOINTS

    def resize(self, size, *args, **kwargs):
        return self.clone().resize_(size, *args, **kwargs)

    def resize_(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            self.kps *= ratios[0]
        else:
            self.kps *= self.kps.new_tensor([[[ratios[0], ratios[1]]]])
        self.size = size
        return self

    def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
        if horizontal:
            self.kps[:, :, 0] = self.size[0] - self.kps[:, :, 0]
            if self.FLIP_INDEXES is not None:
                self.kps = self.kps[:, self.FLIP_INDEXES, :]
                self.scores = self.scores[:, self.FLIP_INDEXES]

        if vertical:
            self.kps[:, :, 1] = self.size[1] - self.kps[:, :, 1]

        return self

    def flip(self, horizontal=False, vertical=False, *args, **kwargs):
        return self.clone().flip_(horizontal, vertical, *args, **kwargs)

    def crop_(self, x, y, w, h, clamp=True, *args, **kwargs):
        self.kps -= self.kps.new_tensor([[[x, y]]])
        self.size = (w, h)
        if clamp:
            self.clip_to_image()
        else:
            self.mark_invisible()
        return self

    def crop(self, x, y, w, h, clamp=True, *args, **kwargs):
        return self.clone().crop_(x, y, w, h, *args, clamp=clamp, **kwargs)

    def pad_(self, padding, **kwargs):
        """
        Pad some pixels around the bounding box. The order is left(, top, (right, and bottom)).
        padding: int: left_pad = top_pad = right_pad = bottom_pad
                (int, int): left_pad = right_pad, top_pad = bottom_pad
                (int, int, int, int): left_pad, top_pad, right_pad, bottom_pad
        """
        left_pad, top_pad, right_pad, bottom_pad = self._get_padding_param(padding)
        self.kps += self.kps.new_tensor([[[left_pad, top_pad]]])
        w, h = self.size
        self.size = (w + left_pad + right_pad, h + top_pad + bottom_pad)
        return self

    def pad(self, padding, **kwargs):
        return self.clone().pad_(padding)

    def to(self, device):
        return KeyPoints(self.kps.to(device), self.size, self.scores.to(device))

    def mark_invisible(self):
        invisible = (self.kps[:, :, 0] < 0)
        invisible |= (self.kps[:, :, 0] > self.size[0])
        invisible |= (self.kps[:, :, 1] < 0)
        invisible |= (self.kps[:, :, 1] > self.size[1])
        invisible &= (self.scores > 0)
        self.scores[invisible].neg_()

    def mark_none(self):
        invisible = (self.kps[:, :, 0] < 0)
        invisible |= (self.kps[:, :, 0] > self.size[0])
        invisible |= (self.kps[:, :, 1] < 0)
        invisible |= (self.kps[:, :, 1] > self.size[1])
        self.scores[invisible] = 0

    def clip_to_image(self):
        self.mark_none()
        ## Do not need to clamp
        # self.kps[:, :, 0].clamp_(min=0, max=self.size[0])
        # self.kps[:, :, 1].clamp_(min=0, max=self.size[1])
        return self

    def affine(self, am, output_size, **kwargs):
        n = len(self)
        if n == 0:
            return self
        if isinstance(am, np.ndarray):
            am = torch.from_numpy(am)
        assert isinstance(am, torch.Tensor)
        am = am.to(self.kps)
        t = torch.ones(n * self.num_keypoints(), 3)
        t[:, :2] = self.kps.view(-1, 2)
        t = t.mm(am.T).reshape(n, self.num_keypoints(), 3)
        new_kps = KeyPoints(t[:, :, :2].contiguous(), self.size, self.scores)
        new_kps.crop_(0, 0, *output_size)
        return new_kps

    def restore(self, rt: Restoration):
        result = self
        for t in rt.get_transforms():
            result = getattr(result, t[0])(*t[1:-1], **t[-1])
        return result

    def draw(self, img: np.ndarray = None, show_skeleton=True, *args, **kwargs) -> np.ndarray:
        points = self.kps.detach().int().cpu().numpy()  # type: np.ndarray
        # scores = self.scores.detach().cpu().numpy()  # type: np.ndarray
        colors = get_colors(self.kps.shape[1])
        for i in range(points.shape[0]):
            ps = points[i]
            for j in range(self.kps.shape[1]):
                # score = scores[i, j]
                # if score > 0:
                #     plt.scatter(ps[j, 0], ps[j, 1], s=10, color=color)
                # elif score < 0:
                cv2.circle(img, (ps[j, 0], ps[j, 1]), 1, colors[j], thickness=2)
                # for a, b in KeyPoints.SKELETON:
                #     x1, y1 = ps[a]
                # x2, y2 = ps[b]
                # if show_skeleton and scores[i, a] != 0 and scores[i, b] != 0:
                #     plt.plot([x1, x2], [y1, y2], color=color, lw=2)
        return img
        # raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(num={len(self)}, size={self.size})'
