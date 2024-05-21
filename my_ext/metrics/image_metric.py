from functools import partial
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.distributed import all_reduce

from my_ext import logger
from my_ext.distributed import get_world_size
from .base import Metric, METRICS


@METRICS.register('image')
class ImageMetric(Metric):

    def __init__(self, items=None, **kwargs) -> None:
        self.device = torch.device('cuda')
        self._psnr_sum = torch.zeros(1, device=self.device)
        self._ssim_sum = torch.zeros(1, device=self.device)
        self._ms_ssim_sum = torch.zeros(1, device=self.device)
        self._lpips_alex_sum = torch.zeros(1, device=self.device)
        self._lpips_vgg_sum = torch.zeros(1, device=self.device)
        self.num_images = torch.zeros(1, device=self.device, dtype=torch.long)

        if items is not None:
            items = [item.upper() for item in items]
            assert all(item in ['PSNR', 'SSIM', 'MS_SSIM', 'LPIPS', 'LPIPS_ALEX', 'LPIPS_VGG'] for item in items)
        else:
            items = []

        if not items or 'PSNR' in items:
            # simplified since max_pixel_value is 1 here.
            # self.psnr_f = peak_signal_noise_ratio
            self.psnr_f = lambda x, y: -10 * torch.log10(torch.mean((x - y) ** 2))
            self._names['PSNR'] = None

        if not items or 'SSIM' in items:
            from torchmetrics.functional.image.ssim import structural_similarity_index_measure
            self.ssim_f = structural_similarity_index_measure
            self._names['SSIM'] = None
        else:
            self.ssim_f = None

        if 'MS_SSIM' in items:
            from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure
            self.ms_ssim_f = multiscale_structural_similarity_index_measure
            self._names['MS_SSIM'] = None
        else:
            self.ms_ssim_f = None

        if 'LPIPS' in items or 'LPIPS_ALEX' in items:
            from .lpipsPyTorch import lpips
            self.lpips_f = partial(lpips, net_type='alex')
            self._names['LPIPS'] = None
        else:
            self.lpips_f = None
        if 'LPIPS_VGG' in items:
            from .lpipsPyTorch import lpips
            self.lpips_vgg_f = partial(lpips, net_type='vgg')
            self._names['LPIPS_VGG'] = None
        else:
            self.lpips_vgg_f = None

    def reset(self):
        self.num_images = 0
        self._psnr_sum.zero_()
        self._ssim_sum.zero_()
        self._ms_ssim_sum.zero_()
        self._lpips_alex_sum.zero_()
        self._lpips_vgg_sum.zero_()

    def prepare_input(self, image: Union[np.ndarray, Tensor]):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        image = image.view(-1, *image.shape[-3:])
        image = image.to(self.device)
        if image.dtype == torch.uint8:
            image = image / 255.  # range in [0., 1]
        if image.ndim == 3:
            image = image[None]
        if image.shape[-1] == 3:  # [B, H, W, 3] --> [B, 3, H, W]
            image = image.moveaxis(-1, 1)
        assert image.ndim == 4 and image.shape[1] == 3
        return image

    @torch.no_grad()
    def update(self, images: Tensor, gt: Tensor):
        images = self.prepare_input(images)
        gt = self.prepare_input(gt)
        self.num_images += images.shape[0]

        self._psnr_sum += 0 if self.psnr_f is None else self.psnr_f(images, gt) * images.shape[0]
        self._ssim_sum += 0 if self.ssim_f is None else self.ssim_f(images, gt) * images.shape[0]
        self._ms_ssim_sum += 0 if self.ms_ssim_f is None else self.ms_ssim_f(images, gt) * images.shape[0]
        self._lpips_alex_sum += 0 if self.lpips_f is None else self.lpips_f(images, gt).mean() * images.shape[0]
        self._lpips_vgg_sum += 0 if self.lpips_vgg_f is None else self.lpips_vgg_f(images, gt).mean() * images.shape[0]

    @torch.no_grad()
    def summarize(self):
        if get_world_size() > 1:
            all_reduce(self._psnr_sum)
            all_reduce(self._ssim_sum)
            all_reduce(self._ms_ssim_sum)
            all_reduce(self._lpips_alex_sum)
            all_reduce(self._lpips_vgg_sum)
            all_reduce(self.num_images)

    def PSNR(self):
        return (self._psnr_sum / self.num_images).item()

    def SSIM(self):
        if self.ssim_f is None:
            logger.error('Please install package `torchmetrics` for SSIM metric')

        return (self._ssim_sum / self.num_images).item()

    def MS_SSIM(self):
        if self.ssim_f is None:
            logger.error('Please install package `torchmetrics` for MS_SSIM metric')
        return (self._ms_ssim_sum / self.num_images).item()

    def LPIPS(self):
        return (self._lpips_alex_sum / self.num_images).item()

    def LPIPS_ALEX(self):
        return self.LPIPS()

    def LPIPS_VGG(self):
        return (self._lpips_vgg_sum / self.num_images).item()

    def __repr__(self) -> str:
        s = []
        if 'PSNR' in self.names:
            s.append('PSNR')
        if 'SSIM' in self.names:
            s.append('SSIM')
        if 'MS_SSIM' in self.names:
            s.append('MS_SSIM')
        if 'LPIPS' in self.names or 'LPIPS_ALEX' in self.names:
            s.append(f'LPIPS[ALEX]')
        if 'LPIPS_VGG' in self.names:
            s.append(f'LPIPS[VGG]')
        return f"{self.__class__.__name__}: [{', '.join(s)}]"
