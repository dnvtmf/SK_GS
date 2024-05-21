from typing import Union, Tuple
from pathlib import Path

import matplotlib
import numpy as np
import imageio
from PIL import Image
import torch
from torch import Tensor

__all__ = [
    'image_extensions', 'as_np_image',
    'rgb_to_srgb', 'srgb_to_rgb', 'rgb_to_gray',
    'load_image', 'load_gray', 'load_RGB', 'load_RGBA',
    'save_image', 'save_image_raw', 'save_gray_image',
    'image_checkerboard'
]  # yapf: disable

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']


def as_np_image(img: Union[Tensor, np.ndarray]) -> np.ndarray:
    """convert the [0, 1.]  tensor to [0, 255] uint8 image """
    if isinstance(img, Tensor):
        return img.detach().clamp(0, 1).mul_(255.).cpu().numpy().astype(np.uint8)
    elif img.dtype != np.uint8:
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        return img


# ----------------------------------------------------------------------------
# sRGB color transforms
# ----------------------------------------------------------------------------


@torch.jit.script
def _rgb_to_srgb(f: Tensor) -> Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055)


def rgb_to_srgb(f: Tensor) -> Tensor:
    if f.shape[-1] == 3:
        return _rgb_to_srgb(f)
    else:
        assert f.shape[-1] == 4
        return torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:]), dim=-1)


@torch.jit.script
def _srgb_to_rgb(f: Tensor) -> Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))


def srgb_to_rgb(f: Tensor) -> Tensor:
    if f.shape[-1] == 3:
        return _srgb_to_rgb(f)
    else:
        assert f.shape[-1] == 4
        return torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:]), dim=-1)


def rgb_to_gray(image: Tensor) -> Tensor:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def load_image(filepath, **kwargs):
    # return imageio.v3.imread(filepath, **kwargs)
    return np.array(Image.open(filepath))


def load_gray(filepath):
    return np.array(Image.open(filepath).convert('L'))


def load_RGB(filepath):
    return np.array(Image.open(filepath).convert('RGB'))


def load_RGBA(filepath):
    return np.array(Image.open(filepath).convert('RGBA'))


def save_gray_image(
    filepath, image: Union[np.ndarray, Tensor], color: Union[np.ndarray, Tensor, str, None] = 'viridis', **kwargs
):
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()
    if image.dtype != np.uint8:
        # assert not image.dtype.is_floating_point
        assert 0 <= image.min() and image.max() < 256
        image = image.astype(np.uint8)
    if color is None:
        save_image(filepath, image, **kwargs)
        return
    if isinstance(color, str):
        indices = np.unique(image)
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cmap = matplotlib.colormaps['viridis'].resampled(len(indices))
        color = np.zeros((256, 3), dtype=np.float64)
        color[indices] = cmap(np.arange(len(indices)))[..., :3]
    elif isinstance(color, Tensor):
        color = color.detach().cpu().numpy()
    if color.dtype != np.uint8:
        color = np.clip(np.rint(color[..., :3] * 255.), 0, 255).astype(np.uint8)
    assert color.shape == (256, 3)
    img_ = Image.fromarray(image, 'P')
    img_.putpalette(color)
    img_.save(Path(filepath).with_suffix('.png'), format='PNG')


def save_image(filepath, image: Union[np.ndarray, Tensor], **kwargs):
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image * 255.), 0, 255).astype(np.uint8)
    imageio.v3.imwrite(filepath, image, **kwargs)


def save_image_raw(filepath, image: Union[np.ndarray, Tensor], **kwargs):
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()
    try:
        imageio.v3.imwrite(filepath, image, **kwargs)
    except:
        print(f"ERROR: FAILED to save image {filepath}")


# ----------------------------------------------------------------------------
# Image scaling
# ----------------------------------------------------------------------------


def scale_img_hwc(x: Tensor, size, mag='bilinear', min='area') -> Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhwc(x: Tensor, size, mag='bilinear', min='area') -> Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (
        x.shape[1] < size[0] and x.shape[2] < size[1]
    ), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def avg_pool_nhwc(x: Tensor, size) -> Tensor:
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


## generate special image


def image_checkerboard(size: Tuple[int, int], checker_size=8) -> np.ndarray:
    """ 生成棋盘状图像

    Args:
        size: (W, H) 图像的大小
        checker_size: 棋格的大小

    Returns:
        images with shape [size[0], size[1], 3] range [0, 1]
    """
    tiles_y = (size[1] + (checker_size * 2) - 1) // (checker_size * 2)
    tiles_x = (size[0] + (checker_size * 2) - 1) // (checker_size * 2)
    check = np.kron([[1, 0] * tiles_x, [0, 1] * tiles_x] * tiles_y, np.ones((checker_size, checker_size))) * 0.33 + 0.33
    check = check[:size[1], :size[0]]
    return np.stack((check, check, check), axis=-1)


def test():
    import matplotlib.pyplot as plt
    plt.imshow(image_checkerboard((400, 200), 8))
    plt.show()
