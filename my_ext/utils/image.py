from typing import Union

import numpy as np
import seaborn as sns
import torch
from torch import Tensor

__all__ = ['get_colors', 'color_labels', 'color_one_hot']


def get_colors(num_colors=8, mode=0, return_numpy=False, channels=3, shuffle=True, **kwargs):
    if mode == 0:
        colors = sns.color_palette("hls", num_colors, **kwargs)
    elif mode == 1:
        colors = sns.color_palette('Blues', num_colors)
    else:
        colors = sns.color_palette(n_colors=num_colors, **kwargs)
    colors = np.array(colors)
    if channels == 4:
        colors = np.concatenate([colors, np.ones_like(colors[:, :1])], axis=-1)
    if shuffle:
        np.random.shuffle(colors)
    return colors if return_numpy else torch.from_numpy(colors)


def color_labels(
    labels: Union[np.ndarray, Tensor],
    max_value: int = None,
    channels=3,
    special: int = None,
    special_color=(1., 1., 1.),
    mode=0,
    shuffle=True,
    **kwargs
):
    """给每种label一种颜色"""
    max_value = labels.max().item() + 1 if max_value is None else max_value
    colors = get_colors(max_value, mode, not isinstance(labels, Tensor), channels, shuffle, **kwargs)
    if special is not None:
        assert 0 <= special < max_value
        colors[special] = torch.tensor(special_color) if isinstance(labels, Tensor) else np.array(special_color)
    if isinstance(labels, Tensor):
        colors = colors.to(labels.device)
    return colors[labels]


def color_one_hot(masks: Tensor, channels=3, empty_color=(1., 1., 1.), dim=-1, **kwargs):
    """给one_hot的masks赋予颜色, 如果是多标签则取最大的一个, 无标签时颜色为: empty_color"""
    colors = get_colors(masks.shape[dim], channels=channels, **kwargs).to(masks.device)
    colors = torch.cat([colors.new_tensor([empty_color]), colors], dim=0)
    shape = [1] * masks.ndim
    shape[dim] = -1
    index = torch.arange(masks.shape[dim], device=masks.device).view(shape) + 1
    label = (masks.bool() * index).amax(dim)
    return colors[label]


def test():
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    C = 10
    labels = torch.randint(0, C, (20, 20))
    labels = F.interpolate(labels[None, None].float(), (200, 200), mode='nearest')[0, 0].long()
    print(labels.shape)
    colored = color_labels(labels, max_value=C).numpy()
    print(colored.shape)
    plt.imshow(colored)
    plt.show()
    colored = color_one_hot(F.one_hot(labels, C)).numpy()
    print(colored.shape)
    plt.imshow(colored)
    plt.show()
