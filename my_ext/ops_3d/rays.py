"""
处理射线相关函数
"""
from typing import Union, List, Tuple

import torch
from torch import Tensor


def get_rays(
    K: Tensor,
    v2w: Tensor,
    xy: Union[Tensor, List[Tensor]] = None,
    size: Tuple[int, int] = None,
    normalize=True,
    offset=0.5,
    stack_mode=True,
    sample_stride=1,
    coord='opengl',
):
    """ 生成从相机位置出发，通过成像平面上点xy的射线的起点 和 方向
    Args:
        K: 相机内参, shape: [n1, ..., nk, 3, 3]
        v2w: 相机坐标转世界坐标系的转换矩阵系数, shape: [n1, ..., nk, 3/4, 4]  k >= 0
        xy: 射线在成像平面上的投影点, shape: [m1, ... , ml, 2] l>=0
        size: (W, H) 当xy为None且H，W都不为None时， 生成HxW条射线，一一对应图像上的每个像素
        offset: 生成HxW条射线时，坐标的偏移量. 默认0.5
        normalize: 归一化射线的模长
        stack_mode: =True时, 生成的射线为[n1, ..., nk, m1, ..., ml]条， 否则为[n1, ..., nk]条
        sample_stride: 可能生成H/sample_stride x W/sample_stride条射线
    Returns: 
        rays_o: 射线起点 shape: [n1, ..., nk, m1, ..., ml, 3] 或 [n1, ..., nk]
        rays_d: 射线方向 shape: [n1, ..., nk, m1, ..., ml, 3] 或 [n1, ..., nk]
    """
    if isinstance(xy, Tensor):
        assert xy.shape[-1] == 2
        x, y = (xy + offset).unbind(-1)
    elif xy is not None:
        assert len(xy) == 2
        x, y = xy[0] + offset, xy[1] + offset
    else:
        W, H = int(size[0]), int(size[1])
        assert H > 0 and W > 0
        y = torch.arange(0, H, sample_stride, dtype=K.dtype, device=K.device) + offset
        x = torch.arange(0, W, sample_stride, dtype=K.dtype, device=K.device) + offset
        x, y = torch.meshgrid(x, y, indexing='xy')
    points = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    if stack_mode:
        K = K.view(list(K.shape[:-2]) + [1] * x.ndim + list(K.shape[-2:]))
        v2w = v2w.view(list(v2w.shape[:-2]) + [1] * x.ndim + list(v2w.shape[-2:]))

    dirs = torch.sum(points[..., None, :] * torch.inverse(K), dim=-1)  # = points @ K.T
    # if coord == 'opengl':
    dirs = -dirs
    rays_d = torch.sum(dirs[..., None, :] * v2w[..., :3, :3], dim=-1)  # = dirs @ v2w.T
    rays_o = v2w[..., :3, -1].expand_as(rays_d)
    if normalize:
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    return rays_o, rays_d


def ndc_rays(H: int, W: int, focal: float, near: float, rays_o: Tensor, rays_d: Tensor):
    """ convert to normalized device coordinate (NDC) space, using for "forward facing" scene
    for o' + t' * d', uniform sample t' ~ [0, 1], 
    equal to a linear sampling in disparity from near to ∞ in the original space
    See the Appendix in "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" 
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
