""" OPENCV coordinate transformation 三维坐标变换

世界空间World space和观察空间view space为右手系
(COLMAP坐标系) RDF,  +x 朝右, +y 朝下,  +z 朝里
COLMAP/OpenCV:
     z
   ↗
  .-------> x
  |
  |
  ↓
  y
裁剪空间Clip space为右手系
     z
   ↗
  .-------> x
  |
  |
  ↓
  y
屏幕坐标系： X 轴向右为正，Y 轴向下为正，坐标原点位于窗口的左上角 (左手系: z轴向屏幕内，表示深度)
    z
  ↗
.------> x
|
|
↓
y
坐标变换矩阵: T{s}2{d} Transform from {s} space to {d} space
{s} 和 {d} 包括世界World坐标系、观察View坐标系、裁剪Clip坐标系、屏幕Screen坐标

Tv2s即相机内参，Tw2v即相机外参
"""

from typing import Tuple, Union, Any, Sequence, Optional
import math

import torch
import numpy as np
from torch import Tensor

from my_ext.utils.torch_utils import to_tensor
from .misc import normalize
from .coord_trans_common import fov_to_focal
from .xfm import apply

TensorType = Union[float, int, np.ndarray, Tensor]


## 世界坐标系相关
def coord_spherical_to(radius: TensorType, thetas: TensorType, phis: TensorType) -> Tensor:
    """ 球坐标系 转 笛卡尔坐标系

    Args:
        radius: 径向半径 radial distance, 原点O到点P的距离 [0, infity]
        thetas: 极角 polar angle, -y轴与连线OP的夹角 [0, pi]
        phis: 方位角 azimuth angle, 正x轴与连线OP在xz平面的投影的夹角, [0, 2 * pi], 顺时针, +z轴 -0.5pi
    Returns:
        Tensor: 点P的笛卡尔坐标系, shape: [..., 3]
    """
    radius = to_tensor(radius, dtype=torch.float32)
    thetas = torch.pi - to_tensor(thetas, dtype=torch.float32)
    phis = -to_tensor(phis, dtype=torch.float32)
    return torch.stack([
        radius * torch.sin(thetas) * torch.cos(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.sin(phis),
    ], dim=-1)


def coord_to_spherical(points: Tensor):
    """ 笛卡尔坐标系(OpenGL) 转 球坐标系

    Args:
        points (Tensor): 点P的笛卡尔坐标系, shape: [..., 3]

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 点P的球坐标系: 径向半径、极角和方位角
    """
    raduis = points.norm(p=2, dim=-1)  # type: Tensor
    thetas = torch.arccos(-points[..., 1] / raduis)
    phis = torch.arctan2(-points[..., 2], points[..., 0])
    return raduis, thetas, phis


## 相机坐标系相关
def look_at(eye: Tensor, at: Tensor = None, up: Tensor = None, inv=False) -> Tensor:
    if at is None:
        dir_vec = torch.zeros_like(eye)
        dir_vec[..., 3] = 1.
    else:
        dir_vec = normalize(eye - at)
    dir_vec = -dir_vec

    if up is None:
        up = torch.zeros_like(dir_vec)
        # if dir is parallel with y-axis, up dir is z axis, otherwise is y-axis
        y_axis = dir_vec.new_tensor([0, -1., 0]).expand_as(dir_vec)
        y_axis = torch.cross(dir_vec, y_axis, dim=-1).norm(dim=-1, keepdim=True) < 1e-6
        up = torch.scatter_add(up, -1, y_axis + 1, 1 - y_axis.to(up.dtype) * 2)
    shape = eye.shape
    right_vec = -normalize(torch.cross(up, dir_vec, dim=-1))  # 相机空间x轴方向
    up_vec = torch.cross(dir_vec, right_vec, dim=-1)  # 相机空间y轴方向
    if inv:
        Tv2w = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
        Tv2w[..., :3, 0] = right_vec
        Tv2w[..., :3, 1] = up_vec
        Tv2w[..., :3, 2] = dir_vec
        Tv2w[..., :3, 3] = eye
        return Tv2w
    else:
        R = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
        T = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
        R[..., 0, :3] = right_vec
        R[..., 1, :3] = up_vec
        R[..., 2, :3] = dir_vec
        T[..., :3, 3] = -eye
        world2view = R @ T
        return world2view


def look_at_get(Tv2w: Tensor):
    eye = Tv2w[..., :3, 3]
    right_vec = -Tv2w[..., :3, 0]
    up_vec = Tv2w[..., :3, 1]
    dir_vec = Tv2w[..., :3, 2]
    at = eye + dir_vec
    return eye, at, torch.cross(dir_vec, right_vec, dim=-1)


def camera_intrinsics(focal=None, cx_cy=None, size=None, fovy=np.pi, inv=False, **kwargs) -> Tensor:
    """生成相机内参K/Tv2s, 请注意坐标系
    .---> u
    |
    ↓
    v
    """
    W, H = size
    if focal is None:
        focal = fov_to_focal(fovy, H)
    if cx_cy is None:
        cx, cy = 0.5 * W, 0.5 * H
    elif isinstance(cx_cy, Tensor):
        cx, cy = cx_cy.unbind(-1)
    else:
        cx, cy = cx_cy
    if isinstance(focal, Tensor) and focal.ndim > 1 and focal.shape[-1] == 2:
        focal_x, focal_y = focal.unbind(-1)
    else:
        focal_x, focal_y = focal, focal
    shape = [x.shape for x in [focal_x, focal_y, cx, cy] if isinstance(x, Tensor)]
    if len(shape) > 0:
        shape = list(torch.broadcast_shapes(*shape))
    if inv:  # Ts2v
        Ts2v = torch.zeros(shape + [3, 3], **kwargs)
        Ts2v[..., 0, 0] = 1 / focal_x
        Ts2v[..., 0, 2] = -cx * Ts2v[..., 0, 0]
        Ts2v[..., 1, 1] = 1. / focal_y
        Ts2v[..., 1, 2] = -cy * Ts2v[..., 1, 1]
        Ts2v[..., 2, 2] = 1
        return Ts2v
    else:
        K = torch.zeros(shape + [3, 3], **kwargs)  # Tv2s
        K[..., 0, 0] = focal_x
        K[..., 0, 2] = cx
        K[..., 1, 1] = focal_y
        K[..., 1, 2] = cy
        K[..., 2, 2] = 1
        return K


def perspective(fovy: Union[float, Tensor] = 0.7854, aspect=1.0, n=0.1, f=1000.0, device=None, size=None):
    """OpenCV 透视投影矩阵

    Args:
        fovy: 弧度. Defaults to 0.7854.
        aspect: 长宽比W/H. Defaults to 1.0.
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.
        size: (W, H)

    Returns:
        Tensor: 透视投影矩阵
    """
    shape = []
    if size is not None:
        aspect = size[0] / size[1]
    for x in [fovy, aspect, n, f]:
        if isinstance(x, Tensor):
            shape = x.shape
    Tv2c = torch.zeros(*shape, 4, 4, dtype=torch.float, device=device)
    y = torch.tan(fovy * 0.5) if isinstance(fovy, Tensor) else np.tan(fovy * 0.5)
    x = y * aspect
    top = y * n
    bottom = -top
    right = x * n
    left = -right
    z_sign = 1.0

    Tv2c[..., 0, 0] = 2.0 * n / (right - left)
    Tv2c[..., 1, 1] = 2.0 * n / (top - bottom)
    Tv2c[..., 0, 2] = (right + left) / (right - left)
    Tv2c[..., 1, 2] = (top + bottom) / (top - bottom)
    Tv2c[..., 3, 2] = z_sign
    Tv2c[..., 2, 2] = z_sign * f / (f - n)
    Tv2c[..., 2, 3] = -(f * n) / (f - n)
    return Tv2c


def perspective2(size, focals=None, FoV=None, pp=None, n=0.1, f=1000.0, device=None):
    """OpenCV 透视投影矩阵

    Args:
        size (int | Tuple[int, int]): (W, H)
        focals (float | list | tuple | Tensor): focal length, shape: [...], [..., 1], [..., 2]
        FoV (float | list | tuple | Tensor): filed of view, radians (e.g, pi), shape: [...], [..., 1], [..., 2]
        pp (float | list | tuple | Tensor): position of center, shape: [...], [..., 1], [..., 2]
        n (float | list | tuple | Tensor): near. Defaults to 0.1.
        f (float | list | tuple | Tensor): far. Defaults to 1000.0.
        device: Defaults to None.
    Returns:
        Tensor: 透视投影矩阵
    """
    shape = torch.Size([1])
    focals, FoV, pp, n, f = [to_tensor(x) for x in [focals, FoV, pp, n, f]]
    for x in [focals, FoV, pp, n, f]:
        if isinstance(x, Tensor):
            if x.ndim > 0 and x.shape[-1] in [1, 2]:
                shape = torch.broadcast_shapes(shape, x.shape[:-1])
            else:
                shape = torch.broadcast_shapes(shape, x.shape)
    Tv2c = torch.zeros(*shape, 4, 4, device=device)
    W, H = size
    if pp is None:
        cx, cy = W / 2, H / 2
    else:
        if pp.ndim > 0 and pp.shape[-1] <= 2:
            cx, cy = pp[..., 0], pp[..., 1 if pp.shape[-1] == 2 else 0]
        else:
            cx, cy = pp, pp

    if FoV is not None:
        if FoV.ndim > 0 and FoV.shape[-1] <= 2:
            fx = fov_to_focal(FoV[..., 0], W)
            fy = fov_to_focal(FoV[..., 1 if FoV.shape[-1] == 2 else 0], H)
        else:
            fx = fov_to_focal(FoV, W)
            fy = fov_to_focal(FoV, H)
    else:
        assert focals is not None
        if focals.ndim > 0 and focals.shape[-1] <= 2:
            fx, fy = focals[..., 0], focals[..., 1 if focals.shape[-1] == 2 else 0]
        else:
            fx = fy = focals

    z_sign = 1.0
    Tv2c[..., 0, 0] = 2.0 * fx / W
    Tv2c[..., 1, 1] = 2.0 * fy / H
    Tv2c[..., 0, 2] = (2.0 * cx - W) / W
    Tv2c[..., 1, 2] = (2.0 * cy - H) / H
    Tv2c[..., 2, 2] = z_sign * (f + n) / (f - n)
    Tv2c[..., 2, 3] = -(f * n) / (f - n)
    Tv2c[..., 3, 2] = z_sign
    return Tv2c


def point2pixel(
    points: Tensor, Tw2v: Tensor = None, Tv2s: Tensor = None, Tv2c: Tensor = None, Tw2c: Tensor = None,
    size: Tuple[int, int] = None
):
    """
    Args:
        points (Tensor):  containing 3D points with shape [..., 3] or [..., 4]
        Tw2v (Union[None, Tensor]): world to view space transformation matrix with shape [..., 4, 4]
        Tv2s (Optional[Tensor]): view to screen space transformation matrix with shape [..., 3, 3]
        Tv2c (Optional[Tensor]): view to clip space transformation matrix with shape [..., 4, 4]
        Tw2c (Optional[Tensor]): world to clip space transformation matrix with shape [..., 4, 4]
        size (Optional[Tuple[int, int]]): (W, H)
    Returns:
        (Tensor, Tensor): pixel containing 2D pixel with shape [..., 2] and depth: [...]
    """
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # to homo
    if Tw2v is not None:
        points = apply(points, Tw2v)
    if Tv2s is not None:
        points = apply(points[..., :3], Tv2s)
        pixel = points[..., :2] / points[..., 2:3]
        return pixel, points[..., 2]
    else:
        ndc = apply(points, Tv2c)
        ndc = ndc[..., :3] / ndc[..., 3:]
        pixel = torch.zeros_like(ndc[..., :2])
        pixel[..., 0] = (1 + ndc[..., 0]) * 0.5 * size[0]  # - 0.5
        pixel[..., 1] = (1 + ndc[..., 1]) * 0.5 * size[1]  # - 0.5
        return pixel, points[..., 2]
