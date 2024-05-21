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
裁剪空间Clip space为左手系: +x指向右手边, +y 指向上方, +z指向屏幕内; z 的坐标值越小，距离观察者越近
y [-1, 1]
↑
|   z [-1, 1]
| ↗
.------> x [-1, 1]
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

from typing import Tuple, Union
import math

import torch
import numpy as np
from torch import Tensor

from my_ext.utils.torch_utils import to_tensor
from .misc import normalize
from .coord_trans_common import fov_to_focal

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
    thetas = to_tensor(thetas, dtype=torch.float32)
    phis = to_tensor(phis, dtype=torch.float32)
    # yapf: disable
    return torch.stack([
            radius * torch.sin(thetas) * torch.cos(phis),
            -radius * torch.cos(thetas),
            -radius * torch.sin(thetas) * torch.sin(phis),
        ], dim=-1)
    # yapf: enable


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

    if up is None:
        up = torch.zeros_like(dir_vec)
        # if dir is parallel with y-axis, up dir is z axis, otherwise is y-axis
        y_axis = dir_vec.new_tensor([0, -1., 0]).expand_as(dir_vec)
        y_axis = torch.cross(dir_vec, y_axis, dim=-1).norm(dim=-1, keepdim=True) < 1e-6
        up = torch.scatter_add(up, -1, y_axis + 1, 1 - y_axis.to(up.dtype) * 2)
    shape = eye.shape
    right_vec = normalize(torch.cross(up, dir_vec, dim=-1))  # 相机空间x轴方向
    up_vec = torch.cross(right_vec, dir_vec, dim=-1)  # 相机空间y轴方向
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
        T[..., :3, 3] = eye
        world2view = R @ T
        return world2view


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
    else:
        cx, cy = cx_cy
    shape = [x.shape for x in [focal, cx, cy] if isinstance(x, Tensor)]
    if len(shape) > 0:
        shape = list(torch.broadcast_shapes(*shape))
    if inv:  # Ts2v
        fr = 1. / focal
        Ts2v = torch.zeros(shape + [3, 3], **kwargs)
        Ts2v[..., 0, 0] = fr
        Ts2v[..., 0, 2] = cx * fr
        Ts2v[..., 1, 1] = fr
        Ts2v[..., 1, 2] = cy * fr
        Ts2v[..., 2, 2] = 1
        return Ts2v
    else:
        K = torch.zeros(shape + [3, 3], **kwargs)  # Tv2s
        K[..., 0, 0] = focal
        K[..., 0, 2] = cx
        K[..., 1, 1] = focal
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


def perspective2(size, fx, fy, cx, cy, n=0.1, f=1000.0, device=None):
    """OpenCV 透视投影矩阵

    Args:
        size: (W, H)
        fx:
        fy: focal length
        cx:
        cy: position of center
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.


    Returns:
        Tensor: 透视投影矩阵
    """
    shape = []
    for x in [cx, cy, n, f]:
        if isinstance(x, Tensor):
            shape = x.shape
    Tv2c = torch.zeros(*shape, 4, 4, dtype=torch.float, device=device)
    W, H = size
    z_sign = 1.0

    Tv2c[..., 0, 0] = 2.0 * fx / W
    Tv2c[..., 1, 1] = 2.0 * fy / H
    Tv2c[..., 0, 2] = (2.0 * cx - W) / W
    Tv2c[..., 1, 2] = (2.0 * cy - H) / H
    Tv2c[..., 2, 2] = z_sign * (f + n) / (f - n)
    Tv2c[..., 2, 3] = -(f * n) / (f - n)
    Tv2c[..., 3, 2] = z_sign
    return Tv2c


# @try_use_C_extension
def ortho(l=-1., r=1.0, b=-1., t=1.0, n=0.1, f=1000.0, device=None):
    """正交投影矩阵

    Args:
        # size: 长度. Defaults to 1.0.
        # aspect: 长宽比W/H. Defaults to 1.0.
        l: left plane
        r: right plane
        b: bottom place
        t: top plane
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.

    Returns:
        Tensor: 正交投影矩阵
    """
    raise NotImplementedError
    # yapf: disable
    return torch.tensor([
        [2/(r-l), 0,       0,       (l+r)/(l-r)],
        [0,       2/(t-b), 0,       (b+t)/(b-t)],
        [0,       0,       2/(n-f), (n+f)/(n-f)],
        [0,       0,       0,       1],
    ], dtype=torch.float32, device=device)
    # return torch.tensor([
    #     [1 / (size * aspect), 0, 0, 0],
    #     [0, 1 / size, 0, 0],
    #     [0, 0, -(f + n) / (f - n), -(f + n) / (f - n)],
    #     [0, 0, 0, 0],
    # ], dtype=torch.float32, device=device)
    # yapf: enable


def test():
    from my_ext.utils import set_printoptions
    from my_ext.ops_3d import opengl, convert_coord_system_matrix
    print()
    set_printoptions()
    for i in range(4):
        print(coord_spherical_to(0.1, np.deg2rad(30), np.deg2rad(90 * i)))
    for _ in range(10):
        thetas = np.random.uniform(0, 180)
        phis = np.random.uniform(0, 360)
        eye = coord_spherical_to(0.1, np.deg2rad(thetas), np.deg2rad(phis))
        r, t_, p_ = coord_to_spherical(eye)
        print(np.rad2deg(t_.item()) - thetas, np.rad2deg(p_.item()) - phis)

    eye = torch.randn(3)
    eye = torch.tensor([1, 0, 1.])
    print(eye, coord_spherical_to(*coord_to_spherical(eye)))
    at = torch.tensor([0, 0, 0.])
    up = torch.tensor([0, -1, 0.])
    pose = look_at(eye, at, up, True)
    print(pose @ look_at(eye, at, up))
    r, t, p = coord_to_spherical(eye)
    print(r, math.degrees(t), math.degrees(p))
    eye_gl = opengl.coord_spherical_to(r, t, p)
    print('eye_gl', eye_gl)
    at_gl = torch.tensor([0, 0, 0])
    up_gl = torch.tensor([0, 1, 0.])
    Tw2v = opengl.look_at(eye_gl, at_gl, up_gl)
    Tw2v = convert_coord_system_matrix(Tw2v, 'opengl', 'opencv')
    print('opengl:', Tw2v.inverse(), 'opencv', pose, sep='\n')
    print('diff:', Tw2v.inverse() - pose, sep='\n')
    # with vis3d:
    #     vis3d.add_camera_poses(pose, color=(1, 0, 0))


def test_perspective():
    from my_ext._C import get_C_function
    print()
    cu_f = get_C_function('perspective')
    fovy = torch.tensor(math.radians(120.))
    aspect = 1.0
    near, far = 0.1, 1000
    mat_1 = cu_f(fovy, aspect, near, far)
    print(mat_1)
    mat_2 = cu_f(fovy.cuda(), aspect, near, far)
    print(mat_2)
    # py_f = get_python_function('perspective')
    py_f = perspective
    mat_py = py_f(fovy, aspect, near, far)
    print(mat_py)
    assert (mat_1 - mat_2.cpu()).abs().max() < 1e-6
    assert (mat_1 - mat_py).abs().max() < 1e-6


def test_ortho():
    from my_ext._C import get_C_function
    print()
    cu_f = get_C_function('ortho')
    data = torch.tensor([-2, 3, -4, 5, 0.1, 1000])
    mat_1 = cu_f(data)
    print(mat_1)
    mat_2 = cu_f(data.cuda()).cpu()
    print(mat_2)
    # py_f = get_python_function('ortho')
    py_f = ortho
    mat_py = py_f(*data.tolist())
    print(mat_py)
    assert (mat_1 - mat_2).abs().max() < 1e-6
    assert (mat_1 - mat_py).abs().max() < 1e-6


def test_camera_intrinsics():
    import matplotlib.pyplot as plt
    from my_ext.ops_3d.xfm import point2pixel
    torch.set_printoptions(precision=6, sci_mode=False)
    np.set_printoptions(precision=6, suppress=True)
    print()
    points = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
    campos = coord_spherical_to(2., math.radians(45.), math.radians(45.))
    # campos = coord_spherical_to(2., math.radians(90.), math.radians(0.))
    print('campos:', campos)
    Tw2v = look_at(campos, torch.zeros(3))

    fovy = math.radians(90)
    size = (400, 400)
    Tv2s = camera_intrinsics(size=size, fovy=fovy)
    Ts2v = camera_intrinsics(size=size, fovy=fovy, inv=True)
    Tv2c = perspective(fovy=fovy, size=size)
    # print('Tv2w:', Tw2v.inverse())
    # print(Tw2v)
    print('Ts2v error', Tv2s.inverse() - Ts2v)
    points2 = point2pixel(points, Tw2v=Tw2v, Tv2c=Tv2c, size=size).numpy()
    points1 = point2pixel(points, Tw2v=Tw2v, Tv2s=Tv2s, size=size).numpy()
    # print(Tv2c)
    points = torch.cat([points, points.new_ones((len(points), 1))], dim=-1)
    # points2 = (points @ (Tv2c @ Tw2v).T)
    # print(points2)
    # points2 = (points2[:, :2] + 1.) * 0.5 * torch.tensor(size)
    print('points2:', points2)
    points = (points @ Tw2v.T)
    print(points)
    points = points[:, :3] / points[:, 3:]
    points = (points @ Tv2s.T)
    print(points)
    points = points[:, :] / points[:, 2:]
    # points = (Tv2s @ (Tw2v[:3, :3] @ points[:, :3].T) + Tw2v[:3, 3:4]).T
    points = points.numpy()
    print('points', points)
    print('points1', points1)
    plt.plot(points1[(0, 1), 0], size[1] - points1[(0, 1), 1], c='r')
    plt.plot(points1[(0, 2), 0], size[1] - points1[(0, 2), 1], c='g')
    plt.plot(points1[(0, 3), 0], size[1] - points1[(0, 3), 1], c='b')
    plt.plot(points2[(0, 1), 0], size[1] - points2[(0, 1), 1], c='r', ls='--')
    plt.plot(points2[(0, 2), 0], size[1] - points2[(0, 2), 1], c='g', ls='--')
    plt.plot(points2[(0, 3), 0], size[1] - points2[(0, 3), 1], c='b', ls='--')
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.show()
