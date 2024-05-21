""" coordinate transformation 三维坐标变换

世界空间World space和观察空间view space为右手系
(OpenGL坐标系) RUB，+x向右手边, +y指向上方, +z指向屏幕外 (MeshLab, Open3D)
(Blender坐标系） RUB, +x 朝右, +y 朝里,  +z 朝上
(COLMAP坐标系) RDF,  +x 朝右, +y 朝下,  +z 朝里
(LLFF坐标系) DRB,  +x 朝下, +y 朝外,  +z 朝左
(Pytorch3d坐标系) LUF,  +x 朝里, +y 朝上,  +z 朝右
(见: https://zhuanlan.zhihu.com/p/593204605/ )
OpenGL:            Blender:       COLMAP/OpenCV:    LLFF     Pytorch3D
    y              z                   z                     y
    ↑              ↑                 ↗                       ↑
    |              |   y            .-------> x  z <------.  |   x
    |              | ↗              |                    ↙|  | ↗
    .-------> x    .-------> x      |                  y  |  .------> z
   ↙                                ↓                     ↓
  z                                 y                     x
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
import logging
from typing import Optional, Tuple, Any

from torch import Tensor

from .coord_trans_common import *
from . import coord_trans_opencv, coord_trans_opengl

TensorType = Union[float, int, np.ndarray, Tensor]

__all__ = [
    'COORDINATE_SYSTEM', 'coordinate_system', 'set_coorder_system',
    'convert_coord_system_matrix', 'convert_coord_system_points', 'convert_coord_system', 'opengl_to_opencv',
    'coord_spherical_to', 'coord_to_spherical', 'look_at', 'camera_intrinsics', 'perspective', 'ortho'
]
COORDINATE_SYSTEM = 'opengl'
coordinate_system = {
    'opengl': 'opengl',
    'blender': 'blender',
    'colmap': 'opencv',
    'opencv': 'opencv',
    'llff': 'llff',
    'pytorch3d': 'pytorch3d',
}


def opengl_to_opencv(Tw2v: Tensor = None, Tv2w: Tensor = None, inv=False):
    if Tw2v is None:
        Tw2v = Tv2w.inverse()
    Tw2v[..., 1:3, :3] *= -1
    Tw2v[..., :3, 3] *= -1
    return Tw2v, torch.inverse(Tw2v)


def convert_coord_system(T: Tensor, src='opengl', dst='opengl', inverse=False) -> Tensor:
    """ convert coordiante system from <source> to <goal>: p_dst = M @ p_src
    Args:
        T: transformation matrix with shape [..., 4, 4], can be Tw2v or Tv2w
        src: the source coordinate system, must be blender, colmap, opencv, llff, PyTorch3d, opengl
        dst: the destination coordinate system
        inverse: inverse apply (used when T is Tv2w)
    Returns:
        Tensor: converted transformation matrix
    """
    mapping = {
        'opengl': 'opengl',
        'blender': 'opengl',
        'colmap': 'opencv',
        'opencv': 'opencv',
        'llff': 'llff',
        'pytorch3d': 'pytorch3d',
    }
    src = mapping[src.lower()]
    dst = mapping[dst.lower()]
    if src == dst:
        return T
    if inverse:
        src, dst = dst, src
    if src == 'opengl':
        M = T.new_tensor({
            'opencv': [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
            'llff': [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'pytorch3d': [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.]],
        }[dst])
    elif src == 'opencv':
        M = T.new_tensor({
            'opengl': [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
            'llff': [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'pytorch3d': [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.]],
        }[dst])
    elif src == 'llff':
        M = T.new_tensor({
            'opengl': [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]],
            'opencv': [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.]],
            'pytorch3d': [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
        }[dst])
    elif src == 'pytorch3d':
        M = T.new_tensor({
            'opengl': [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'opencv': [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.]],
            'llff': [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
        }[dst])
    else:
        raise NotImplementedError(f"src={src}, dst={dst}")
    return T @ M if inverse else M @ T


def convert_coord_system_matrix(T: Tensor, src='opengl', dst='opengl') -> Tensor:
    """ convert coordiante system from <source> to <goal>: p_dst = M @ p_src
    Args:
        T: transformation matrix with shape [..., 4, 4], can be Tw2v or Tv2w
        src: the source coordinate system, must be blender, colmap, opencv, llff, PyTorch3d, opengl
        dst: the destination coordinate system
    Returns:
        Tensor: converted transformation matrix
    """
    if src == dst:
        return T
    T = T.clone()
    # T[..., :3] = convert_coord_system_points(T[..., :3], src, dst)
    T = T.transpose(-1, -2)
    T[..., :3] = convert_coord_system_points(T[..., :3], src, dst)
    T = T.transpose(-1, -2)
    return T


def convert_coord_system_points(points: Tensor, src='opengl', dst='opengl') -> Tensor:
    if src == 'opengl':
        if dst == 'opengl':
            return points
        elif dst == 'blender':
            return torch.stack([points[..., 0], -points[..., 2], points[..., 1]], dim=-1)
        elif dst == 'opencv':
            return torch.stack([points[..., 0], -points[..., 1], -points[..., 2]], dim=-1)
        elif dst == 'llff':
            return torch.stack([-points[..., 1], points[..., 2], -points[..., 0]], dim=-1)
        else:
            return torch.stack([-points[..., 2], points[..., 1], points[..., 0]], dim=-1)
    elif src == 'blender':
        if dst == 'opengl':
            return torch.stack([points[..., 0], points[..., 2], -points[..., 1]], dim=-1)
        elif dst == 'blender':
            return points
        elif dst == 'opencv':
            return torch.stack([points[..., 0], -points[..., 2], points[..., 1]], dim=-1)
        elif dst == 'llff':
            return torch.stack([-points[..., 2], -points[..., 1], -points[..., 0]], dim=-1)
        else:
            return torch.stack([points[..., 1], points[..., 2], points[..., 0]], dim=-1)
    elif src == 'opencv':
        if dst == 'opengl':
            return torch.stack([points[..., 0], -points[..., 1], -points[..., 2]], dim=-1)
        elif dst == 'blender':
            return torch.stack([points[..., 0], points[..., 2], -points[..., 1]], dim=-1)
        elif dst == 'opencv':
            return points
        elif dst == 'llff':
            return torch.stack([points[..., 1], -points[..., 2], -points[..., 0]], dim=-1)
        else:
            return torch.stack([points[..., 2], -points[..., 1], points[..., 0]], dim=-1)
    elif src == 'llff':
        if dst == 'opengl':
            return torch.stack([-points[..., 2], -points[..., 0], points[..., 1]], dim=-1)
        elif dst == 'blender':
            return torch.stack([-points[..., 2], -points[..., 1], -points[..., 0]], dim=-1)
        elif dst == 'opencv':
            return torch.stack([-points[..., 2], points[..., 0], -points[..., 1]], dim=-1)
        elif dst == 'llff':
            return points
        else:
            return torch.stack([-points[..., 1], -points[..., 0], -points[..., 2]], dim=-1)
    elif src == 'pytorch3d':
        if dst == 'opengl':
            return torch.stack([points[..., 2], points[..., 1], -points[..., 0]], dim=-1)
        elif dst == 'blender':
            return torch.stack([points[..., 2], points[..., 0], points[..., 1]], dim=-1)
        elif dst == 'opencv':
            return torch.stack([points[..., 2], -points[..., 1], points[..., 0]], dim=-1)
        elif dst == 'llff':
            return torch.stack([-points[..., 1], -points[..., 0], -points[..., 2]], dim=-1)
        else:
            return points
    else:
        raise NotImplementedError(f"src={src}, dst={dst}")


_coord_spherical_to = coord_trans_opengl.coord_spherical_to
_coord_to_spherical = coord_trans_opengl.coord_to_spherical
_look_at = coord_trans_opengl.look_at
_camera_intrinsics = coord_trans_opengl.camera_intrinsics
_perspective = coord_trans_opengl.perspective
_ortho = coord_trans_opengl.ortho


def coord_spherical_to(radius: TensorType, thetas: TensorType, phis: TensorType) -> Tensor:
    """ 球坐标系 转 笛卡尔坐标系

    Args:
        radius: 径向半径 radial distance, 原点O到点P的距离 [0, infity]
        thetas: 极角 polar angle, 朝上(+y轴)与连线OP的夹角 [0, pi]
        phis: 方位角 azimuth angle, 正x轴与连线OP在xz平面的投影的夹角, [0, 2 * pi], 顺时针, +z：0.5pi
    Returns:
        Tensor: 点P的笛卡尔坐标系, shape: [..., 3]
    """
    return _coord_spherical_to(radius, thetas, phis)


def coord_to_spherical(points: Tensor):
    """ 笛卡尔坐标系(OpenGL) 转 球坐标系

    Args:
        points (Tensor): 点P的笛卡尔坐标系, shape: [..., 3]

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 点P的球坐标系: 径向半径、极角和方位角
    """
    return _coord_to_spherical(points)


def look_at(eye: Tensor, at: Tensor = None, up: Tensor = None, inv=False) -> Tensor:
    return _look_at(eye, at, up, inv)


def camera_intrinsics(focal=None, cx_cy=None, size=None, fovy=np.pi, inv=False, **kwargs) -> Tensor:
    """生成相机内参K/Tv2s, 请注意坐标系
    .---> u
    |
    ↓
    v
    """
    return _camera_intrinsics(focal, cx_cy, size, fovy, inv, **kwargs)


def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None, size=None):
    # type: (Union[Tensor, float], float, float, float, Any, Optional[Tuple[int, int]])->Tensor
    """透视投影矩阵

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
    return _perspective(fovy, aspect, n, f, device, size)


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
    return _ortho(l, r, b, t, n, f, device)


def set_coorder_system(coord):
    coord = coordinate_system[coord.lower()]
    global COORDINATE_SYSTEM
    COORDINATE_SYSTEM = coord
    global _coord_spherical_to, _coord_to_spherical, _look_at, _camera_intrinsics, _perspective, _ortho
    if coord == 'opengl':
        _coord_spherical_to = coord_trans_opengl.coord_spherical_to
        _coord_to_spherical = coord_trans_opengl.coord_to_spherical
        _look_at = coord_trans_opengl.look_at
        _camera_intrinsics = coord_trans_opengl.camera_intrinsics
        _perspective = coord_trans_opengl.perspective
        _ortho = coord_trans_opengl.ortho
    elif coord == 'opencv':
        _coord_spherical_to = coord_trans_opencv.coord_spherical_to
        _coord_to_spherical = coord_trans_opencv.coord_to_spherical
        _look_at = coord_trans_opencv.look_at
        _camera_intrinsics = coord_trans_opencv.camera_intrinsics
        _perspective = coord_trans_opencv.perspective
        _ortho = coord_trans_opencv.ortho
    else:
        raise NotImplementedError(f"coord {coord} not supported")
    logging.info(f"[red]set coordinate system to {coord}")


def test_coord_system():
    from my_ext.utils import set_printoptions
    set_printoptions()
    """
       OpenGL:        Blender:       COLMAP/OpenCV:    LLFF     Pytorch3D
      y              z                   z                     y
      ↑              ↑                 ↗                       |   x
      |              |   y            .-------> x  z <------.  | ↗
      |              | ↗              |                    ↙|  .------> z
      .-------> x    .-------> x      |                  y  |
     ↙                                ↓                     ↓
    z                                 y                     x
    """
    p_list = [
        ('opengl', np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1., 2., 3.]])),
        ('blender', np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0], [1., -3., 2.]])),
        ('opencv', np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1], [1., -2., -3.]])),
        ('llff', np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0], [-2., 3., -1.]])),
        ('pytorch3d', np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0], [-3., 2, 1]])),
    ]
    np.set_printoptions(precision=6, suppress=True)
    print()

    # Tw2v = torch.eye(4, dtype=torch.float64)
    from my_ext.ops_3d import opengl
    Tw2v = opengl.look_at(torch.randn(3), torch.randn(3), torch.randn(3)).to(torch.float64)
    Tv2w = torch.inverse(Tw2v)
    for src, src_p in p_list:
        for dst, dst_p in p_list:
            print(src, dst)
            assert (dst_p == convert_coord_system_points(torch.from_numpy(src_p), src, dst).numpy()).all()
            p1 = torch.from_numpy(src_p) @ Tw2v[:3, :3].T + Tw2v[:3, 3]
            Tw2v_dst = convert_coord_system_matrix(Tw2v.clone(), src, dst)
            Tw2v_dst2 = convert_coord_system(Tw2v, src, dst)
            print(Tw2v_dst - Tw2v_dst2)
            # p2 = torch.from_numpy(dst_p) @ Tw2v_dst[:3, :3].T + Tw2v_dst[:3, 3]
            # p2 = convert_coord_system_points(p2, dst, src)
            # print((p1 - p2).abs().max())
            # assert (p1 - p2).abs().max() < 1e-10
            # T1 = convert_coord_system_matrix(Tw2v.clone(), src, dst)
            # T2 = convert_coord_system_matrix(Tv2w.clone(), src, dst)
            # assert ((T1 @ T2) - torch.eye(4)).abs().max() < 1e-10
