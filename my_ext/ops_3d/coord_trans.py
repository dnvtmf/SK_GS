""" coordinate transformation 三维坐标变换

世界空间World space和观察空间view space为右手系
(OpenGL坐标系) RUB，+x向右手边, +y指向上方, +z指向屏幕外 (MeshLab, Open3D)
(Blender坐标系） RUB, +x 朝右, +y 朝里,  +z 朝上
(COLMAP坐标系) RDF,  +x 朝右, +y 朝下,  +z 朝里
(LLFF坐标系) DRB,  +x 朝下, +y 朝右,  +z 朝外, 见 https://github.com/Fyusion/LLFF
(Pytorch3d坐标系) LUF,  +x 朝里, +y 朝上,  +z 朝右
(见: https://zhuanlan.zhihu.com/p/593204605/ )
OpenGL/Vulkan:     Blender:       COLMAP/OpenCV:    LLFF     Pytorch3D
    y              z                   z                     y
    ↑              ↑                 ↗                       ↑
    |              |   y            .-------> x   .-----> y  |   x
    |              | ↗              |           ↙ |          | ↗
    .-------> x    .-------> x      |          z  |          .------> z
   ↙                                ↓             ↓
  z                                 y             x
裁剪空间Clip space为左手系: +x指向右手边, +y 指向上方, +z指向屏幕内; z 的坐标值越小，距离观察者越近
y [-1, 1]                      z [-1, 1]
↑                            ↗
|                          .----> x [-1, 1]
|   z [-1, 1]              |
| ↗                        ↓
.------> x [-1, 1]         y [-1, 1]
OpenGL/DirectX              Vulkan/OpenCV (右手系)
屏幕坐标系： X 轴向右为正，Y 轴向下为正，坐标原点位于窗口的左上角 (左手系: z轴向屏幕内，表示深度)
    z
  ↗
.------> x
|
|
↓
y
坐标变换矩阵: T{s}2{d} Transform from {s} space to {d} space
{s} 和 {d} 包括世界World坐标系、观察View坐标系、裁剪Clip坐标系、屏幕Screen坐标系
例：Tv2s即相机内参，Tw2v即相机外参
"""
import logging
from typing import Optional, Tuple, Union, Any
from torch import Tensor

from .coord_trans_common import *
from . import coord_trans_opencv, coord_trans_opengl

TensorType = Union[float, int, np.ndarray, Tensor]

__all__ = [
    'coordinate_system', 'set_coord_system', 'get_coord_system',
    'convert_coord_system_matrix', 'convert_coord_system_points', 'convert_coord_system', 'opengl_to_opencv',
    'coord_spherical_to', 'coord_to_spherical',
    'look_at', 'look_at_get',
    'camera_intrinsics', 'perspective', 'perspective2',
    'point2pixel', 'ndc2pixel'
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

_convert_matrix = {
    'opengl': {
        'opengl': [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        'blender': [[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]],
        'opencv': [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]],
        'llff': [[0, -1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        'pytorch3d': [[0, 0, -1.0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 0, 1.0]]
    },
    'blender': {
        'opengl': [[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]],
        'blender': [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        'opencv': [[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]],
        'llff': [[0, 0, -1.0, 0], [1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]],
        'pytorch3d': [[0, 1.0, 0, 0], [0, 0, 1.0, 0], [1.0, 0, 0, 0], [0, 0, 0, 1.0]]
    },
    'opencv': {
        'opengl': [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]],
        'blender': [[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]],
        'opencv': [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        'llff': [[0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]],
        'pytorch3d': [[0, 0, 1.0, 0], [0, -1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 0, 1.0]]
    },
    'llff': {
        'opengl': [[0, 1.0, 0, 0], [-1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        'blender': [[0, 1.0, 0, 0], [0, 0, -1.0, 0], [-1.0, 0, 0, 0], [0, 0, 0, 1.0]],
        'opencv': [[0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]],
        'llff': [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        'pytorch3d': [[0, 0, -1.0, 0], [-1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]],
    },
    'pytorch3d': {
        'opengl': [[0, 0, 1.0, 0], [0, 1.0, 0, 0], [-1.0, 0, 0, 0], [0, 0, 0, 1.0]],
        'blender': [[0, 0, 1.0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]],
        'opencv': [[0, 0, 1.0, 0], [0, -1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 0, 1.0]],
        'llff': [[0, -1.0, 0, 0], [0, 0, 1.0, 0], [-1.0, 0, 0, 0], [0, 0, 0, 1.0]],
        'pytorch3d': [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    }
}


def opengl_to_opencv(Tw2v: Tensor = None, Tv2w: Tensor = None, inv=False):
    if Tw2v is None:
        Tw2v = Tv2w.inverse()
    Tw2v[..., 1:3, :3] *= -1
    Tw2v[..., :3, 3] *= -1
    return Tw2v, torch.inverse(Tw2v)


def convert_coord_system(T: Tensor, src='opengl', dst='opengl', inverse=False) -> Tensor:
    """ convert coordiante system from <source> to <goal>: p_dst = M @ p_src 适用于Tw2v或Tv2w

    原理: Tw2v: 对任意点p, 先从dst转到src坐标系，再从src坐标系转到view坐标系
    Args:
        T: transformation matrix with shape [..., 4, 4], can be Tw2v or Tv2w
        src: the source coordinate system, must be blender, colmap, opencv, llff, PyTorch3d, opengl
        dst: the destination coordinate system
        inverse: inverse apply (used when T is Tv2w)
    Returns:
        Tensor: converted transformation matrix
    """
    src = coordinate_system[src.lower()]
    dst = coordinate_system[dst.lower()]
    if src == dst:
        return T
    M = T.new_tensor(_convert_matrix[src][dst] if inverse else _convert_matrix[dst][src])
    if dst == 'opencv' or src == 'opencv':  # due to left <-> right-hand clip space
        T = T @ M if inverse else M @ T
    else:
        T = M @ T if inverse else T @ M
    return T


def convert_coord_system_matrix(T: Tensor, src='opengl', dst='opengl') -> Tensor:
    """ convert coordiante system from <source> to <goal>: p_dst = M @ p_src 适用于旋转矩阵
    Args:
        T: transformation matrix with shape [..., 4, 4], can be Tw2v or Tv2w
        src: the source coordinate system, must be blender, colmap, opencv, llff, PyTorch3d, opengl
        dst: the destination coordinate system
    Returns:
        Tensor: converted transformation matrix
    """
    src = coordinate_system[src.lower()]
    dst = coordinate_system[dst.lower()]
    if src == dst:
        return T
    return T.new_tensor(_convert_matrix[src][dst]) @ T @ T.new_tensor(_convert_matrix[dst][src])


def convert_coord_system_points(points: Tensor, src='opengl', dst='opengl') -> Tensor:
    if src == 'opengl':
        if dst == 'opengl':
            return points
        elif dst == 'blender':
            return torch.stack([points[..., 0], -points[..., 2], points[..., 1]], dim=-1)
        elif dst == 'opencv':
            return torch.stack([points[..., 0], -points[..., 1], -points[..., 2]], dim=-1)
        elif dst == 'llff':
            return torch.stack([-points[..., 1], points[..., 0], points[..., 2]], dim=-1)
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
            return torch.stack([-points[..., 2], points[..., 0], -points[..., 1]], dim=-1)
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
            return torch.stack([points[..., 1], points[..., 0], -points[..., 2]], dim=-1)
        else:
            return torch.stack([points[..., 2], -points[..., 1], points[..., 0]], dim=-1)
    elif src == 'llff':
        if dst == 'opengl':
            return torch.stack([points[..., 1], -points[..., 0], points[..., 2]], dim=-1)
        elif dst == 'blender':
            return torch.stack([points[..., 1], -points[..., 2], -points[..., 0]], dim=-1)
        elif dst == 'opencv':
            return torch.stack([points[..., 1], points[..., 0], -points[..., 2]], dim=-1)
        elif dst == 'llff':
            return points
        else:
            return torch.stack([-points[..., 2], -points[..., 0], points[..., 1]], dim=-1)
    elif src == 'pytorch3d':
        if dst == 'opengl':
            return torch.stack([points[..., 2], points[..., 1], -points[..., 0]], dim=-1)
        elif dst == 'blender':
            return torch.stack([points[..., 2], points[..., 0], points[..., 1]], dim=-1)
        elif dst == 'opencv':
            return torch.stack([points[..., 2], -points[..., 1], points[..., 0]], dim=-1)
        elif dst == 'llff':
            return torch.stack([-points[..., 1], points[..., 2], -points[..., 0]], dim=-1)
        else:
            return points
    else:
        raise NotImplementedError(f"src={src}, dst={dst}")


_coord_spherical_to = coord_trans_opengl.coord_spherical_to
_coord_to_spherical = coord_trans_opengl.coord_to_spherical
_look_at = coord_trans_opengl.look_at
_look_at_get = coord_trans_opengl.look_at_get
_camera_intrinsics = coord_trans_opengl.camera_intrinsics
_perspective = coord_trans_opengl.perspective
_perspective2 = coord_trans_opengl.perspective2
_point2pixel = coord_trans_opengl.point2pixel
_ndc2pixel = coord_trans_opengl.ndc2pixel


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


def look_at_get(Tv2w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """return eye, at, up"""
    return _look_at_get(Tv2w)


def camera_intrinsics(
    focal: Union[float, None, Tensor, Sequence[float]] = None,
    cx_cy: Union[float, None, Tensor, Sequence[float]] = None,
    size: Union[float, None, Tensor, Sequence[float]] = None,
    fov: Union[float, None, Tensor, Sequence[float]] = np.pi,
    inv: bool = False,
    **kwargs
) -> Tensor:
    """生成相机内参K/Tv2s, 请注意坐标系
    .---> u
    |
    ↓
    v
    Args:
        focal: focal length, shape: [..., 1] or shape: [..., 2]
        cx_cy: principal points, shape: [..., 2]
        size: image size
        fov: filed of view, shape: [..., 1] or [..., 2]
        inv: if True, return Ts2v, else Tv2s
        kwargs: the kwargs for output matrix
    Returns:
         Tensor, shape: [..., 3, 3]
    """
    return _camera_intrinsics(focal, cx_cy, size, fov, inv, **kwargs)


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


def perspective2(size, focals=None, FoV=None, pp=None, n=0.1, f=1000.0, device=None):
    """透视投影矩阵

    Args:
        size (int | Tuple[int, int]): (W, H)
        focals (float | list | tuple | Tensor): focal length, shape: [...], [..., 1], [..., 2]
        FoV (float | list | tuple | Tensor): filed of view, radians (e.g, pi), shape: [...], [..., 1], [..., 2]
        pp (float | list | tuple | Tensor): position of center/principal points, shape: [...], [..., 1], [..., 2]
        n (float | list | tuple | Tensor): near. Defaults to 0.1.
        f (float | list | tuple | Tensor): far. Defaults to 1000.0.
        device: Defaults to None.
    Returns:
        Tensor: 透视投影矩阵
    """
    return _perspective2(size, focals, FoV, pp, n, f, device)


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
    return _point2pixel(points, Tw2v, Tv2s, Tv2c, Tw2c, size)


def ndc2pixel(ndc: Tensor, size: Tuple[int, int]):
    return _ndc2pixel(ndc, size)


def set_coord_system(coord):
    coord = coordinate_system[coord.lower()]
    global COORDINATE_SYSTEM
    COORDINATE_SYSTEM = coord
    global _coord_spherical_to, _coord_to_spherical, _look_at, _look_at_get
    global _perspective, _perspective2, _ortho, _camera_intrinsics
    global _point2pixel, _ndc2pixel
    if coord == 'opengl':
        _coord_spherical_to = coord_trans_opengl.coord_spherical_to
        _coord_to_spherical = coord_trans_opengl.coord_to_spherical
        _look_at = coord_trans_opengl.look_at
        _look_at_get = coord_trans_opengl.look_at_get
        _camera_intrinsics = coord_trans_opengl.camera_intrinsics
        _perspective = coord_trans_opengl.perspective
        _perspective2 = coord_trans_opengl.perspective2
        _point2pixel = coord_trans_opengl.point2pixel
        _ndc2pixel = coord_trans_opengl.ndc2pixel
    elif coord == 'opencv':
        _coord_spherical_to = coord_trans_opencv.coord_spherical_to
        _coord_to_spherical = coord_trans_opencv.coord_to_spherical
        _look_at = coord_trans_opencv.look_at
        _look_at_get = coord_trans_opencv.look_at_get
        _camera_intrinsics = coord_trans_opencv.camera_intrinsics
        _perspective = coord_trans_opencv.perspective
        _perspective2 = coord_trans_opencv.perspective2
        _point2pixel = coord_trans_opencv.point2pixel
        _ndc2pixel = coord_trans_opencv.ndc2pixel
    else:
        raise NotImplementedError(f"coord {coord} not supported")
    logging.info(f"[red]set coordinate system to {coord}")


def get_coord_system():
    global COORDINATE_SYSTEM
    return COORDINATE_SYSTEM


def test_coord_system():
    from my_ext.ops_3d.xfm import apply
    from my_ext.ops_3d.camera import compute_camera_align
    from my_ext.utils import set_printoptions
    set_printoptions()
    torch.set_default_dtype(torch.float64)
    """
    OpenGL/Vulkan:     Blender:       COLMAP/OpenCV:    LLFF     Pytorch3D
        y              z                   z                     y
        ↑              ↑                 ↗                       ↑
        |              |   y            .-------> x   .-----> y  |   x
        |              | ↗              |           ↙ |          | ↗
        .-------> x    .-------> x      |          z  |          .------> z
       ↙                                ↓             ↓
      z                                 y             x
    """
    p_list = [
        ('opengl', torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1., 2., 3.]])),
        ('blender', torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0], [1., -3., 2.]])),
        ('opencv', torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1], [1., -2., -3.]])),
        ('llff', torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1], [-2., 1., 3.]])),
        ('pytorch3d', torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0], [-3., 2, 1]])),
    ]
    print()

    from my_ext.ops_3d import opengl
    Tw2v = opengl.look_at(torch.randn(3), torch.randn(3), torch.randn(3))
    Tv2w = torch.inverse(Tw2v)

    for src, src_p in p_list:
        for dst, dst_p in p_list:
            print(src, dst)
            assert (dst_p == convert_coord_system_points(src_p, src, dst)).all()
            # dst_p_ = apply(src_p, convert_coord_system(I, src, dst, inverse=False))
            # dst_p_ = apply(src_p, src_p.new_tensor(convert_matrix[src][dst]))
            # print((dst_p_ - dst_p).abs().max())
            #
            src_p2 = apply(src_p, Tw2v)
            dst_p2 = convert_coord_system_points(src_p2, src, dst)
            Tw2v_dst_gt = torch.eye(4)
            s, R, t = compute_camera_align(dst_p, dst_p2)
            assert (s - 1).abs().item() < 1e-10
            Tw2v_dst_gt[:3, :3] = R * s
            Tw2v_dst_gt[:3, 3] = t
            dst_p2_ = apply(dst_p, Tw2v_dst_gt)
            assert (dst_p2_ - dst_p2).abs().max().item() < 1e-6
            # print(Tw2v_dst_gt @ Tw2v.inverse())
            Tw2v_dst = convert_coord_system_matrix(Tw2v.clone(), src, dst)
            Tw2v_error = (Tw2v_dst - Tw2v_dst_gt).abs().max()
            assert Tw2v_error < 1e-6
            print(f'Tw2v error: {Tw2v_error:.4e}')
            M = Tw2v.new_tensor(_convert_matrix[src][dst])
            M_ = Tw2v.new_tensor(_convert_matrix[dst][src])
            # print(M @ M.T)
            Tv2w_dst = convert_coord_system_matrix(Tv2w, src, dst)

            print((Tw2v @ Tv2w - torch.eye(4)).abs().max())
            print(((M @ M_) - torch.eye(4)).abs().max())
            print((Tv2w_dst @ Tw2v_dst - torch.eye(4)).abs().max())
            Tv2w_error = (Tv2w_dst - torch.inverse(Tw2v_dst_gt)).abs().max()
            assert (Tv2w_dst - torch.inverse(Tw2v_dst_gt)).abs().max() < 1e-6
            print(f'Tv2w error: {Tv2w_error:.4e}')
            # Tw2v_dst2 = Tw2v.new_tensor(convert_matrix[src][dst]) @ Tw2v @ Tw2v.new_tensor(convert_matrix[dst][src])
            # print(Twass2v_dst - Tw2v_dst_gt)
            # print(Tw2v_dst, Tw2v)
            # print(Tw2v_dst[:3, :3] @ Tw2v_dst[:3, :3].T)
