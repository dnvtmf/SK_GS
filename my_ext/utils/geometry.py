from typing import Optional, Union

import torch
from torch import Tensor

__all__ = ['distance_point_to_line', 'quanernion_to_matrix', 'point_to_line']


def point_to_line(points: Tensor, A: Tensor, B: Tensor = None):
    """点point到直线(A, B) 的垂足"""
    if B is None:
        A, B = A.unbind(-2)
    D = B - A
    k = torch.sum((points - A) * D, dim=-1, keepdim=True) / D.pow(2).sum(-1, keepdim=True)
    return k * D + A


def distance_point_to_line(point: Tensor, line_a: Tensor, line_b: Tensor):
    """点point到直线(line_a, line_b)的距离, shape[..., 2] or shape: [..., 3] """
    if point.shape[-1] == 2:
        # 直线方程: Ax + By + C = 0
        A = line_b[..., 1] - line_a[..., 1]
        B = line_a[..., 0] - line_b[..., 0]
        C = -line_a[..., 0] * A - line_a[..., 1] * B
        # 距离公式： d = |Ax+By+C|/sqrt(A^2 + B^2)
        d = torch.abs(A * point[..., 0] + B * point[..., 1] + C) * torch.rsqrt(A.square() + B.square())
        return d
    else:
        return torch.cross(point - line_a, point - line_b).norm(dim=-1) / (line_b - line_a).norm(dim=-1)


def line_interset_sphere(A: Tensor, B: Optional[Tensor], C: Tensor, R: Union[Tensor, float], D: Tensor = None):
    """ 直线AB与球(C, R)的交点， D=B-A
    """
    D = B - A if D is None else D  # line direct
    C = A - C
    a = D.pow(2).sum(-1)
    b = 2. * (D * C).sum(-1)
    c = C.pow(2).sum(-1) - R ** 2
    # 解一元二次方程
    delta = b * b - 4 * a * c + 1e-9
    mask = delta > 0
    delta = delta.clamp(0).sqrt()
    p1 = (-b + delta) / (2 * a + 1e-6)
    p2 = (-b - delta) / (2 * a + 1e-6)
    points = torch.stack([torch.minimum(p1, p2), torch.maximum(p1, p2)], dim=-1)
    return points, mask


def project_3d_points_to_2d(points_3d, R, T):
    pass


def quanernion_to_matrix(w: float, x: float, y: float, z: float, device=None) -> Tensor:
    """四元数转旋转矩阵"""
    return torch.tensor([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
    ], dtype=torch.float32, device=device)


def test():
    from my_ext import utils
    print()
    P = torch.tensor([[0, 0], [1, 1], [1, 1], [44.8920, -10.2155]])  # point
    A = torch.tensor([[0, 1], [0, 0], [0, 0], [44.8920, -10.2155]])  # line A
    B = torch.tensor([[1, 0], [1, 0], [0, 1], [49.0228, -26.5623]])  # line B
    print('point:', utils.show_shape(P), 'line:', utils.show_shape(A, B))
    d = distance_point_to_line(P, A, B)
    print('2D:', d)
    P, A, B = [torch.constant_pad_nd(x, (0, 1)) for x in [P, A, B]]
    print('point:', utils.show_shape(P), 'line:', utils.show_shape(A, B))
    d = distance_point_to_line(P, A, B)
    print('3D:', d)


def test_2():
    P = torch.rand((3, 3))
    A = torch.rand((3, 3))
    B = torch.rand((3, 3))
    N = point_to_line(P, A, B)
    from my_ext.utils import vis3d

    vis3d.add_points(P, color=(1, 0, 0))
    vis3d.add_points(A, color=(0, 1, 0))
    vis3d.add_points(B, color=(0, 1, 0))
    vis3d.add_points(N, color=(0, 0, 1))
    vis3d.add_lines(torch.stack([A, B], dim=-2))
    vis3d.add_lines(torch.stack([P, N], dim=-2), color=(0.5, 0.5, 0.5))
    vis3d.show()
