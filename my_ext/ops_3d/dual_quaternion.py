"""对偶四元数相关函数
对偶四元数(Dual quaternions)： dq = a + bε, 其中ε x 1= ε, ε x ε = 0
References:
- B. Kenwright, “A Beginners Guide to Dual-Quaternions”
- https://en.wikipedia.org/wiki/Dual_quaternion
-
"""
import torch
from torch import Tensor

from . import quaternion


def conj(dq: Tensor) -> Tensor:
    """dq = a + bε, dq' = a' + b'ε """
    return torch.cat([quaternion.conj(dq[..., :4]), quaternion.conj(dq[..., 4:])], dim=-1)


def dual_conj(dq: Tensor) -> Tensor:
    """dq = a + bε, dq' = a - bε """
    return torch.cat([dq[..., :4], -dq[..., 4:]], dim=-1)


def complex_conj(dq: Tensor) -> Tensor:
    """dq = a + bε, dq' = a' - b'ε """
    return torch.cat([quaternion.conj(dq[..., :4]), -quaternion.conj(dq[..., 4:])], dim=-1)


def add(dq1: Tensor, dq2: Tensor) -> Tensor:
    """dq1 = a + bε, dq2 = c + dε => dq1 + dq2 = (a+c) + (b+d)ε"""
    # return torch.cat([quaternion.add(dq1[..., :4], dq2[..., :4]), quaternion.add(dq1[..., 4:], dq2[..., 4:])], dim=-1)
    return dq1 + dq2


def sub(dq1: Tensor, dq2: Tensor) -> Tensor:
    """dq1 = a + bε, dq2 = c + dε => dq1 + dq2 = (a+c) - (b+d)ε"""
    # return torch.cat([quaternion.add(dq1[..., :4], -dq2[..., :4]),
    # quaternion.add(dq1[..., 4:], -dq2[..., 4:])], dim=-1)
    return dq1 - dq2


def mul(dq1: Tensor, dq2: Tensor) -> Tensor:
    """dq1 = r1 + bε, dq2 = r2 + dε => dq1 * dq2 = ac + (ad + bc)ε"""
    if dq1.ndim == dq2.ndim + 1:
        dq2 = dq2[..., None]
    elif dq2.ndim == dq1.ndim + 1:
        dq1 = dq1[..., None]
    elif dq1.shape[-1] == 1 or dq2.shape[-1] == 1:
        # k * dq = ka + kbε
        return dq1 * dq2
    r1, d1 = dq1.split(4, dim=-1)
    r2, d2 = dq2.split(4, dim=-1)
    p = quaternion.mul(r1, r2)
    q = quaternion.mul(r1, d2) + quaternion.mul(d1, r2)
    return torch.cat([p, q], dim=-1)


def dot(dq1: Tensor, dq2: Tensor) -> Tensor:
    return torch.dot(dq1[..., :4], dq2[..., :4])


def cross(dq1: Tensor, dq2: Tensor) -> Tensor:
    r1, d1 = dq1.split(4, dim=-1)
    r2, d2 = dq2.split(4, dim=-1)
    return torch.cat([quaternion.cross(r1, r2), quaternion.cross(r1, d2) + quaternion.cross(d1, r2)], dim=-1)


def norm(dq: Tensor, keepdim=True) -> Tensor:
    """dq = a + bε => |dq| = |a| """
    return torch.norm(dq[..., :4], dim=-1, keepdim=keepdim)


def dual_norm(dq: Tensor) -> Tensor:
    """dq = r + dε => |dq| = |r| + (d * r' + r * d')/2|r|ε
    |dq|^2 = dq * dq'
    """
    # return mul(dq, conj(dq))
    r, d = dq.split(4, dim=-1)
    rn = dq[..., :4].norm(p=2, dim=-1, keepdim=True)
    n = torch.zeros_like(dq)
    n[..., 3:4] = rn
    n[..., 4:] = (quaternion.mul(d, quaternion.conj(r)) + quaternion.mul(r, quaternion.conj(d))) / (2 * rn)
    return n


# def normalize(dq: Tensor) -> Tensor:
#     # a, b = dq.split(4, dim=-1)
#     # n = a.norm(p=2, dim=-1, keepdim=True) + 1e-10
#     # b_ = quaternion.mul(a, quaternion.conj(b)) + quaternion.mul(b, quaternion.conj(a))
#     # return torch.cat([a, b_ * 0.5], dim=-1) / n
#     return dq / norm(dq)


def inv(dq: Tensor):
    """dq = a + bε => dq^-1 = dq* / |dq|^2 = (a - bε) / a^2 """
    # return conj(dq) / norm(dq).square()
    r, d = dq.split(4, dim=-1)
    r_ = quaternion.inv(r)
    return torch.cat([r_, -quaternion.mul(r_, quaternion.mul(d, r_))], dim=-1)


def div(dq1: Tensor, dq2: Tensor) -> Tensor:
    """dq1 = a + bε, dq2 = c + dε => dq1 / dq2 = dq1 * dq2^-1 = (a + bε)(c - dε) / c^2"""
    return mul(dq1, inv(dq2))


def from_tq(q: Tensor, t: Tensor = None) -> Tensor:
    dq = q.new_zeros(*q.shape[:-1], 8)
    if q.shape[-1] == 7:
        t, q = q.split([3, 4], dim=-1)
    dq[..., :4] = q
    if t is not None:
        t_ = torch.cat([t, torch.zeros_like(t[..., :1])], dim=-1)
        print(t_.shape, q.shape)
        dq[..., 4:] = 0.5 * quaternion.mul(t_, q)
    return dq


def to_tq(dq: Tensor) -> Tensor:
    r, d = dq.split(4, dim=-1)
    t = 2. * quaternion.mul(d, quaternion.conj(r))
    return torch.cat([t[..., :3], r], dim=-1)


def xfm(dq: Tensor, points: Tensor) -> Tensor:
    points_dq = points.new_zeros(*points.shape[:-1], 8)
    points_dq[..., 3] = 1
    points_dq[..., 4:7] = points
    return mul(dq, mul(points_dq, complex_conj(dq)))[..., 4:7]


def is_identity(dq: Tensor, eps=1e-7):
    check_real = (norm(dq, False) - 1).abs() < eps
    check_dual = torch.einsum('...i,...i->...', dq[..., :4], dq[..., 4:]).abs() < eps
    return torch.logical_and(check_real, check_dual)


def test():
    from my_ext import utils, ops_3d
    from my_ext.ops_3d import rigid
    utils.set_printoptions()
    print()
    N = 10
    q = quaternion.normalize(torch.randn(N, 4))
    t = torch.randn(N, 3)
    dq = from_tq(q, t)
    tq = to_tq(dq)
    print('t error:', (t - tq[..., :3]).abs().max())
    print('q error:', (q - tq[..., 3:]).abs().max())
    print(is_identity(dq))
    dq_inv = inv(dq)
    i = mul(dq, dq_inv)
    print(i[:2])
    print(is_identity(i))
    # print(norm(i))
    # print(torch.einsum('...i,...i->...', i[..., :4], i[..., 4:]))
    # print(mul(dual_norm(dq), dual_norm(dq)))
    # print(mul(dq, conj(dq)))

    T = rigid.quaternion_to_Rt(q, t)
    points = torch.randn(N, 3)
    # print(utils.show_shape(T, points))
    points_1 = ops_3d.apply(points, T)
    points_2 = xfm(dq, points)
    # print(points_1)
    # print(points_2)
    print('xfm error:', (points_1 - points_2).abs().max())
