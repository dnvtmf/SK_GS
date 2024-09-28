import numpy as np
import torch
from torch import Tensor
from my_ext._C import get_C_function, try_use_C_extension, get_python_function

from my_ext.utils.utils import extend_list, to_list

__all__ = [
    'make_3d_grid', 'normalize', 'to_homo', 'dot', 'reflect', 'to_4x4'
]


def make_3d_grid(n, delta=0.5, sacle=1., offset=0., device=None, stack_last=True) -> Tensor:
    """ 生成3D网格的坐标
    
    先生成n**3的立方体网格的坐标，即 [0, 0,0], ..., [n-1, n-1, n-1],
    再将这些坐标 变换为 (points + delta) * sacle + offset

    返回形状为 [n, n, n, 3] 或 [3, n, n, n]的坐标
    """
    if isinstance(n, (np.ndarray, Tensor)):
        n = n.tolist()
    n = extend_list(to_list(n), 3)
    points = torch.stack(
        torch.meshgrid(
            torch.arange(n[0], device=device),
            torch.arange(n[1], device=device),
            torch.arange(n[2], device=device),
            indexing='ij'
        ),
        dim=-1 if stack_last else 0
    )
    points = (points.float() + delta) * sacle + offset
    return points


class _safe_normalize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, dim=-1, eps=1e-20):  # noqa
        ctx.save_for_backward(x)
        ctx.eps = eps
        ctx.dim = dim
        return get_C_function('safe_normalize_forward')(x, dim, eps)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        x, = ctx.saved_tensors
        return get_C_function("safe_normalize_backward")(x, grad_outputs[0], ctx.dim, ctx.eps), None, None


@try_use_C_extension(_safe_normalize_func.apply, "safe_normalize_forward", "safe_normalize_backward")
def _safe_normalize(x: Tensor, dim=-1, eps=1e-20):
    # return F.normalize(x, dim=dim, eps=eps)
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, dim, keepdim=True), min=eps))


def normalize(x: Tensor, dim=-1, eps=1e-20):
    return _safe_normalize(x, dim, eps)


def to_homo(x: Tensor) -> Tensor:
    """get homogeneous coordinates"""
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def dot(x: Tensor, y: Tensor, keepdim=True, dim=-1) -> Tensor:
    """Computes the dot product of two tensors in the given dimension dim"""
    return torch.sum(x * y, dim, keepdim=keepdim)
    # return torch.linalg.vecdot(x, y, dim=dim)


def bmv(mat: Tensor, vec: Tensor) -> Tensor:
    """Performs a batch matrix-vector product of the matrix mat and the vector vec """
    return torch.sum(vec[..., None, :] * mat, -1, keepdim=False)


def reflect(l: Tensor, n: Tensor) -> Tensor:
    """给定法线n和入射方向l, 计算反射方向"""
    return 2 * dot(l, n) * n - l


def to_4x4(T: Tensor):
    """convert T from 3x3 or 3x4 matrices to 4x4 matrics"""
    if T.shape[-2:] == (4, 4):
        return T
    T_ = torch.eye(4, dtype=T.dtype, device=T.device).expand(*T.shape[:-2], 4, 4).clone()
    T_[..., :T.shape[-2], :T.shape[-1]] = T
    return T_


def test():
    points = torch.randn(2, 1, 4, 3)
    M = torch.randn(3, 4, 4, 3)
    assert dot(points[..., None, :], M, keepdim=False).shape == (2, 3, 4, 4)
    assert bmv(M, points).shape == (2, 3, 4, 4)
    from .xfm import xfm
    assert xfm(points, M).shape == (2, 3, 4, 4)


def test_normalize():
    x = torch.randn((100, 3))
    x[:10] *= 1e-20
    x.requires_grad_(True)
    y1 = _safe_normalize(x)
    g1 = torch.autograd.grad(y1, x, torch.ones_like(x))[0].clone()

    x_ = x.detach().clone().cuda().requires_grad_(True)
    y2 = _safe_normalize(x_)
    g2 = torch.autograd.grad(y2, x_, torch.ones_like(x_))[0].clone().cpu()
    y2 = y2.cpu()

    y3 = get_python_function('_safe_normalize')(x)
    g3 = torch.autograd.grad(y3, x, torch.ones_like(x))[0].clone()
    print((y1 - y3).abs().max())
    print((y2 - y3).abs().max())
    print((g1 - g3).abs().max())
    print((g2 - g3).abs().max())
    # print(g1[:10], g2[:10], g3[:10])
    x = x[10:].double().detach().cuda().requires_grad_(True)
    test = torch.autograd.gradcheck(_safe_normalize, x, eps=1e-6, atol=1e-4)
    print(test)
