"""转变点或向量"""
import torch
import torch.utils.cpp_extension
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from my_ext._C import get_C_function, try_use_C_extension

__all__ = ['xfm', 'xfm_vectors', 'apply', 'pixel2points']


class _xfm_func(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, points, matrix, isPoints=True, to_homo=False):  # noqa
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        ctx.to_homo = to_homo
        return get_C_function('xfm_fwd')(points, matrix, isPoints, to_homo)

    @staticmethod
    @torch.autograd.function.once_differentiable
    @custom_bwd
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        grad_points, grad_matrix = get_C_function('xfm_bwd')(
            dout, points, matrix, ctx.isPoints, ctx.to_homo, ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        )
        return grad_points, grad_matrix, None, None


@try_use_C_extension(_xfm_func.apply, "xfm_fwd", 'xfm_bwd')
def _xfm(points: Tensor, matrix: Tensor, is_points=True, to_homo=False) -> Tensor:
    dim = points.shape[-1]
    if dim + 1 == matrix.shape[-1]:
        points = torch.constant_pad_nd(points, (0, 1), 1.0 if is_points else 0.0)
    else:
        to_homo = False
    if is_points:
        out = torch.matmul(points, torch.transpose(matrix, -1, -2))
    else:
        out = torch.matmul(points, torch.transpose(matrix, -1, -2))
    if not to_homo:
        out = out[..., :dim]
    return out


def xfm(points: Tensor, matrix: Tensor, homo=False) -> Tensor:
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [..., num_vertices, 3] or [..., num_vertices, 4]
        matrix: A 4x4 transform matrix with shape [..., 4, 4] or [4, 4]
        homo: convert output to homogeneous
    Returns:
        Tensor: Transformed points in homogeneous 4D with shape [..., num_vertices, 3/4]
    '''
    return _xfm(points, matrix, True, homo)


def apply(points: Tensor, matrix: Tensor, homo=False, is_points=True) -> Tensor:
    """Transform points p' = M @ p.
    Args:
       points: Tensor containing 3D points with shape [...,  C]
       matrix: A transform matrix with shape [..., C, C] or [..., C+1, C+1]
       homo: convert output to homogeneous
       is_points: is points or vector?
    Returns:
       Tensor: Transformed points with shape [..., C] or [..., C+1]
    """
    dim = points.shape[-1]
    if dim + 1 == matrix.shape[-1]:
        points = torch.constant_pad_nd(points, (0, 1), 1.0 if is_points else 0.0)
    else:
        homo = False
    # out = torch.einsum('...ij,...j->...i', matrix, points)
    out = torch.sum(matrix * points[..., None, :], dim=-1)
    if not homo:
        out = out[..., :dim]
    return out


def xfm_vectors(vectors: Tensor, matrix: Tensor, to_homo=True) -> Tensor:
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [..., num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [..., 4, 4] or [4, 4]

    Returns:
        Tensor: Transformed vectors in homogeneous 4D with shape [..., num_vertices, 3/4].
    '''
    return _xfm(vectors, matrix, False, to_homo)


def pixel2points(
    depth: Tensor, Tv2s: Tensor = None, Ts2v: Tensor = None, Tw2v: Tensor = None, Tv2w: Tensor = None,
    pixel: Tensor = None
) -> Tensor:
    """convert <pixel, depth> to 3D point"""
    if pixel is None:
        H, W = depth.shape[-2:]
        pixel = torch.stack(
            torch.meshgrid(
                torch.arange(W, device=depth.device, dtype=depth.dtype),
                torch.arange(H, device=depth.device, dtype=depth.dtype),
                indexing='xy'
            ), dim=-1
        )  # shape: [H, W, 2]
    xyz = torch.cat([pixel, torch.ones_like(pixel[..., :1])], dim=-1) * depth[..., None]  # [..., H, W, 3]
    xyz = apply(xyz, Tv2s.inverse() if Ts2v is None else Ts2v)
    if Tv2w is not None:
        xyz = apply(xyz, Tv2w)
    elif Tw2v is not None:
        xyz = apply(xyz, Tw2v.inverse())
    return xyz
