"""转变点或向量"""
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.utils.cpp_extension
import pytest
from torch.cuda.amp import custom_bwd, custom_fwd

from my_ext._C import get_C_function, try_use_C_extension, get_python_function

__all__ = ['xfm', 'xfm_vectors', 'apply']


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
            dout, points, matrix, ctx.isPoints, ctx.to_homo, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
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
        (Tensor):  containing 2D pixel with shape [..., 2]
    """
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # to homo
    if Tw2v is not None and Tv2s is not None:
        points = points @ Tw2v.transpose(-1, -2)
        print(points)
        points = points[..., :3] @ Tv2s.transpose(-1, -2)
        print(points)
        pixel = points[..., :2] / points[..., 2:3]
        return pixel
    else:
        if Tw2c is None:
            assert Tw2v is not None and Tv2c is not None
            Tw2c = Tv2c @ Tw2v
        ndc = points @ Tw2c.transpose(-1, -2)
        pixel = torch.zeros_like(ndc[..., :2])
        pixel[..., 0] = (1 + ndc[..., 0] / ndc[..., 3]) * 0.5 * size[0]
        pixel[..., 1] = (1 - ndc[..., 1] / ndc[..., 3]) * 0.5 * size[1]
        return pixel


@pytest.mark.parametrize("C,to_homo", [(3, False), (4, False), (3, True)])
def test_xfm_points(C, to_homo):
    print()
    from my_ext.utils.test_utils import get_run_speed
    BATCH = 800
    RES = 1024
    DTYPE = torch.float32

    def relative_loss(name, ref, cuda):
        ref = ref.float()
        cuda = cuda.float()
        print(name, (torch.abs(ref - cuda).max() / ref.abs().max()).item())

    points_cuda = torch.rand(BATCH, RES, C, dtype=DTYPE, device='cuda', requires_grad=True)
    points_ref = points_cuda.clone().detach().requires_grad_(True)
    points_py2 = points_cuda.clone().detach().requires_grad_(True)
    mtx_cuda = torch.rand(BATCH, 4, 4, dtype=DTYPE, device='cuda', requires_grad=True)
    mtx_ref = mtx_cuda.clone().detach().requires_grad_(True)
    mtx_py2 = mtx_cuda[:, None].clone().detach().requires_grad_(True)
    grad = torch.rand(BATCH, RES, C + to_homo, dtype=DTYPE, device='cuda')

    py_func = lambda p, T: get_python_function('_xfm')(p, T, True, to_homo)
    cuda_func = lambda p, T: _xfm_func.apply(p, T, True, to_homo)
    py2_func = lambda p, T: apply(p, T, to_homo, True)

    ref_out = py_func(points_ref, mtx_ref)
    torch.autograd.backward(ref_out, grad)

    cuda_out = cuda_func(points_cuda, mtx_cuda)
    torch.autograd.backward(cuda_out, grad)

    print("-" * 40, f"{points_cuda.shape} --> {ref_out.shape}", '-' * 40)
    relative_loss("forward:", ref_out, cuda_out)
    relative_loss("grad points:", points_ref.grad, points_cuda.grad)
    relative_loss("grad mtx:", mtx_cuda.grad, mtx_ref.grad)

    py2_out = py2_func(points_py2, mtx_py2)
    torch.autograd.backward(py2_out, grad)

    relative_loss("py2 forward:", ref_out, py2_out)
    relative_loss("py2 grad points:", points_ref.grad, points_py2.grad)
    relative_loss("py2 grad mtx:", mtx_ref.grad[:, None], mtx_py2.grad)

    get_run_speed((points_ref, mtx_ref), torch.randn_like(ref_out),
        py_func=py_func, cuda_func=cuda_func)
    get_run_speed((points_py2, mtx_py2), torch.randn_like(ref_out),
        py2_func=py2_func)


def test_xfm_vectors():
    BATCH = 8
    RES = 1024
    DTYPE = torch.float32

    def relative_loss(name, ref, cuda):
        ref = ref.float()
        cuda = cuda.float()
        print(name, torch.max(torch.abs(ref - cuda) / torch.abs(ref)).item())

    points_cuda = torch.rand(BATCH, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    points_ref = points_cuda.clone().detach().requires_grad_(True)
    points_cuda_p = points_cuda.clone().detach().requires_grad_(True)
    points_ref_p = points_cuda.clone().detach().requires_grad_(True)
    mtx_cuda = torch.rand(BATCH, 4, 4, dtype=DTYPE, device='cuda', requires_grad=False)
    mtx_ref = mtx_cuda.clone().detach().requires_grad_(True)
    grad = torch.rand(BATCH, RES, 4, dtype=DTYPE, device='cuda', requires_grad=True)

    py_func = get_python_function('_xfm')
    cu_func = xfm_vectors

    ref_out = py_func(points_ref.contiguous(), mtx_ref, False, False)
    torch.autograd.backward(ref_out, grad[..., :3])

    cuda_out = cu_func(points_cuda.contiguous(), mtx_cuda, False)
    torch.autograd.backward(cuda_out, grad[..., :3])

    ref_out_p = py_func(points_ref_p.contiguous(), mtx_ref, False, True)
    torch.autograd.backward(ref_out_p, grad)

    cuda_out_p = cu_func(points_cuda_p.contiguous(), mtx_cuda, True)
    torch.autograd.backward(cuda_out_p, grad)

    print("-------------------------------------------------------------")

    relative_loss("res:", ref_out, cuda_out)
    relative_loss("points:", points_ref.grad, points_cuda.grad)
    relative_loss("points_p:", points_ref_p.grad, points_cuda_p.grad)
