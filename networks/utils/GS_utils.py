import math

import torch
from torch import Tensor

from my_ext import get_C_function, try_use_C_extension, ops_3d
from my_ext.ops_3d.lietorch import SO3


class _compute_cov3D(torch.autograd.Function):
    _forward = get_C_function('gs_compute_cov3D_forward')
    _backward = get_C_function('gs_compute_cov3D_backward')

    @staticmethod
    def forward(ctx, *inputs):
        scaling, scaling_modifier, rotation = inputs
        cov3D = _compute_cov3D._forward(rotation, scaling)
        ctx.save_for_backward(rotation, scaling)
        return cov3D

    @staticmethod
    def backward(ctx, *grad_outputs):
        rotation, scaling = ctx.saved_tensors
        grad_rotation, grad_scaling = _compute_cov3D._backward(rotation, scaling, grad_outputs[0])
        return grad_scaling, None, grad_rotation


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    if r.shape[-1] == 3:
        return SO3.exp(r).matrix()[..., :3, :3]
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device)
    x, y, z, r = q.unbind(-1)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


@try_use_C_extension(_compute_cov3D.apply, "gs_compute_cov3D_forward", "gs_compute_cov3D_backward")
def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class _compute_cov2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs, **kwargs):
        cov3D, mean, viewmatrix, fx, fy, tx, ty = inputs
        ctx.focal_fov = (fx, fy, tx, ty)
        cov2D = get_C_function('gs_compute_cov2D_forward')(cov3D, mean, viewmatrix, fx, fy, tx, ty)
        ctx.save_for_backward(cov3D, mean, viewmatrix)
        return cov2D

    @staticmethod
    def backward(ctx, *grad_outputs):
        cov3D, mean, viewmatrix = ctx.saved_tensors
        grad_cov3D, grad_mean, grad_viewmatrix = get_C_function('compute_cov2D_backward')(
            cov3D, mean, viewmatrix, *ctx.focal_fov, grad_outputs[0])
        return grad_cov3D, grad_mean, None, None, None, None, None


def compute_cov2D(points: Tensor, cov3D: Tensor, Tw2v: Tensor, focal_x, focal_y, tan_fovx, tan_fovy):
    P = points.shape[0]
    p = ops_3d.xfm(points, Tw2v)
    limit_x = 1.3 * tan_fovx
    limit_y = 1.3 * tan_fovy
    x, y, z = p.unbind(-1)
    x = (x / z).clamp(-limit_x, limit_x) * z
    y = (y / z).clamp(-limit_y, limit_y) * z
    v0 = torch.zeros_like(z)
    J = torch.stack([
        focal_x / z, v0, -(focal_x * x) / (z * z),
        v0, focal_y / z, -(focal_y * y) / (z * z),
        v0, v0, v0
    ], dim=-1).view(P, 3, 3)
    W = Tw2v[..., :3, :3]
    T = W @ J
    V = torch.stack([
        cov3D[:, 0], cov3D[:, 1], cov3D[:, 2],
        cov3D[:, 1], cov3D[:, 3], cov3D[:, 4],
        cov3D[:, 2], cov3D[:, 4], cov3D[:, 5],
    ], dim=-1).view(-1, 3, 3)
    V_ = T.transpose(-1, -2) @ V @ T
    cov2D = torch.stack([V_[:, 0, 0] + 0.3, V_[:, 0, 1], V_[:, 1, 1]], dim=-1)
    return cov2D


def test_compute_cov3D():
    from my_ext._C import get_python_function
    N = 10000
    scaling = torch.randn(N, 3)
    rotation = torch.randn(N, 4)
    rotation = ops_3d.quaternion.standardize(ops_3d.quaternion.normalize(rotation))
    py_func = get_python_function('build_covariance_from_scaling_rotation')
    cu_func = _compute_cov3D.apply
    s1 = scaling.cuda().requires_grad_()
    r1 = rotation.cuda().requires_grad_()
    o_cu = cu_func(s1, 1, r1)

    s2 = scaling.cuda().requires_grad_()
    r2 = rotation.cuda().requires_grad_()
    o_py = py_func(s2, 1, r2)

    print('error:', (o_cu - o_py).abs().max())
    g = torch.randn_like(o_cu)
    torch.autograd.backward(o_cu, g)
    torch.autograd.backward(o_py, g)
    print('grad_scaling error:', (s1.grad - s2.grad).abs().max())
    print('grad_rotation error:', (r1.grad - r2.grad).abs().max())
    print(r1.grad[0])
    print(r2.grad[0])

    num_test = 100
    start_timer = torch.cuda.Event(enable_timing=True)
    end_timer = torch.cuda.Event(enable_timing=True)
    for name, func in [("cuda", cu_func), ("python", py_func)]:
        start_timer.record()
        for _ in range(num_test):
            R1 = func(s1, 1, r1)
            torch.autograd.backward(R1, g)
        end_timer.record()
        end_timer.synchronize()
        used_t = start_timer.elapsed_time(end_timer)
        print(f'{name} time: {used_t / num_test:.3f} ms')


def test_compute_cov2D():
    torch.set_default_dtype(torch.float64)
    fovx = fovy = math.radians(60)
    size = 400
    focal = ops_3d.fov_to_focal(fovy, size)
    N = 100
    scaling = torch.randn(N, 3).cuda()
    rotation = torch.randn(N, 4).cuda()
    rotation = ops_3d.quaternion.standardize(ops_3d.quaternion.normalize(rotation))
    cov3D = build_covariance_from_scaling_rotation(scaling, 1., rotation)
    Tw2v = ops_3d.look_at(torch.randn(3), torch.zeros(3)).cuda()
    # Tw2v = torch.eye(4).cuda()
    means = torch.randn((N, 3)).cuda()
    means = (means * 0.3).clamp(-1, 1)
    print(cov3D.shape, Tw2v.shape, means.shape)
    cov2D = _compute_cov2D.apply(cov3D, means, Tw2v, focal, focal, fovx, fovy)
    print(cov2D.shape)
    cov3D_py = cov3D.clone().requires_grad_()
    means_py = means.clone().requires_grad_()
    cov2D_py = compute_cov2D(means_py, cov3D_py, Tw2v, focal, focal, fovx, fovy)
    print(cov2D_py.shape, cov2D.shape)
    print('forward error:', (cov2D_py - cov2D).abs().max())

    # print(means, ops_3d.xfm(means, Tw2v))
    # check_func = lambda x, y: _compute_cov2D.apply(x, y, Tw2v, focal, focal, fovx, fovy)
    # cov3D.requires_grad_()
    # # means.requires_grad_()
    # torch.autograd.gradcheck(check_func, (cov3D, means), nondet_tol=1e-6)
