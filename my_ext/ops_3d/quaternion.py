"""四元数 quaternion
表示为shape[-1] == 4的Tensor, 分别表示 q(x, y, z, w) = w + xi + yj + zk 

Reference:
    - https://krasjet.github.io/quaternion/quaternion.pdf
    - https://zhuanlan.zhihu.com/p/375199378
"""
from typing import Union, Any
import torch
from torch import Tensor
from my_ext._C import get_C_function, try_use_C_extension


def norm(q: Tensor, keepdim=False) -> Tensor:
    return q.norm(dim=-1, keepdim=keepdim)


def normalize(q: Tensor):
    return torch.nn.functional.normalize(q, dim=-1)


def add(q1: Tensor, q2: Tensor):
    return q1 + q2


def mul(q: Tensor, s: Union[float, Tensor]):
    if isinstance(s, float):
        return q * s
    elif isinstance(s, Tensor):
        if q.ndim > s.ndim:
            return q * s[..., None]
        else:
            assert s.shape[-1] == 4
            # yapf: disable
            # b, c, d, a = q.unbind(-1)
            # f, g, h, e = q.unbind(-1)
            # return torch.stack([
            #     a * e - b * f - c * g - d * h,
            #     b * e + a * f - d * g + c * h,
            #     c * e + d * f + a * g - b * h,
            #     d * e - c * f + b * g + a * h,
            # ], dim=-1)
            # #(Graßmann Product)
            qxyz, qw = q[..., :3], q[..., -1:]
            sxyz, sw = s[..., :3], s[..., -1:]
            return torch.cat([
                sw * qxyz + qw * sxyz + torch.linalg.cross(qxyz, sxyz),
                qw * sw - torch.linalg.vecdot(qxyz, sxyz)[..., None]
            ], dim=-1)
            # yapf: enable
    else:
        raise ValueError()


def conj(q: Tensor):
    """共轭 conjugate"""
    return torch.cat([-q[..., :3], q[..., -1:]], dim=-1)


def inv(q: Tensor):
    """四元数的逆

    q是单位四元数时, 逆为conj(q)"""
    return conj(q) / norm(q)[..., None]


def cross(qa: Tensor, qb: Tensor):
    qc = torch.zeros_like(qa)
    qc[..., :3] = qa[..., 3:] * qb[..., :3] + qb[..., 3:] * qa[..., :3] + torch.cross(qa[..., :3], qb[..., :3], dim=-1)
    return qc


def from_rotate(u: Tensor, theta: Tensor):
    """ 从旋转轴u和旋转角θ构造四元数

    Args:
        u: 旋转轴, 单位向量, shape: [..., 3]
        theta: 旋转角, 弧度; shape [....]
    Returns:
        四元数 [..., 4]
    """
    theta = theta[..., None] * 0.5
    return torch.cat([theta.sin() * u, theta.cos()], dim=-1)


def to_rotate(q: Tensor):
    """
    从四元数提取 旋转轴u和旋转角θ

    Args:
        q: 单位四元数 shape [..., 4]

    Returns:
        u: 旋转轴, 单位向量, shape: [..., 3]; theta: 旋转角, 弧度; shape [....]
    """
    theta = torch.arccos(q[..., -1])
    u = q[..., :3] / torch.sin(theta)
    return u, 2. * theta


def xfm(points: Tensor, q: Tensor):
    """使用四元数旋转点"""
    points = torch.cat([points, torch.zeros_like(points[..., :1])], dim=-1)
    return mul(mul(q, points), conj(q))[..., :3]


def pow(q: Tensor, t: Tensor):
    u, theta = to_rotate(q)
    return from_rotate(u, t * theta)


def interpolation(t: Union[float, Tensor], q1: Tensor, q2: Tensor, method='slerp'):
    """插值, 插值角度较小时可使用nlerp, slerp避免插值角度接近0"""
    if method == 'slerp0':  # Spherical Linear Interpolation
        return mul(pow(mul(q2, conj(q1)), t), q1)
    elif method == 'slerp':
        theta = torch.arccos(torch.linalg.vecdot(q1, q2))
        a = torch.sin((1 - t) * theta)
        b = torch.sin(t * theta)
        c = torch.sin(theta)
        return a / c * q1 + b / c * q2
    elif method == 'nlerp':  # 正规化线性插值（Normalized Linear Interpolation）
        return normalize((1 - t) * q1 + t * q2)
    elif method == 'squad':  # 球面四边形插值 (Spherical and quadrangle)
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def standardize(quaternions: Tensor) -> Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


class _to_R(torch.autograd.Function):
    _forward = get_C_function('quaternion_to_R_forward')
    _backward = get_C_function('quaternion_to_R_backward')

    @staticmethod
    def forward(ctx, *args, **kwargs):
        q, = args
        R = _to_R._forward(q)
        ctx.save_for_backward(q)
        return R

    @staticmethod
    def backward(ctx, *grad_outputs):
        q, = ctx.saved_tensors
        grad_q = _to_R._backward(q, grad_outputs[0])
        return grad_q


@try_use_C_extension(_to_R.apply, "quaternion_to_R_forward", "quaternion_to_R_backward")
def toR(q: Tensor):
    """将四元数标准化并得到旋转矩阵"""
    x, y, z, w = normalize(q).unbind(-1)
    # yapf: disable
    R = torch.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * w * y + 2 * x * z,
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x,
        2 * x * z - 2 * w * y, 2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y,
    ], dim=-1).reshape(*x.shape, 3, 3)
    # yapf: enable
    return R


def test():
    from my_ext import utils
    utils.set_printoptions()
    print()
    q1 = normalize(torch.randn(10, 4, device='cuda'))
    q2 = normalize(torch.randn(10, 4, device='cuda'))
    print(utils.show_shape(mul(q1, q2)))
    print(mul(q1, inv(q1)))
    R = toR(q1)
    print(R[0], R[0] @ R[0].T)
    from lietorch import SO3
    R_ = SO3.InitFromVec(q1)
    T = R_.matrix()
    print(T.shape, R.shape)
    print(T[0], R[0], R[0].inverse())
    print((T[:, :3, :3] - R).abs().max())
    print('cmp quaternion vs. SO3->q', q1 - R_.vec())

    u = torch.tensor([1., 0., 0.], device='cuda')
    theta = torch.tensor(30.).deg2rad_().cuda()
    q = from_rotate(u, theta)
    print(to_rotate(q), u, theta)
    R = toR(q)
    print(R, SO3.InitFromVec(q).matrix())
    from scipy.spatial.transform import Rotation
    print(Rotation.from_quat(q.cpu().numpy()).as_matrix())
    print(Rotation.from_quat(q.cpu().numpy()).as_euler('xyz', degrees=True))
    from my_ext.ops_3d.coord_trans import rotate_x
    print(rotate_x(theta))
    points = torch.randn(10, 3, device='cuda')
    print(xfm(points, q[None]) - points @ R.T)


def test_to_R():
    from my_ext.utils.test_utils import get_run_speed
    torch.set_default_dtype(torch.float32)
    from my_ext._C import get_python_function
    print()

    py_func = get_python_function('toR')
    cu_func = _to_R.apply
    cpu_func = cu_func
    N = 10000
    q = torch.randn(N, 4)
    q = standardize(normalize(q))
    q1 = q.clone().cuda().requires_grad_()
    q2 = q.clone().cuda().requires_grad_()
    q3 = q.clone().cpu().requires_grad_()
    R1 = py_func(q1)
    R2 = cu_func(q2)
    R3 = cpu_func(q3)
    assert R1.shape == R2.shape and R1.shape == R3.shape
    print('error:', (R1 - R2).abs().max(), R1.dtype, R2.dtype)
    print('error:', (R1 - R3.cuda()).abs().max(), R1.dtype, R3.dtype)

    g = torch.randn_like(R1)
    torch.autograd.backward(R1, g)
    torch.autograd.backward(R2, g)
    torch.autograd.backward(R3, g.cpu())
    print('grad error:', (q1.grad - q2.grad).abs().max())
    print('grad error:', (q1.grad - q3.grad.cuda()).abs().max())
    # index = (q1.grad - q3.grad.cuda()).abs().argmax().item() // 4
    # print('error:', index, q1[index], q3[index], q1.grad[index], q3.grad[index])

    get_run_speed(q1, g, py_func, cu_func, cpu_func)


def test_xfm():
    N = 100
    q = torch.randn(N, 4).cuda()
    p = torch.randn(N, 3).cuda()
    q = normalize(q)
    y = xfm(p, q)
    y2 = torch.einsum('bij,bj->bi', toR(q), p)
    print(y.shape, y2.shape)
    print((y - y2).abs().max())
