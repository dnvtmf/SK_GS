""" 刚体变换 rigid/Euclidean Transform 
仅包含旋转Rotation/平移Translation 
matrix: 4x4矩阵
eluer: 欧拉角 (tx, ty, tz, angle1, angle2, angle3) 有万向节死锁(Gimbal Lock)的问题
quaternion: 四元数 (tx, ty, tz, x, y, z, w)
axis_angle:  (tx, ty, tz, ux, uy, uz, theta) 向量(ux, uy, uz)表示旋转轴, theta表示绕旋转轴逆时针旋转的弧度
rotvec: (tx, ty, tz, ux, uy, uz) 其归一化向量(ux, uy, uz)表示旋转轴, 模长||(ux, uy, uz)||表示绕旋转轴逆时针旋转的弧度
lie: (tx, ty, tz, so3_x, so3_y, so3_z)

reference: 
    https://zhuanlan.zhihu.com/p/45404840
"""
from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from . import quaternion, rotation
from .lietorch import SE3, SO3


def translate(*xyz: Union[float, Tensor], dtype=None, device=None):
    if len(xyz) == 1:
        if isinstance(xyz[0], Tensor):
            t = xyz[0]
            assert t.shape[-1] == 3
            dtype = t.dtype if dtype is None else dtype
            device = t.device if device is None else device
            T = torch.eye(4, dtype=dtype, device=device).expand(list(t.shape[:-1]) + [4, 4]).contiguous()
            T[..., :3, 3] = t
            return T
        else:
            assert isinstance(xyz[0], (list, tuple))
            x, y, z = xyz[0]
    else:
        assert len(xyz) == 3
        x, y, z = xyz
    shape = []
    for t in [x, y, z]:
        if isinstance(t, Tensor):
            dtype = t.dtype if dtype is None else dtype
            device = t.device if device is None else device
            shape.append(t.shape)
    if not shape:
        return torch.tensor([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1.]], dtype=dtype, device=device)
    else:
        shape = torch.broadcast_shapes(*shape)
        T = torch.eye(4, dtype=dtype, device=device).expand(list(shape) + [4, 4]).contiguous()
        T[..., 0, 3] = x
        T[..., 1, 3] = y
        T[..., 2, 3] = z
        return T


def _vec2ss_matrix(vector: Tensor):
    """vector to skew-symmetric 3x3 matrix"""
    a, b, c = vector.unbind(-1)
    zeros = torch.zeros_like(a)
    return torch.stack([zeros, -c, b, c, zeros, -a, -b, a, zeros], dim=-1).reshape(*a.shape, 3, 3)


def euler_to_R(xyz, t: Tensor = None, order='xyz'):
    if xyz.shape[-1] == 6:
        t, xyz = xyz[..., :3], xyz[..., 3:]
    assert len(order) == 3 and xyz.shape[-1] == 3

    T = None
    for axis, angle in zip(order, xyz.unbind(-1)):
        c, s = torch.cos(angle), torch.sin(angle)
        o, z = torch.ones_like(angle), torch.zeros_like(angle)
        if axis == "x":
            R_flat = (o, z, z, z, z, c, -s, z, z, s, c, z, z, z, z, o)
        elif axis == "y":
            R_flat = (c, z, s, z, z, o, z, z, -s, z, c, z, z, z, z, o)
        elif axis == "z":
            R_flat = (c, -s, z, z, s, c, z, z, z, z, o, z, z, z, z, o)
        else:
            raise ValueError("letter must be either x, t or z.")

        Ro = torch.stack(R_flat, -1).reshape(angle.shape + (4, 4))
        T = Ro if T is None else T @ Ro
    if t is not None:
        T[..., :3, 3] = t
    return T


def euler_to_quaternion(x=0, y=0, z=0., order='xyz', t: Tensor = None):
    q = None
    x, y, z = [(i if isinstance(i, Tensor) else torch.tensor(i)) for i in [x, y, z]]
    zeros = torch.zeros_like(x)
    for o in order:
        if o == 'x':
            x = 0.5 * x
            qo = torch.cat([x.sin(), zeros, zeros, x.cos()], dim=-1)
        elif o == 'y':
            y = 0.5 * y
            qo = torch.cat([zeros, y.sin(), zeros, y.cos()], dim=-1)
        else:
            z = 0.5 * z
            qo = torch.cat([zeros, zeros, z.sin(), z.cos()], dim=-1)
        q = qo if q is None else quaternion.mul(q, qo)
    if t is None:
        t = torch.zeros_like(q[..., 3])
    return torch.cat([t, q], dim=-1)


def quaternion_to_Rt(q: Tensor, t: Tensor = None):
    if q.shape[-1] == 7:
        t = q[..., :3]
        q = q[..., 3:]
    assert q.shape[-1] == 4  # [x, y, z, w]
    x, y, z, w = q.unbind(-1)
    if t is None:
        t = x.new_zeros(*x.shape, 3)
    else:
        shape = torch.broadcast_shapes(x.shape, t[..., 0].shape)
        t = t.expand(*shape, 3)
        x, y, z, w = [tmp.expand(shape) for tmp in [x, y, z, w]]
    # yapf: disable
    T = torch.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * w * y + 2 * x * z, t[..., 0],
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x, t[..., 1],
        2 * x * z - 2 * w * y, 2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y, t[..., 2],
        torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x),
    ], dim=-1).reshape(*x.shape, 4, 4)
    # yapf: enable
    return T


def quaternion_to_axis_angle(q: Tensor, t: Tensor = None):
    if q.shape[-1] == 7:
        t = q[..., :3]
        q = q[..., 3:]
    assert q.shape[-1] == 4  # [x, y, z, w]
    u, theta = quaternion.to_rotate(q)
    return u, theta, t


def quaternion_to_lie(q: Tensor, t: Tensor = None):
    if t is not None:
        q = torch.cat([q, t], dim=-1)
    return SE3.InitFromVec(q).log().data


def axis_angle_to_Rt(u: Tensor, theta: Tensor = None, t: Tensor = None):
    """convert axis_angle (u, theta) and translate (v) to 4x4 Rigid Transform Matrix  
    Args:
        u : rotate axis, shape: [..., 3] or [..., 6], or [..., 7]
        theta: rotate radian, shape: [...], if None theta = ||w||
        t: translation, shape: [..., 3]
    Returns:
        maxtrix with shape [..., 4, 4]
    """
    if u.shape[-1] == 7:
        assert theta is None and t is None
        theta = u[..., -1]
        t = u[..., :3]
        u = u[..., 3:6]
    elif u.shape[-1] == 6:
        assert t is None
        t = u[..., :3]
        u = u[..., 3:6]
    else:
        assert u.shape[-1] == 3
    matrix = u.new_zeros((*u.shape[:-1], 4, 4))
    u_nrom = u.norm(dim=-1, keepdim=True)
    w_ = _vec2ss_matrix(u / (u_nrom + 1e-15))
    theta = (u_nrom[..., None] if theta is None else theta[..., None, None])
    I = torch.eye(3, dtype=theta.dtype, device=theta.device)
    m2 = torch.matmul(w_, w_)
    matrix[..., :3, :3] = I + theta.sin() * w_ + (1 - theta.cos()) * m2
    if t is not None:
        matrix[..., :3, 3] = t
    matrix[..., 3, 3] = 1.
    return matrix


def axis_angle_to_quaternion(u: Tensor, theta: Tensor = None, t: Tensor = None):
    if theta is None:
        theta = u.norm(dim=-1, keepdim=False)
        u = u / theta[..., None]
    theta = theta[..., None] * 0.5
    if t is None:
        t = torch.zeros_like(u)
    return torch.cat([t, theta.sin() * u, theta.cos()], dim=-1)


def Rt_to_euler(Rt: Tensor, order='xyz'):
    angles = rotation.R_to_euler(Rt[..., :3, :3], order=order)
    return torch.cat([Rt[..., :3, 3], angles], dim=-1)


def Rt_to_quaternion(Rt: Tensor):
    t = Rt[..., :3, 3]
    w = 0.5 * torch.sqrt(Rt[..., 0, 0] + Rt[..., 1, 1] + Rt[..., 2, 2] + 1)
    w_ = 1. / (4 * w + 1e-10)  # 避免数值不稳定, 存在不用除法的算法
    x = (Rt[..., 2, 1] - Rt[..., 1, 2]) * w_
    y = (Rt[..., 0, 2] - Rt[..., 2, 0]) * w_
    z = (Rt[..., 1, 0] - Rt[..., 0, 1]) * w_
    return torch.cat([t, x[..., None], y[..., None], z[..., None], w[..., None]], dim=-1)


def Rt_to_axis_angle(Rt: Tensor):
    return quaternion_to_axis_angle(Rt_to_quaternion(Rt))


def Rt_to_lie(Rt: Tensor) -> Tensor:
    q = Rt_to_quaternion(Rt)
    t = q[..., :3]
    so3 = SO3.InitFromVec(q[..., 3:]).log().data
    return torch.cat([t, so3], dim=-1)


def lie_to_quaternion(lie: Tensor):
    return SE3.exp(lie).vec()


def lie_to_Rt(lie: Tensor, t: Tensor = None) -> Tensor:
    """ Convert lie representation to 4x4 Transform Matrix

    Args:
        lie: [ϕ, so3], shape: [..., 6],  Note: lie is not [t, so3], rather that
        t: None or [..., 3]

    Returns:
        matrix with shape [..., 4, 4]
    """
    if t is None:
        t, lie = lie.split([3, 3], dim=-1)
        #     return SE3.exp(lie).matrix()
        # else:
    q = SO3.exp(lie).vec()
    tq = torch.cat([t, q], dim=-1)
    # return quaternion_to_Rt(tq)
    return SE3.InitFromVec(tq).matrix()


def lie_to_Rt_(lie: Tensor, eps=1e-6) -> Tensor:
    """ Convert lie representation to 4x4 Transform Matrix

    Args:
        lie: [ϕ, ρ], shape: [..., 6], 

    Returns:
        matrix with shape [..., 4, 4]
    """
    phi, rho = lie.split(3, dim=-1)
    theta = rho.norm(dim=-1, keepdim=False).clamp_min(eps)  # avoid divide zero
    a = rho / theta[..., None]
    I = torch.eye(3, device=rho.device)
    theta = theta[..., None, None]
    cos = theta.cos()
    sin = theta.sin()
    matrix = torch.eye(4, dtype=rho.dtype, device=rho.device).expand(*rho.shape[:-1], 4, 4).contiguous()
    aaT = a[..., None, :] * a[..., :, None]
    a_up = _vec2ss_matrix(a)
    matrix[..., :3, :3] = cos * I + (1 - cos) * aaT + sin * a_up
    sin_theta = sin / theta
    J = sin_theta * I + (1 - sin_theta) * aaT + (1 - cos) / theta * a_up
    matrix[..., :3, 3] = torch.sum(J * phi[..., None, :], dim=-1)
    return matrix


def Rt_to_lie_np(matrix: Tensor):
    from scipy.spatial.transform import Rotation
    # from .lietorch import SE3
    quat = Rotation.from_matrix(matrix[..., :3, :3].detach().cpu().numpy()).as_quat()
    trans = matrix[..., :3, 3].detach().cpu().numpy()
    pose_data = np.concatenate((trans, quat), axis=-1)
    T = SE3.InitFromVec(torch.from_numpy(pose_data).to(matrix))
    return T.log()


def rotation_6d_to_Rt(d6: torch.Tensor, t: Tensor = None) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6) or (*, 9)
        t: translation

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    assert (d6.shape[-1] == 9 and t is None) or (d6.shape[-1] == 6 and t.shape[-1] == 3)
    a1, a2 = d6[..., -6:-3], d6[..., -3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    T = d6.new_zeros((*d6.shape[:-1], 4, 4))
    T[..., 0, :3] = b1
    T[..., 1, :3] = b2
    T[..., 2, :3] = b3
    T[..., :3, 3] = d6[..., :3] if t is None else t
    T[..., 3, 3] = 1
    return T


def Rt_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 4, 4)

    Returns:
        6D rotation representation + translatation, of size (*, 9)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return torch.cat([matrix[..., :3, 3], matrix[..., :2, :3].flatten(-2, -1)], dim=-1)


def test():
    import pytorch3d.transforms
    from scipy.spatial.transform import Rotation

    from my_ext import utils
    utils.set_printoptions(6)
    print()

    n = 10
    u = torch.randn(n, 3)
    u = u / u.norm(dim=-1, keepdim=True)
    theta = torch.rand(n) * 2 * torch.pi
    t = torch.rand(n, 3)

    gtM = torch.eye(4).expand(n, 4, 4).contiguous()
    gt2 = gtM.clone()
    gtM[:, :3, :3] = torch.from_numpy(Rotation.from_rotvec((u * theta[..., None]).numpy()).as_matrix())
    gtM[:, :3, 3] = t
    gt2[:, :3, :3] = pytorch3d.transforms.axis_angle_to_matrix(u * theta[..., None])
    gt2[:, :3, 3] = t
    print(gtM.shape, gt2.shape)

    def cmp_m(m, eps=1e-6):
        error_1 = (m - gtM).abs().max()
        error_2 = (m - gt2).abs().max()
        assert error_1 < eps and error_2 < eps
        return f"{error_1}, {error_2}"

    m1 = axis_angle_to_Rt(u, theta, t)
    m1_ = axis_angle_to_Rt(torch.cat([t, u * theta[..., None]], dim=-1))
    print('axis_angle_to_matrix:', cmp_m(m1), cmp_m(m1_))

    my_tq = axis_angle_to_quaternion(u, theta, t)
    m2 = quaternion_to_Rt(my_tq)
    print('quaternion_to_matrix:', cmp_m(m2))

    my_tq2 = Rt_to_quaternion(gtM)
    print(my_tq[0], my_tq2[0])
    assert torch.min((my_tq + my_tq2).abs(), (my_tq - my_tq2).abs()).abs().max() < 1e-5

    assert (Rt_to_lie(gtM) - Rt_to_lie_np(gtM)).abs().max() < 1e-6
    m3 = lie_to_Rt(Rt_to_lie(gtM))
    print('lie_to_quaternion:', ((lie_to_quaternion(quaternion_to_lie(my_tq))) - my_tq).abs().max())
    # print(m2[0], m3[0])
    print('lie_to_matrix:', cmp_m(m3))
    t_ = Rt_to_lie(gtM)
    print(t[0], t_[0])
    m3_ = quaternion_to_Rt(lie_to_quaternion(quaternion_to_lie(my_tq)))
    print('lie_to_quaternion:', cmp_m(m3_))

    for xyz in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
        m4 = euler_to_R(Rt_to_euler(gtM, order=xyz), order=xyz)
        print(f'euler_to_matrix[{xyz}]', cmp_m(m4))

    m5 = rotation_6d_to_Rt(Rt_to_rotation_6d(gtM))
    print(f'rotation_6d_to_Rt:', cmp_m(m5))
