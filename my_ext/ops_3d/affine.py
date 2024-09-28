""" 仿射变换 Affine Transform 
包含旋转Rotation, 平移Translation, 缩放Scale, 对称Reflection, 错切Shear
TODO: 完全未实现
matrix: 4x4矩阵
"""
import torch
import pytorch3d.transforms
from torch import Tensor
import numpy as np
from . import quaternion
from .coord_trans import rotate_x, rotate_y, rotate_z


def _vec2ss_matrix(vector: Tensor):
    """vector to skew-symmetric 3x3 matrix"""
    a, b, c = vector.unbind(-1)
    zeros = torch.zeros_like(a)
    return torch.stack([zeros, -c, b, c, zeros, -a, -b, a, zeros], dim=-1).reshape(*a.shape, 3, 3)


def euler_to_matrix(x=0, y=0, z=0., order='xyz'):
    T = None
    for o in order:
        if o == 'x':
            R = rotate_x(x)
        elif o == 'y':
            R = rotate_y(y)
        else:
            R = rotate_z(z)
        T = R if T is None else R @ T
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


def quaternion_to_matrix(q: Tensor, t: Tensor = None):
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
    T = torch.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * w * y + 2 * x * z, t[..., 0],
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x, t[..., 1],
        2 * x * z - 2 * w * y, 2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y, t[..., 2],
        torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x),
    ], dim=-1).reshape(*x.shape, 4, 4) # yapf: disable
    return T


def quaternion_to_axis_angle(q: Tensor, t: Tensor = None):
    if q.shape[-1] == 7:
        t = q[..., :3]
        q = q[..., 3:]
    assert q.shape[-1] == 4  # [x, y, z, w]
    u, theta = quaternion.to_rotate(q)
    return u, theta, t


def axis_angle_to_matrix(u: Tensor, theta: Tensor = None, t: Tensor = None):
    """convert axis_angle (w, theta) and translate (v) to 3x3 Rotation Matrix  
    Args:
        w : rotate axis, shape: [..., 3]
        theta: rotate radian, shape: [...], if None theta = ||w||
        t: translation, shape: [..., 3]
    Returns:
        maxtrix with shape [..., 4, 4]
    """
    matrix = u.new_zeros((*u.shape[:-1], 4, 4))
    w_skewsym = _vec2ss_matrix(u)
    theta = (u.norm(dim=-1) if theta is None else theta)[..., None, None]
    I = torch.eye(3, dtype=theta.dtype, device=theta.device)
    matrix[..., :3, :3] = I + theta.sin() * w_skewsym + (1 - theta.cos()) * torch.matmul(w_skewsym, w_skewsym)
    if t is not None:
        matrix[..., :3, 3] = torch.matmul(
            I * theta + (1 - theta.cos()) * w_skewsym + (theta - theta.sin()) * torch.matmul(w_skewsym, w_skewsym), t
        )
    matrix[..., 3, 3] = 1.
    return matrix


def axis_angle_to_quaternion(u: Tensor, theta: Tensor = None, t: Tensor = None):
    if theta is None:
        theta = u.norm(dim=-1, keepdim=False)
        u = u / theta[..., None]
    q = quaternion.from_rotate(u, theta)
    if t is not None:
        return torch.cat([t, q], dim=-1)
    return q


def matrix_to_euler():
    raise NotImplementedError()


def matrix_to_quaternion(T: Tensor):
    t = T[..., :3, 3]
    w = 0.5 * torch.sqrt(T[..., 0, 0] + T[..., 1, 1] + T[..., 2, 2] + 1)
    w_ = 1. / (4 * w + 1e-6)  # 避免数值不稳定, 存在不用除法的算法
    x = (T[..., 2, 1] - T[..., 1, 2]) * w_
    y = (T[..., 0, 2] - T[..., 2, 0]) * w_
    z = (T[..., 1, 0] - T[..., 0, 1]) * w_
    return torch.cat([t, x[..., None], y[..., None], z[..., None], w[..., None]], dim=-1)


def matrix_to_axis_angle(T: Tensor):
    return quaternion_to_axis_angle(matrix_to_quaternion(T))


def matrix_to_lie(matrix):
    from lietorch import SE3
    quat = pytorch3d.transforms.matrix_to_quaternion(matrix[..., :3, :3])
    quat = torch.cat((quat[..., 1:], quat[..., 0:1]), -1)  # swap real first to real last
    trans = matrix[..., :3, 3]
    vec = torch.cat((trans, quat), -1)
    Ps = SE3.InitFromVec(vec)
    return Ps


def matrix_to_lie_np(matrix: Tensor):
    from lietorch import SE3
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(matrix[..., :3, :3].detach().cpu().numpy()).as_quat()
    trans = matrix[..., :3, 3].detach().cpu().numpy()
    pose_data = np.concatenate((trans, quat), axis=-1)
    T = SE3.InitFromVec(torch.from_numpy(pose_data).to(matrix))
    return T


def test():
    import pytorch3d.transforms
    from scipy.spatial.transform import Rotation
    utils.set_printoptions(6)
    print()
    n = 10
    matrix = torch.eye(4).expand(n, 4, 4).contiguous()
    matrix[:, :3, :3] = pytorch3d.transforms.random_rotations(n)
    matrix[:, :3, 3] = torch.randn(n, 3)
    T = matrix_to_lie(matrix)
    error = (T.matrix() - matrix).sum()
    print(error)

    u = torch.randn(10, 3)
    u = u / u.norm(dim=-1, keepdim=True)
    theta = torch.rand(10) * 2 * torch.pi
    gt = torch.from_numpy(Rotation.from_rotvec((u * theta[..., None]).numpy()).as_matrix())
    gt2 = pytorch3d.transforms.axis_angle_to_matrix(u * theta[..., None])
    print(gt.shape, gt2.shape)

    def cmp(m, eps=1e-6):
        m = m[..., :3, :3]
        return (m - gt).abs().max() < eps and (m - gt2).abs().max() < eps

    m1 = axis_angle_to_matrix(u, theta)
    assert cmp(m1)
    gt_tq = matrix_to_lie(m1).data
    my_q = axis_angle_to_quaternion(u, theta)
    print(gt_tq[0], my_q[0])
    assert torch.min((gt_tq[..., 3:] + my_q).abs(), (gt_tq[..., 3:] - my_q).abs()).abs().max() < 1e-6
    m2 = quaternion_to_matrix(my_q)
    assert cmp(m2)
    my_tq = matrix_to_quaternion(m2)
    assert torch.min((gt_tq + my_tq).abs(), (gt_tq - my_tq).abs()).abs().max() < 1e-5
