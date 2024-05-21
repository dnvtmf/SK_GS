import torch
import my_ext.ops_3d.lie_pypose as pp

from my_ext.ops_3d.lietorch import SE3, SO3


def sym(x):
    z = torch.zeros_like(x[..., 0])
    R = torch.stack([z, -x[..., 2], x[..., 1], x[..., 2], z, -x[..., 0], -x[..., 1], x[..., 0], z], dim=-1)
    R = R.reshape(*z.shape, 3, 3)
    return R


def exp_SO3(phi):
    theta = phi.norm(dim=-1, keepdim=True)
    a = phi / theta
    I = torch.eye(3, device=a.device).expand(*a.shape[:-1], 3, 3)
    cos_theta = theta.cos()[..., None]
    sin_theta = theta.sin()[..., None]
    R = I * cos_theta + (1 - cos_theta) * a[..., None] * a[..., None, :] + sin_theta * sym(a)
    return R


def exp_SE3(phi, rho):
    R = exp_SO3(phi)
    theta = phi.norm(dim=-1, keepdim=True)
    a = phi / theta
    I = torch.eye(3, device=a.device).expand(*a.shape[:-1], 3, 3)
    cos_theta = ((1 - theta.cos()) / theta)[..., None]
    sin_theta = (theta.sin() / theta)[..., None]
    J = I * sin_theta + (1 - sin_theta) * a[..., None] * a[..., None, :] + cos_theta * sym(a)
    t = J.bmm(rho[..., None])[..., 0]
    T = R.new_zeros(*R.shape[:-2], 4, 4)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def test():
    N = 10
    phi = torch.randn(N, 3).cuda()
    rho = torch.randn(N, 3).cuda()

    R1 = SO3.exp(phi).matrix()
    R2 = exp_SO3(phi)
    so3 = pp.LieTensor(phi, ltype=pp.so3_type)
    R3 = so3.matrix()
    print(R1.shape, R2.shape, R3.shape)
    print('SO3 Error:', (R1[..., :3, :3] - R2).abs().max())
    print('SO3 Error:', (R3 - R2).abs().max())

    T1 = SE3.exp(torch.cat([rho, phi], dim=-1)).matrix()
    T2 = exp_SE3(phi, rho)
    se3 = pp.LieTensor(torch.cat([rho, phi], dim=-1), ltype=pp.se3_type)
    T3 = se3.matrix()
    print(T1.shape, T2.shape, T3.shape)
    print('SE3 Error', (T1 - T2).abs().max())
    print('SE3 Error', (T1 - T3).abs().max())


if __name__ == '__main__':
    test()
