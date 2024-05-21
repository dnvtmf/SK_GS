"""Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/camera.py"""
import torch
from torch import Tensor
import numpy as np

from .misc import to_4x4


class Lie:
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/camera.py
    """

    @classmethod
    def so3_to_SO3(cls, w):  # [...,3]
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    @classmethod
    def SO3_to_so3(cls, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        # ln(R) will explode if theta==pi
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[..., None, None] % torch.pi
        lnR = 1 / (2 * cls.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    @classmethod
    def se3_to_SE3(cls, wu: Tensor, u: Tensor = None):  # [...,6]
        if u is None:
            w, u = wu.split([3, 3], dim=-1)
        else:
            w = wu
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        C = cls.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return to_4x4(Rt)

    @classmethod
    def SE3_to_se3(cls, Rt: Tensor, eps=1e-8):  # [...,3/4,4]
        R, t = Rt[..., :3, :].split([3, 1], dim=-1)
        w = cls.SO3_to_so3(R)
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    @classmethod
    def skew_symmetric(cls, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([
            torch.stack([O, -w2, w1], dim=-1),
            torch.stack([w2, O, -w0], dim=-1),
            torch.stack([-w1, w0, O], dim=-1)
        ],
            dim=-2)
        return wx

    @classmethod
    def taylor_A(cls, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_B(cls, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_C(cls, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans


class Lie2:
    """based on nope-nerf"""

    @staticmethod
    def SO3_to_quat(R):
        """
        :param R:  (N, 3, 3) or (3, 3) np
        :return:   (N, 4, ) or (4, ) np
        """
        from scipy.spatial.transform import Rotation as RotLib
        x = RotLib.from_matrix(R)
        quat = x.as_quat()
        return quat

    @staticmethod
    def quat_to_SO3(quat):
        """
        :param quat:    (N, 4, ) or (4, ) np
        :return:        (N, 3, 3) or (3, 3) np
        """
        from scipy.spatial.transform import Rotation as RotLib
        x = RotLib.from_quat(quat)
        R = x.as_matrix()
        return R

    @staticmethod
    def convert3x4_4x4(input):
        """
        :param input:  (N, 3, 4) or (3, 4) torch or np
        :return:       (N, 4, 4) or (4, 4) torch or np
        """
        if torch.is_tensor(input):
            if len(input.shape) == 3:
                output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
                output[:, 3, 3] = 1.0
            else:
                output = torch.cat([input, torch.tensor([[0, 0, 0, 1]], dtype=input.dtype, device=input.device)],
                    dim=0)  # (4, 4)
        else:
            if len(input.shape) == 3:
                output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
                output[:, 3, 3] = 1.0
            else:
                output = np.concatenate([input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
                output[3, 3] = 1.0
        return output

    @staticmethod
    def vec2skew(v):
        """
        :param v:  (3, ) torch tensor
        :return:   (3, 3)
        """
        zero = torch.zeros(1, dtype=torch.float32, device=v.device)
        skew_v0 = torch.cat([zero, -v[2:3], v[1:2]])  # (3, 1)
        skew_v1 = torch.cat([v[2:3], zero, -v[0:1]])
        skew_v2 = torch.cat([-v[1:2], v[0:1], zero])
        skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
        return skew_v  # (3, 3)

    @staticmethod
    def Exp(r):
        """so(3) vector to SO(3) matrix
        :param r: (3, ) axis-angle, torch tensor
        :return:  (3, 3)
        """
        skew_r = Lie2.vec2skew(r)  # (3, 3)
        norm_r = r.norm() + 1e-15
        eye = torch.eye(3, dtype=torch.float32, device=r.device)
        R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r ** 2) * (skew_r @ skew_r)
        return R

    @staticmethod
    def make_c2w(r, t):
        """
        :param r:  (3, ) axis-angle             torch tensor
        :param t:  (3, ) translation vector     torch tensor
        :return:   (4, 4)
        """
        R = Lie2.Exp(r)  # (3, 3)
        c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
        c2w = Lie2.convert3x4_4x4(c2w)  # (4, 4)
        return c2w


def test():
    from my_ext import ops_3d
    q = ops_3d.quaternion.normalize(torch.randn(4))
    print(q)
    R = ops_3d.quaternion.toR(q)
    print(R)
    so3 = Lie.SO3_to_so3(R)
    print(so3)
    R_ = Lie.so3_to_SO3(so3)
    print(R_, R - R_)
    delat_q = q.clone()
    delat_q[0] = 0.1
    print('delat_q', delat_q)
    delat_q = ops_3d.quaternion.normalize(delat_q)
    dR = ops_3d.quaternion.toR(delat_q)
    q2 = ops_3d.quaternion.mul(delat_q, q)
    R2 = ops_3d.quaternion.toR(q2)
    R3 = dR @ R
    so3_d = Lie.SO3_to_so3(dR)
    so3_2 = so3 + so3_d
    R4 = Lie.so3_to_SO3(so3_2)
    print('R2 vs R3:', R2 - R3)
    print('R2 vs R4:', R2 - R4)
    so3_3 = Lie.SO3_to_so3(R2)
    print('delta_so3:', so3_d, so3_3 - so3)

    so3_np = Lie2.quat_to_SO3(q.numpy())
    print('so3 Lie vs Lie2:', so3_np - R.numpy())
    print(dR - Lie2.Exp(so3_d))
    print(R_ - Lie2.Exp(so3))
    print(R4 - Lie2.Exp(so3_2))
