"""spherical harmonic 球谐系数相关"""
import torch
from torch import Tensor

__all__ = ['rotation_SH']


def rotation_SH(sh: Tensor, R: Tensor):
    """Reference:
        https://en.wikipedia.org/wiki/Wigner_D-matrix
        https://github.com/andrewwillmott/sh-lib
        http://filmicworlds.com/blog/simple-and-fast-spherical-harmonic-rotation/
    """
    from scipy.spatial.transform import Rotation
    import sphecerix

    # option 1
    Robj = Rotation.from_matrix(R[..., :3, :3].cpu().numpy())
    B, N, _ = sh.shape
    sh = sh.transpose(1, 2).reshape(-1, N)
    new_sh = sh.clone()
    cnt = 0
    i = 0
    while cnt < N:
        D = sphecerix.tesseral_wigner_D(i, Robj)
        D = torch.from_numpy(D).to(sh)
        new_sh[:, cnt:cnt + D.shape[0]] = sh[:, cnt:cnt + D.shape[0]] @ D.T
        cnt += D.shape[0]
        i += 1

    # option 2
    # from e3nn import o3
    # rot_angles = o3._rotation.matrix_to_angles(R)
    # D_2 = o3.wigner_D(2, rot_angles[0], rot_angles[1], rot_angles[2])
    #
    # Y_2 = self._features_rest[:, [3, 4, 5, 6, 7]]
    # Y_2_rotated = torch.matmul(D_2, Y_2)
    # self._features_rest[:, [3, 4, 5, 6, 7]] = Y_2_rotated
    # print((sh - new_sh).abs().mean())
    return new_sh.reshape(B, 3, N).transpose(1, 2)
