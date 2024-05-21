"""相机位姿相关函数"""
import torch
from torch import Tensor

__all__ = ['compute_camera_algin', 'align_camera_poses', 'camera_poses_error', 'get_center_and_diag']


def compute_camera_algin(X0: Tensor, X1: Tensor):
    """
    compute camera algin parameters using procrustes analysis

    Args:
        X0: shape: [N, 3]
        X1: shape: [N, 3]

    Returns:
        t0, t1, s0, s1, R: X1to0 = (X1 - t1) / s1 @ R.t() * s0 + t0
    """
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = (X0c ** 2).sum(dim=-1).mean().sqrt()
    s1 = (X1c ** 2).sum(dim=-1).mean().sqrt()
    X0cs = X0c / s0
    X1cs = X1c / s1
    try:
        # rotation (use double for SVD, float loses precision)
        # U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
        # R = (U @ V.t()).float()  # type:Tensor
        U, S, Vh = torch.linalg.svd((X0cs.t() @ X1cs).double())
        R = (U @ Vh).float()  # type:Tensor
        if R.det() < 0:
            R[2] *= -1
    except:
        print("warning: SVD did not converge...")
        return 0, 0, 1, 1, torch.eye(3, device=X0.device, dtype=X0.dtype)
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    t0, t1 = t0[0], t1[0]  # shape: [3], [3]
    return t0, t1, s0, s1, R


def align_camera_poses(poses1: Tensor, poses2: Tensor):
    assert poses1.shape == poses2.shape and poses1.ndim == 3 and poses1.shape[1:] == (4, 4)
    # center = poses1.new_zeros(3)
    # X1 = xfm(center, poses1)
    # X2 = xfm(center, poses2)
    # X1 = X1[..., :3] / X1[..., 3:]
    # X2 = X2[..., :3] / X2[..., 3:]
    X1 = poses1[:, :3, 3]
    X2 = poses2[:, :3, 3]
    t1, t2, s1, s2, R = compute_camera_algin(X1, X2)
    R_aligned = poses1[..., :3, :3] @ R
    t_aligned = (X1 - t1) / s1 @ R * s2 + t2
    pose_aligned = poses1.clone()
    pose_aligned[..., :3, :3] = R_aligned
    pose_aligned[..., :3, 3] = t_aligned
    return pose_aligned


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def camera_poses_error(poses: Tensor, gt_poses: Tensor, aligned=False, degree=True, reduction='mean'):
    """ calculate the rotation error and translation error between poses and gt_poses

    Args:
        poses: shape: [..., 4, 4]
        gt_poses:  shape: [..., 4, 4]
        aligned: poses are aligned?. Defaults to False.
        degree: convert rotation error to degree. Defaults to True.
        reduction: 'mean', 'sum', None
    Returns:
        rotation_error, translation_error
    """
    pose_aligned = align_camera_poses(poses, gt_poses) if not aligned else poses
    R_aligned, t_aligned = pose_aligned[:, :3, :3], pose_aligned[:, :3, 3]
    R_GT, t_GT = gt_poses[:, :3, :3], gt_poses[:, :3, 3]

    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned - t_GT).norm(dim=-1)  # type: Tensor
    if degree:
        R_error.rad2deg_()
    if reduction == 'mean':
        return R_error.mean(), t_error.mean()
    elif reduction == 'sum':
        return R_error.sum(), t_error.sum()
    else:
        return R_error, t_error


def camera_translate_scale(Tv2w: Tensor = None, K: Tensor = None, translate=0., scale=1.0):
    if Tv2w is not None:
        Tv2w[..., :3, 3] = (Tv2w[..., :3, 3] + translate) * scale
    if K is not None:
        assert translate == 0, f"translate != 0 not implemented"
        K[..., 0, 0] *= scale
        K[..., 1, 1] *= scale
        K[..., 0, 2] *= scale
        K[..., 1, 2] *= scale
    return Tv2w, K


def get_center_and_diag(cam_pos: Tensor):
    center = torch.mean(cam_pos, dim=0, keepdim=True)
    dist = torch.linalg.norm(cam_pos - center, dim=1, keepdim=True)
    diagonal = torch.max(dist)
    return center.flatten().view(-1), diagonal
