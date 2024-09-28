"""相机位姿相关函数"""
from typing import Union

import numpy as np
import torch
from torch import Tensor
import scipy

from .rotation import axis_angle_to_R
from . import rotation_np

__all__ = ['compute_camera_align', 'align_camera_poses', 'get_center_and_diag', 'camera_translate_scale',
           'camera_poses_error', 'compute_RPE', 'compute_ATE']


def compute_camera_align(p_unalgined: Tensor, p_target: Tensor, return_aligned: bool = False):
    """
    compute camera algin parameters using procrustes analysis

    Args:
        p_unalgined: shape: [N, 3]
        p_target: shape: [N, 3]
        return_aligned: bool
    Returns:
        - s: shape: 1
        - R: shape: [3, 3]
        - t: shape [3]
        p_target = s * R @ p_unalgined + t
    """
    if isinstance(p_unalgined, Tensor):
        # subtract mean
        mu1 = p_unalgined.mean(dim=0, keepdim=True)
        mu2 = p_target.mean(dim=0, keepdim=True)
        X0c = p_unalgined - mu1
        X1c = p_target - mu2
        # scale
        s0 = X0c.norm()  # (X0c ** 2).sum(dim=-1).mean().sqrt()
        s1 = X1c.norm()  # (X1c ** 2).sum(dim=-1).mean().sqrt()
        # s0 = (X0c ** 2).sum(dim=-1).mean().sqrt()
        # s1 = (X1c ** 2).sum(dim=-1).mean().sqrt()

        X0cs = X0c / s0
        X1cs = X1c / s1
        U, S, Vh = torch.linalg.svd((X0cs.t().double() @ X1cs.double()))
        R = (U @ Vh).to(p_target.dtype)  # type:Tensor
        if R.det() < 0:
            R[2] *= -1
        scale = S.sum()
        if return_aligned:
            return X0cs * scale, X1cs
        scale = scale * s1 / s0
        t = mu2[0] - mu1[0] * scale @ R  # shape[3]
        R = R.T
    else:
        mtx1 = np.array(p_unalgined, dtype=np.double, copy=True)
        mtx2 = np.array(p_target, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mu1 = np.mean(mtx1, 0)
        mu2 = np.mean(mtx2, 0)
        mtx1 -= mu1
        mtx2 -= mu2

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, scale = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        if return_aligned:
            return mtx1 * scale, mtx2
        scale = scale * norm2 / norm1
        t = mu2 - scale * np.dot(mu1, R)
        R = R.transpose()
    return scale, R, t


def compute_camera_align_umeyama(p_unalgined: Tensor, p_target: Tensor, known_scale=False, yaw_only=False):
    """  p_target = s * R @ p_unalgined + t
    Implementation of the paper: S. Umeyama,
    Least-Squares Estimation of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    Args:
        p_unalgined: shape: [N, 3]
        p_target: shape: [N, 3]
        known_scale: Ture: s=1
        yaw_only:
    Returns:
        - s: shape: [1]
        - R: shape: [3, 3]
        - t: shape [3]
    """
    if isinstance(p_unalgined, Tensor):
        # subtract mean
        mu_D = p_unalgined.mean(dim=0)
        mu_M = p_target.mean(dim=0)
        data_zerocentered = p_unalgined - mu_D
        model_zerocentered = p_target - mu_M
        n = p_target.shape[0]
        # correlation
        C = 1.0 / n * (model_zerocentered.T @ data_zerocentered)
        sigma2 = 1.0 / n * (data_zerocentered * data_zerocentered).sum()
        U_svd, D_svd, V_svd = torch.linalg.svd(C)

        D_svd = torch.diag(D_svd)
        V_svd = V_svd.T

        S = torch.eye(3, device=p_target.device, dtype=p_target.dtype)
        if torch.linalg.det(U_svd) * torch.linalg.det(V_svd) < 0:
            S[2, 2] = -1

        if yaw_only:
            rot_C = data_zerocentered.T @ model_zerocentered
            # get best yaw, maximize trace(Rz(theta) * C)
            theta = torch.pi * 0.5 - torch.arctan2(rot_C[0] + rot_C[1, 1], rot_C[0, 1] - rot_C[1, 0])
            R = axis_angle_to_R(theta.new_tensor([0, 0, 1]), theta)
        else:
            R = U_svd @ (S @ V_svd.T)

        s = 1 if known_scale else (1.0 / sigma2 * torch.trace(D_svd @ S))
        t = mu_M - s * (R @ mu_D)

        dtype = p_unalgined.dtype
        return s.to(dtype), R.to(dtype), t.to(dtype)
    else:
        # subtract mean
        mu_D = p_unalgined.mean(0)
        mu_M = p_target.mean(0)
        data_zerocentered = p_unalgined - mu_D
        model_zerocentered = p_target - mu_M
        n = np.shape(p_target)[0]

        # correlation
        C = 1.0 / n * np.dot(model_zerocentered.transpose(), data_zerocentered)
        sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
        U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

        D_svd = np.diag(D_svd)
        V_svd = np.transpose(V_svd)

        S = np.eye(3)
        if np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0:
            S[2, 2] = -1

        if yaw_only:
            rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
            # get best yaw, maximize trace(Rz(theta) * C)
            theta = np.pi * 0.5 - np.arctan2(rot_C[0] + rot_C[1, 1], rot_C[0, 1] - rot_C[1, 0])
            R = rotation_np.axis_angle_to_R(np.array([0, 0, 1]), theta)
        else:
            R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

        if known_scale:
            s = 1
        else:
            s = 1.0 / sigma2 * np.trace(np.dot(D_svd, S))

        t = mu_M - s * np.dot(R, mu_D)

        dtype = p_unalgined.dtype
        return s.astype(dtype), R.astype(dtype), t.astype(dtype)


def align_camera_poses(poses_unaligned: Tensor, target: Tensor, poses: Tensor = None, method='SIM3'):
    """align poses_unalign to target, ie, poses_unalign' → target
    Args:
        poses_unaligned: optimizer poses, [N, 4, 4]
        target: target pose, [N, 4, 4]
        poses: poses need to aglin, None or [..., 3/4, 4]
        method: 'SIM3' or 'SE3', 'posyaw', 'none'
    Returns:
        algined poses, [N, 4, 4] simlar to poses
    """
    assert poses_unaligned.shape == target.shape and poses_unaligned.ndim == 3 and poses_unaligned.shape[1:] == (4, 4)
    # s, R, t = compute_camera_align(poses_unaligned[:, :3, 3], target[:, :3, 3])
    if poses is None:
        poses = poses_unaligned
    method = method.lower()
    if method == 'sim3':
        s, R, t = compute_camera_align_umeyama(poses_unaligned[:, :3, 3], target[:, :3, 3])
    elif method == 'se3':
        s, R, t = compute_camera_align_umeyama(poses_unaligned[:, :3, 3], target[:, :3, 3], known_scale=True)
    elif method == 'posyaw':
        s, R, t = compute_camera_align_umeyama(poses_unaligned[:, :3, 3], target[:, :3, 3], True, True)
    else:
        return poses

    pose_aligned = poses.clone()
    pose_aligned[..., :3, :3] = R @ poses[..., :3, :3]
    pose_aligned[..., :3, 3] = s * poses[..., :3, 3] @ R.T + t
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
    """scale and translate camera poses"""
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


def rotation_error(pose_error: Union[np.ndarray, Tensor]):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[..., 0, 0]
    b = pose_error[..., 1, 1]
    c = pose_error[..., 2, 2]
    d = 0.5 * (a + b + c - 1.0)
    if isinstance(d, Tensor):
        rot_error = d.clamp(-1., 1.0).arccos_()
    else:
        rot_error = np.arccos(np.clip(d, -1.0, 1.0))
    return rot_error


def compute_RPE(pred: Union[np.ndarray, Tensor], gt: Union[np.ndarray, Tensor]):
    """ compute Relative Pose Error
    Args:
        gt: ground-truth poses (Tv2w), shape: [..., 4, 4]
        pred: predicted poses (Tv2w), shape: same to gt
    Return:
        - RPE_t
        - RPE_r
    """
    if isinstance(gt, Tensor):
        gt_rel_ = gt[..., 1:, :, :].inverse() @ gt[..., :-1, :, :]
        pred_rel = pred[..., :-1, :, :].inverse() @ pred[..., 1:, :, :]
        rel_err = gt_rel_ @ pred_rel
        rpe_trans = rel_err[..., :3, 3].norm(dim=-1).mean()
    else:
        gt_rel_ = np.linalg.inv(gt[..., 1:, :, :]) @ gt[..., :-1, :, :]
        pred_rel = np.linalg.inv(pred[..., :-1, :, :]) @ pred[..., 1:, :, :]
        rel_err = gt_rel_ @ pred_rel
        rpe_trans = np.linalg.norm(rel_err[..., :3, 3], axis=-1).mean()
    rpe_rot = rotation_error(rel_err).mean()
    return rpe_trans, rpe_rot


def compute_ATE(pred: Union[np.ndarray, Tensor], gt: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """Compute RMSE of ATE (Absolute Trajectory Error)
    Args:
        pred: predicted poses (Tv2w), shape, [..., 4, 4]
        gt: ground-truth poses (Tv2w), shape: [..., 4, 4]
    Returns:
        ATE, shape: 1
    """
    assert gt.shape == pred.shape
    error = gt[..., :3, 3] - pred[..., :3, 3]
    if isinstance(error, Tensor):
        ate = error.square().sum(dim=-1).mean().sqrt()
    else:
        ate = np.sqrt(np.mean(np.sum(error ** 2, axis=-1)))
    return ate
