"""
估计相机内参和外参
"""
import numpy as np
import torch
from torch import Tensor

__all__ = ['estimate_focal_knowing_depth']


def estimate_focal_knowing_depth(pts3d: Tensor, pp: Tensor = None, focal_mode='median', min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error

    Args:
        pts3d: shape [B, H, W, 3]
        pp: center pixel, shape [2] or [B, 2], default [0.5 * W, 0.5 * H]
        focal_mode: 'median' or 'weiszfeld'
        min_focal:
        max_focal:
    Returns:
        Tensor: esitmated focal, shape [B]
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3
    # centered pixel grid
    if pp is None:
        pp = pts3d.new_tensor([0.5 * W, 0.5 * H])
    pixels = torch.stack(torch.meshgrid(
        torch.arange(0, W, device=pts3d.device), torch.arange(0, H, device=pts3d.device), indexing='xy'
    ), dim=-1).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == 'median':
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == 'weiszfeld':
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for step in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f'bad focal_mode={focal_mode}')

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    return focal


def test_estimate_focal_knowing_depth():
    focal = torch.rand(1) * 100 + 100  # [100, 200]
    W, H = 256, 128
    z = torch.rand(H, W) + 0.1  # [0.1, 1.1]

    from my_ext.ops_3d.coord_trans import camera_intrinsics
    # K = camera_intrinsics(focal, size=(W, H))
    K = torch.tensor([[focal, 0, W * 0.5], [0, focal, H * 0.5], [0, 0, 1]])
    print(K)
    from my_ext.ops_3d.xfm import pixel2points
    points = pixel2points(z, K)[None]
    print(points.shape)
    focal1 = estimate_focal_knowing_depth(points, focal_mode='median')
    focal2 = estimate_focal_knowing_depth(points, focal_mode='weiszfeld')
    print(focal1, focal2, focal)
