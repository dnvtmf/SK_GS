"""
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code: https://github.com/graphdeco-inria/gaussian-splatting
"""

import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from torch import Tensor


def render_gs_offical(
    points: Tensor,
    opacity: Tensor,
    raster_settings: GaussianRasterizationSettings,
    scales: Tensor = None,
    rotations: Tensor = None,
    covariance: Tensor = None,
    sh_features: Tensor = None,
    colors=None,
    extras=None,
    **kwargs
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(points, requires_grad=True) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = points
    means2D = screenspace_points
    assert extras is None and len(kwargs) == 0, f"Not supported"
    if rotations is not None:
        rotations = rotations[..., (3, 0, 1, 2)]  # (x, y, z w) -> (w, x, y, z)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    outputs = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=sh_features,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=covariance)
    if len(outputs) == 4:
        rendered_image, radii, depth, alpha = outputs
    elif len(outputs) == 3:
        (rendered_image, radii, depth), alpha = outputs, None
    else:
        (rendered_image, radii), depth, alpha = outputs, None, None
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "images": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "alpha": alpha,
    }
