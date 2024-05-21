"""
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code: https://github.com/graphdeco-inria/gaussian-splatting
"""
from typing import NamedTuple, Tuple, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd

from my_ext._C import get_C_function


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


class RasterizeBuffer(NamedTuple):
    W: int
    """output image width"""
    H: int
    """output image width"""
    P: int
    """number of gaussian"""
    R: int
    """number of rendered"""
    geomBuffer: Tensor
    binningBuffer: Tensor
    imgBuffer: Tensor


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    detach_other_extra: bool = False
    """other extra是否与渲染隔离开, 即对 means2D, opacities等的梯度有贡献"""
    colmap: bool = False


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        extras,
        raster_settings: GaussianRasterizationSettings,
        *other_extras
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.image_height, raster_settings.image_width,
            raster_settings.tanfovx, raster_settings.tanfovy,
            raster_settings.sh_degree,
            raster_settings.scale_modifier,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.colmap,
            raster_settings.viewmatrix, raster_settings.projmatrix, raster_settings.campos,  # tensor params
            means3D, opacities, sh, scales, rotations, extras, colors_precomp, cov3Ds_precomp,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, opaticy, radii, geomBuffer, binningBuffer, imgBuffer, out_extra = \
                    get_C_function('rasterize_gaussians')(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, opaticy, radii, geomBuffer, binningBuffer, imgBuffer, out_extra = \
                get_C_function('rasterize_gaussians')(*args)

        buffer = RasterizeBuffer(
            raster_settings.image_width,
            raster_settings.image_height,
            len(opacities),
            num_rendered,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        pixel_extras = []
        for extra_i in other_extras:
            pixel_extras.append(get_C_function('gaussian_rasterize_extra_forward')(
                buffer.W, buffer.H, buffer.R, extra_i, geomBuffer, binningBuffer, imgBuffer
            ))
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp, cov3Ds_precomp, means3D, scales, rotations, sh, extras,
            geomBuffer, binningBuffer, imgBuffer,
            radii, opaticy,
            *other_extras
        )
        return (color, opaticy, out_extra, radii, buffer, *pixel_extras)

    @staticmethod
    @custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_color, grad_out_opacity, grad_out_extra, *grad_outputs):  # noqa
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, cov3Ds_precomp, means3D, scales, rotations, sh, extras = ctx.saved_tensors[:7]
        geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors[7:10]
        radii, opaticy = ctx.saved_tensors[10:12]
        other_extras = ctx.saved_tensors[12:]

        grad_means2D = None
        grad_conic = None
        grad_opacity = None
        grad_extras = []
        grad_outputs = grad_outputs[2:]
        assert len(grad_outputs) == len(other_extras)
        for i, extra_i in enumerate(other_extras):
            if grad_outputs[i] is None:
                grad_extras.append(None)
                continue
            grad_extra_i, grad_means2D, grad_conic, grad_opacity = get_C_function('gaussian_rasterize_extra_backward')(
                raster_settings.image_width, raster_settings.image_height, num_rendered,
                extra_i, opaticy,  # inputs, outputs
                grad_outputs[i],  # grad_outputs
                geomBuffer, binningBuffer, imgBuffer,  # buffer
                grad_means2D, grad_conic, grad_opacity  # grad_inputs
            )
            grad_extras.append(grad_extra_i)
        if ctx.raster_settings.detach_other_extra:
            grad_means2D, grad_conic, grad_opacity = None, None, None
        args = (
            raster_settings.scale_modifier, raster_settings.tanfovx, raster_settings.tanfovy, raster_settings.sh_degree,
            raster_settings.debug, raster_settings.colmap,  # const
            raster_settings.viewmatrix, raster_settings.projmatrix, raster_settings.campos,  # tensor parameters
            means3D, colors_precomp, extras, scales, rotations, cov3Ds_precomp, sh,  # inputs
            num_rendered, radii, opaticy,  # outputs
            grad_out_color, grad_out_opacity, grad_out_extra,  # grad_outputs
            grad_means2D, grad_conic, grad_opacity,  # grad_inputs
            geomBuffer, binningBuffer, imgBuffer,  # buffer
        )
        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh,
                 grad_scales, grad_rotations, grad_extra) = get_C_function("rasterize_gaussians_backward")(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh,
             grad_scales, grad_rotations, grad_extra) = get_C_function("rasterize_gaussians_backward")(*args)
        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_extra,
            None,
            *grad_extras
        )
        return grads


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    extras,
    raster_settings: GaussianRasterizationSettings,
    **kwargs
):
    outputs = _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        extras,
        raster_settings,
        *kwargs.values()
    )
    color, opaticy, out_extra, radii, buffer = outputs[:5]
    output_extras = {k: v for k, v in zip(kwargs.keys(), outputs[5:])}
    return color, opaticy, out_extra, radii, buffer, output_extras


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = get_C_function('mark_visible')(positions, raster_settings.viewmatrix, raster_settings.projmatrix)

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        extras=None,
        **kwargs
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            extras,
            raster_settings,
            **kwargs
        )


def render(
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

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_opacity, rendered_extra, radii, buffer, outputs_extras = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=sh_features,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=covariance,
        extras=extras,
        **kwargs
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "images": rendered_image,
        "opacity": rendered_opacity,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        'extras': rendered_extra,
        'buffer': buffer,
        **outputs_extras
    }


def topk_weights(topk, buffer: RasterizeBuffer) -> Tuple[Tensor, Tensor]:
    """return topk indices and weights """
    return get_C_function('gaussian_topk_weights')(
        topk, buffer.W, buffer.H, buffer.P, buffer.R, buffer.geomBuffer, buffer.binningBuffer, buffer.imgBuffer
    )


def debug_backward():
    dump = torch.load('snapshot_bw.dump', map_location='cpu')
    dump = list(dump)  # debug
    print(dump[4])
    dump[4] = True
    (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh,
     grad_scales, grad_rotations, grad_extra) = get_C_function("rasterize_gaussians_backward")(*dump)
    print('No Error')


if __name__ == '__main__':
    debug_backward()
