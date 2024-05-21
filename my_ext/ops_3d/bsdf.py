# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import math

import pytest
import torch
import torch.utils.cpp_extension

from my_ext._C import get_C_function

NORMAL_THRESHOLD = 0.1
from .misc import dot, reflect, normalize


################################################################################
# Simple lambertian diffuse BSDF
################################################################################

def bsdf_lambert(nrm, wi):
    return torch.clamp(dot(nrm, wi), min=0.0) / math.pi


################################################################################
# Frostbite diffuse
################################################################################

def bsdf_frostbite(nrm, wi, wo, linearRoughness):
    wiDotN = dot(wi, nrm)
    woDotN = dot(wo, nrm)

    h = normalize(wo + wi)
    wiDotH = dot(wi, h)

    energyBias = 0.5 * linearRoughness
    energyFactor = 1.0 - (0.51 / 1.51) * linearRoughness
    f90 = energyBias + 2.0 * wiDotH * wiDotH * linearRoughness
    f0 = 1.0

    wiScatter = bsdf_fresnel_shlick(f0, f90, wiDotN)
    woScatter = bsdf_fresnel_shlick(f0, f90, woDotN)
    res = wiScatter * woScatter * energyFactor
    return torch.where((wiDotN > 0.0) & (woDotN > 0.0), res, torch.zeros_like(res))


################################################################################
# Phong specular, loosely based on mitsuba implementation
################################################################################

def bsdf_phong(nrm, wo, wi, N):
    dp_r = torch.clamp(dot(reflect(wo, nrm), wi), min=0.0, max=1.0)
    dp_l = torch.clamp(dot(nrm, wi), min=0.0, max=1.0)
    return (dp_r ** N) * dp_l * (N + 2) / (2 * math.pi)


################################################################################
# PBR's implementation of GGX specular
################################################################################

specular_epsilon = 1e-4


def bsdf_fresnel_shlick(f0, f90, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    return f0 + (f90 - f0) * (1.0 - _cosTheta) ** 5.0


def bsdf_ndf_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * math.pi)


def bsdf_lambda_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res


def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)


def bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08):
    _alpha = torch.clamp(alpha, min=min_roughness * min_roughness, max=1.0)
    alphaSqr = _alpha * _alpha

    h = normalize(wo + wi)
    woDotN = dot(wo, nrm)
    wiDotN = dot(wi, nrm)
    woDotH = dot(wo, h)
    nDotH = dot(nrm, h)

    D = bsdf_ndf_ggx(alphaSqr, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alphaSqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(col, 1, woDotH)

    w = F * D * G * 0.25 / torch.clamp(woDotN, min=specular_epsilon)

    frontfacing = (woDotN > specular_epsilon) & (wiDotN > specular_epsilon)
    return torch.where(frontfacing, w, torch.zeros_like(w))


def bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF):
    wo = normalize(view_pos - pos)
    wi = normalize(light_pos - pos)

    spec_str = arm[..., 0:1]  # x component
    roughness = arm[..., 1:2]  # y component
    metallic = arm[..., 2:3]  # z component
    ks = (0.04 * (1.0 - metallic) + kd * metallic) * (1 - spec_str)
    kd = kd * (1.0 - metallic)

    if BSDF == 0:
        diffuse = kd * bsdf_lambert(nrm, wi)
    else:
        diffuse = kd * bsdf_frostbite(nrm, wi, wo, roughness)
    specular = bsdf_pbr_specular(ks, nrm, wo, wi, roughness * roughness, min_roughness=min_roughness)
    return diffuse + specular


# ----------------------------------------------------------------------------
# Internal kernels, just used for testing functionality

# noinspection PyMethodOverriding
class _fresnel_shlick_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, f90, cosTheta):
        out = get_C_function("fresnel_shlick_fwd")(f0, f90, cosTheta, False)
        ctx.save_for_backward(f0, f90, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        f0, f90, cosTheta = ctx.saved_variables
        return get_C_function("fresnel_shlick_bwd")(f0, f90, cosTheta, dout) + (None,)


def _fresnel_shlick(f0, f90, cosTheta, use_python=False):
    if use_python:
        out = bsdf_fresnel_shlick(f0, f90, cosTheta)
    else:
        out = _fresnel_shlick_func.apply(f0, f90, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _fresnel_shlick contains inf or NaN"
    return out


# noinspection PyMethodOverriding
class _ndf_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = get_C_function("ndf_ggx_fwd")(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return get_C_function("ndf_ggx_bwd")(alphaSqr, cosTheta, dout) + (None,)


def _ndf_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_ndf_ggx(alphaSqr, cosTheta)
    else:
        out = _ndf_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _ndf_ggx contains inf or NaN"
    return out


# noinspection PyMethodOverriding
class _lambda_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = get_C_function("lambda_ggx_fwd")(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return get_C_function("lambda_ggx_bwd")(alphaSqr, cosTheta, dout) + (None,)


def _lambda_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_lambda_ggx(alphaSqr, cosTheta)
    else:
        out = _lambda_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _lambda_ggx contains inf or NaN"
    return out


# noinspection PyMethodOverriding
class _masking_smith_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosThetaI, cosThetaO):
        ctx.save_for_backward(alphaSqr, cosThetaI, cosThetaO)
        out = get_C_function("masking_smith_fwd")(alphaSqr, cosThetaI, cosThetaO, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosThetaI, cosThetaO = ctx.saved_variables
        return get_C_function("masking_smith_bwd")(alphaSqr, cosThetaI, cosThetaO, dout) + (None,)


def _masking_smith(alphaSqr, cosThetaI, cosThetaO, use_python=False):
    if use_python:
        out = bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO)
    else:
        out = _masking_smith_func.apply(alphaSqr, cosThetaI, cosThetaO)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _masking_smith contains inf or NaN"
    return out


# ----------------------------------------------------------------------------
# BSDF functions
# noinspection PyMethodOverriding
class _lambert_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi):
        out = get_C_function("lambert_fwd")(nrm, wi, False)
        ctx.save_for_backward(nrm, wi)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi = ctx.saved_variables
        return get_C_function("lambert_bwd")(nrm, wi, dout) + (None,)


def lambert(nrm, wi, use_python=False):
    '''Lambertian bsdf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_lambert(nrm, wi)
    else:
        out = _lambert_func.apply(nrm, wi)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out


# noinspection PyMethodOverriding
class _frostbite_diffuse_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi, wo, linearRoughness):
        out = get_C_function("frostbite_fwd")(nrm, wi, wo, linearRoughness, False)
        ctx.save_for_backward(nrm, wi, wo, linearRoughness)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi, wo, linearRoughness = ctx.saved_variables
        return get_C_function("frostbite_bwd")(nrm, wi, wo, linearRoughness, dout) + (None,)


def frostbite_diffuse(nrm, wi, wo, linearRoughness, use_python=False):
    '''Frostbite, normalized Disney Diffuse bsdf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        wo: World space camera vector.
        linearRoughness: Material roughness
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_frostbite(nrm, wi, wo, linearRoughness)
    else:
        out = _frostbite_diffuse_func.apply(nrm, wi, wo, linearRoughness)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out


# noinspection PyMethodOverriding
class _pbr_specular_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, col, nrm, wo, wi, alpha, min_roughness):
        ctx.save_for_backward(col, nrm, wo, wi, alpha)
        ctx.min_roughness = min_roughness
        out = get_C_function("pbr_specular_fwd")(col, nrm, wo, wi, alpha, min_roughness, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        col, nrm, wo, wi, alpha = ctx.saved_variables
        return get_C_function("pbr_specular_bwd")(col, nrm, wo, wi, alpha, ctx.min_roughness, dout) + (None, None)


def pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08, use_python=False):
    '''Physically-based specular bsdf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        col: Specular lobe color
        nrm: World space shading normal.
        wo: World space camera vector.
        wi: World space light vector
        alpha: Specular roughness parameter with shape [minibatch_size, height, width, 1]
        min_roughness: Scalar roughness clamping threshold

        use_python: Use PyTorch implementation (for validation)
    Returns:
        Shaded specular color
    '''

    if use_python:
        out = bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=min_roughness)
    else:
        out = _pbr_specular_func.apply(col, nrm, wo, wi, alpha, min_roughness)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_specular contains inf or NaN"
    return out


# noinspection PyMethodOverriding
class _pbr_bsdf_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
        ctx.min_roughness = min_roughness
        ctx.BSDF = BSDF
        out = get_C_function("pbr_bsdf_fwd")(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_variables
        return get_C_function("pbr_bsdf_bwd")(kd,
            arm,
            pos,
            nrm,
            view_pos,
            light_pos,
            ctx.min_roughness,
            ctx.BSDF,
            dout) + (
            None, None, None)


def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, bsdf="lambert", use_python=False):
    '''Physically-based bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold
        bsdf: Controls diffuse BSDF, can be either 'lambert' or 'frostbite'

        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded color.
    '''

    BSDF = 0
    if bsdf == 'frostbite':
        BSDF = 1

    if use_python:
        out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)
    else:
        out = _pbr_bsdf_func.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_bsdf contains inf or NaN"
    return out


######################### test

_RES = 4
_DTYPE = torch.float32


def relative_loss(name, ref, cuda):
    ref = ref.float()
    cuda = cuda.float()
    print(name, torch.max(torch.abs(ref - cuda) / torch.abs(ref + 1e-7)).item())


def test_schlick():
    f0_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    f0_ref = f0_cuda.clone().detach().requires_grad_(True)
    f90_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    f90_ref = f90_cuda.clone().detach().requires_grad_(True)
    cosT_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True) * 2.0
    cosT_cuda = cosT_cuda.clone().detach().requires_grad_(True)
    cosT_ref = cosT_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda')

    ref = _fresnel_shlick(f0_ref, f90_ref, cosT_ref, use_python=True)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = _fresnel_shlick(f0_cuda, f90_cuda, cosT_cuda)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Fresnel shlick")
    print("-------------------------------------------------------------")
    relative_loss("res:", ref, cuda)
    relative_loss("f0:", f0_ref.grad, f0_cuda.grad)
    relative_loss("f90:", f90_ref.grad, f90_cuda.grad)
    relative_loss("cosT:", cosT_ref.grad, cosT_cuda.grad)


def test_ndf_ggx():
    alphaSqr_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True)
    alphaSqr_cuda = alphaSqr_cuda.clone().detach().requires_grad_(True)
    alphaSqr_ref = alphaSqr_cuda.clone().detach().requires_grad_(True)
    cosT_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True) * 3.0 - 1
    cosT_cuda = cosT_cuda.clone().detach().requires_grad_(True)
    cosT_ref = cosT_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda')

    ref = _ndf_ggx(alphaSqr_ref, cosT_ref, use_python=True)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = _ndf_ggx(alphaSqr_cuda, cosT_cuda)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Ndf GGX")
    print("-------------------------------------------------------------")
    relative_loss("res:", ref, cuda)
    relative_loss("alpha:", alphaSqr_ref.grad, alphaSqr_cuda.grad)
    relative_loss("cosT:", cosT_ref.grad, cosT_cuda.grad)


def test_lambda_ggx():
    alphaSqr_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True)
    alphaSqr_ref = alphaSqr_cuda.clone().detach().requires_grad_(True)
    cosT_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True) * 3.0 - 1
    cosT_cuda = cosT_cuda.clone().detach().requires_grad_(True)
    cosT_ref = cosT_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda')

    ref = _lambda_ggx(alphaSqr_ref, cosT_ref, use_python=True)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = _lambda_ggx(alphaSqr_cuda, cosT_cuda)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Lambda GGX")
    print("-------------------------------------------------------------")
    relative_loss("res:", ref, cuda)
    relative_loss("alpha:", alphaSqr_ref.grad, alphaSqr_cuda.grad)
    relative_loss("cosT:", cosT_ref.grad, cosT_cuda.grad)


def test_masking_smith():
    alphaSqr_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True)
    alphaSqr_ref = alphaSqr_cuda.clone().detach().requires_grad_(True)
    cosThetaI_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True)
    cosThetaI_ref = cosThetaI_cuda.clone().detach().requires_grad_(True)
    cosThetaO_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True)
    cosThetaO_ref = cosThetaO_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda')

    ref = _masking_smith(alphaSqr_ref, cosThetaI_ref, cosThetaO_ref, use_python=True)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = _masking_smith(alphaSqr_cuda, cosThetaI_cuda, cosThetaO_cuda)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Smith masking term")
    print("-------------------------------------------------------------")
    relative_loss("res:", ref, cuda)
    relative_loss("alpha:", alphaSqr_ref.grad, alphaSqr_cuda.grad)
    relative_loss("cosThetaI:", cosThetaI_ref.grad, cosThetaI_cuda.grad)
    relative_loss("cosThetaO:", cosThetaO_ref.grad, cosThetaO_cuda.grad)


def test_lambert():
    normals_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    normals_ref = normals_cuda.clone().detach().requires_grad_(True)
    wi_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    wi_ref = wi_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda')

    ref = lambert(normals_ref, wi_ref, use_python=True)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = lambert(normals_cuda, wi_cuda)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Lambert")
    print("-------------------------------------------------------------")
    relative_loss("res:", ref, cuda)
    relative_loss("nrm:", normals_ref.grad, normals_cuda.grad)
    relative_loss("wi:", wi_ref.grad, wi_cuda.grad)


def test_frostbite():
    normals_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    normals_ref = normals_cuda.clone().detach().requires_grad_(True)
    wi_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    wi_ref = wi_cuda.clone().detach().requires_grad_(True)
    wo_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    wo_ref = wo_cuda.clone().detach().requires_grad_(True)
    rough_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True)
    rough_ref = rough_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda')

    ref = frostbite_diffuse(normals_ref, wi_ref, wo_ref, rough_ref, use_python=True)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = frostbite_diffuse(normals_cuda, wi_cuda, wo_cuda, rough_cuda)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Frostbite")
    print("-------------------------------------------------------------")
    relative_loss("res:", ref, cuda)
    relative_loss("nrm:", normals_ref.grad, normals_cuda.grad)
    relative_loss("wo:", wo_ref.grad, wo_cuda.grad)
    relative_loss("wi:", wi_ref.grad, wi_cuda.grad)
    relative_loss("rough:", rough_ref.grad, rough_cuda.grad)


def test_pbr_specular():
    col_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    col_ref = col_cuda.clone().detach().requires_grad_(True)
    nrm_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    nrm_ref = nrm_cuda.clone().detach().requires_grad_(True)
    wi_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    wi_ref = wi_cuda.clone().detach().requires_grad_(True)
    wo_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    wo_ref = wo_cuda.clone().detach().requires_grad_(True)
    alpha_cuda = torch.rand(1, _RES, _RES, 1, dtype=_DTYPE, device='cuda', requires_grad=True)
    alpha_ref = alpha_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda')

    ref = pbr_specular(col_ref, nrm_ref, wo_ref, wi_ref, alpha_ref, use_python=True)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = pbr_specular(col_cuda, nrm_cuda, wo_cuda, wi_cuda, alpha_cuda)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Pbr specular")
    print("-------------------------------------------------------------")

    relative_loss("res:", ref, cuda)
    if col_ref.grad is not None:
        relative_loss("col:", col_ref.grad, col_cuda.grad)
    if nrm_ref.grad is not None:
        relative_loss("nrm:", nrm_ref.grad, nrm_cuda.grad)
    if wi_ref.grad is not None:
        relative_loss("wi:", wi_ref.grad, wi_cuda.grad)
    if wo_ref.grad is not None:
        relative_loss("wo:", wo_ref.grad, wo_cuda.grad)
    if alpha_ref.grad is not None:
        relative_loss("alpha:", alpha_ref.grad, alpha_cuda.grad)


@pytest.mark.parametrize("bsdf", ['lambert', 'frostbite'])
def test_pbr_bsdf(bsdf):
    kd_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    kd_ref = kd_cuda.clone().detach().requires_grad_(True)
    arm_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    arm_ref = arm_cuda.clone().detach().requires_grad_(True)
    pos_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    pos_ref = pos_cuda.clone().detach().requires_grad_(True)
    nrm_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    nrm_ref = nrm_cuda.clone().detach().requires_grad_(True)
    view_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    view_ref = view_cuda.clone().detach().requires_grad_(True)
    light_cuda = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda', requires_grad=True)
    light_ref = light_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(1, _RES, _RES, 3, dtype=_DTYPE, device='cuda')

    ref = pbr_bsdf(kd_ref, arm_ref, pos_ref, nrm_ref, view_ref, light_ref, use_python=True, bsdf=bsdf)
    ref_loss = torch.nn.MSELoss()(ref, target)
    ref_loss.backward()

    cuda = pbr_bsdf(kd_cuda, arm_cuda, pos_cuda, nrm_cuda, view_cuda, light_cuda, bsdf=bsdf)
    cuda_loss = torch.nn.MSELoss()(cuda, target)
    cuda_loss.backward()

    print("-------------------------------------------------------------")
    print("    Pbr BSDF")
    print("-------------------------------------------------------------")

    relative_loss("res:", ref, cuda)
    if kd_ref.grad is not None:
        relative_loss("kd:", kd_ref.grad, kd_cuda.grad)
    if arm_ref.grad is not None:
        relative_loss("arm:", arm_ref.grad, arm_cuda.grad)
    if pos_ref.grad is not None:
        relative_loss("pos:", pos_ref.grad, pos_cuda.grad)
    if nrm_ref.grad is not None:
        relative_loss("nrm:", nrm_ref.grad, nrm_cuda.grad)
    if view_ref.grad is not None:
        relative_loss("view:", view_ref.grad, view_cuda.grad)
    if light_ref.grad is not None:
        relative_loss("light:", light_ref.grad, light_cuda.grad)


@pytest.mark.parametrize("BATCH,RES,ITR", [(1, 512, 1000), (16, 512, 1000), (1, 2048, 1000)])
def test_bsdf(BATCH, RES, ITR):
    DTYPE = torch.float32
    kd_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    kd_ref = kd_cuda.clone().detach().requires_grad_(True)
    arm_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    arm_ref = arm_cuda.clone().detach().requires_grad_(True)
    pos_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    pos_ref = pos_cuda.clone().detach().requires_grad_(True)
    nrm_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    nrm_ref = nrm_cuda.clone().detach().requires_grad_(True)
    view_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    view_ref = view_cuda.clone().detach().requires_grad_(True)
    light_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    light_ref = light_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(BATCH, RES, RES, 3, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    pbr_bsdf(kd_cuda, arm_cuda, pos_cuda, nrm_cuda, view_cuda, light_cuda)

    print("--- Testing: [%d, %d, %d] ---" % (BATCH, RES, RES))

    start.record()
    for i in range(ITR):
        ref = pbr_bsdf(kd_ref, arm_ref, pos_ref, nrm_ref, view_ref, light_ref, use_python=True)
    end.record()
    torch.cuda.synchronize()
    print("Pbr BSDF python:", start.elapsed_time(end))

    start.record()
    for i in range(ITR):
        cuda = pbr_bsdf(kd_cuda, arm_cuda, pos_cuda, nrm_cuda, view_cuda, light_cuda)
    end.record()
    torch.cuda.synchronize()
    print("Pbr BSDF cuda:", start.elapsed_time(end))
