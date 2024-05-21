"""
Reference code: https://github.com/NVlabs/nvdiffrec/
"""
# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from torch import nn

from my_ext._C import get_C_function
from my_ext.utils.io.image import load_image, save_image_raw
from ..misc import dot, normalize, reflect
from ..xfm import xfm_vectors


# ----------------------------------------------------------------------------
# cubemap filter with filtering across edges

class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):  # noqa
        out = get_C_function('diffuse_cubemap_fwd')(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):  # noqa
        cubemap, = ctx.saved_variables
        cubemap_grad = get_C_function('diffuse_cubemap_bwd')(cubemap, dout)
        return cubemap_grad, None


def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out


class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):  # noqa
        out = get_C_function('specular_cubemap_fwd')(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):  # noqa
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = get_C_function('specular_cubemap_bwd')(cubemap, bounds, dout, ctx.roughness, ctx.theta_cutoff)
        return cubemap_grad, None, None, None


# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi / 2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness ** 4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = get_C_function('specular_bounds')(res, costheta[idx])

    return costheta[idx], bounds


__ndfBoundsDict = {}


def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(
        cubemap.shape)

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]


def cube_to_dir(s, x, y):
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
            indexing='ij'
        )
        v = normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap


def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
        indexing='ij'
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
    return dr.texture(
        cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube'
    )[0]


class cubemap_mip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cubemap):  # noqa
        # return util.avg_pool_nhwc(cubemap, (2, 2))
        y = cubemap.permute(0, 3, 1, 2)  # NHWC -> NCHW
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
        return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

    @staticmethod
    def backward(ctx, dout):  # noqa
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                indexing='ij'
            )
            v = normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube'
            )
        return out


######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################


class ImageBasedLight(nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(ImageBasedLight, self).__init__()
        self.mtx = None
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return ImageBasedLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) /
            (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) +
            len(self.specular) - 2
        )

    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx /
                         (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def forward(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        return self.shade(gb_pos, gb_normal, kd, ks, view_pos, specular)

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        wo = F.normalize(view_pos - gb_pos)

        if specular:
            roughness = ks[..., 1:2]  # y component
            metallic = ks[..., 2:3]  # z component
            spec_col = (1.0 - metallic) * 0.04 + kd * metallic
            diff_col = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = normalize(reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None:  # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = xfm_vectors(
                reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx
            ).view(*reflvec.shape)
            nrmvec = xfm_vectors(
                nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx
            ).view(*nrmvec.shape)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse * diff_col

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                _file = Path('~/Projects/Reconstruction/nvdiffrec/data/irrmaps/bsdf_256_256.bin').expanduser()
                self._FG_LUT = torch.as_tensor(
                    np.fromfile(_file, dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda'
                )
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(
                self.specular[0][None, ...],
                reflvec.contiguous(),
                mip=list(m[None, ...] for m in self.specular[1:]),
                mip_level_bias=miplevel[..., 0],
                filter_mode='linear-mipmap-linear',
                boundary_mode='cube'
            )

            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
            shaded_col += spec * reflectance

        return shaded_col * (1.0 - ks[..., 0:1])  # Modulate by hemisphere visibility

    @classmethod
    def load(cls, filepath: Path, scale=1.) -> 'ImageBasedLight':
        assert filepath.suffix == '.hdr'
        latlong_img = torch.tensor(load_image(filepath), dtype=torch.float32, device='cuda') * scale
        cubemap = latlong_to_cubemap(latlong_img, [512, 512])

        l = cls(cubemap)
        l.build_mips()
        return l

    def save(self, filepath: Path):
        color = cubemap_to_latlong(self.base, [512, 1024])
        save_image_raw(filepath.with_suffix('.hdr'), color)

    @classmethod
    def create(cls, base_res=512, scale=0.5, bias=0.25) -> 'ImageBasedLight':
        """create a random image based light"""
        base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32) * scale + bias
        return cls(base)
