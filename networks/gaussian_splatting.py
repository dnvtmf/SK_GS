"""
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code: https://github.com/graphdeco-inria/gaussian-splatting


Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.

This software is free for non-commercial, research and evaluation use
under the terms of the LICENSE.md file.

For inquiries contact  george.drettakis@inria.fr
"""
import math
import os
from typing import NamedTuple, Mapping, Any, Dict, Optional, Union, Sequence, Type

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from torch import nn, Tensor

import my_ext
from my_ext import utils, get_C_function, ops_3d
from lietorch import SO3

from networks.encoders.sphere_harmonics import RGB2SH, eval_sh, SH2RGB
from networks.losses import LossDict
from networks.GS_utils import build_covariance_from_scaling_rotation as ComputeCov3D
from networks.renderer.gaussian_render_origin import (
    GaussianRasterizationSettings as GaussianRasterizationSettings_offical,
    render_gs_offical,
)
from networks.renderer.gaussian_render import (
    GaussianRasterizationSettings, render,
)

from my_ext import Registry
import logging

NETWORKS = Registry()  # type: Registry[Type[GaussianSplatting]]


class BasicPointCloud(NamedTuple):  # TODO: replace by PointClouds
    points: np.array
    colors: np.array
    normals: np.array


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


@NETWORKS.register('Gaussian')
class GaussianSplatting(nn.Module):
    param_names_map = {
        '_xyz': 'xyz',
        '_features_dc': 'f_dc',
        '_features_rest': 'f_rest',
        '_scaling': 'scaling',
        '_rotation': 'rotation',
        '_opacity': 'opacity'
    }
    max_radii2D: Tensor
    xyz_gradient_accum: Tensor
    denom: Tensor

    def __init__(
        self,
        sh_degree: int = 3,
        convert_SHs_python=False,
        compute_cov3D=False,
        loss_cfg: dict = None,
        use_so3=False,
        use_official_gaussians_render=True,
        adaptive_control_cfg=None,
        lr_position_init=0.16,
        lr_position_final=0.0016,
        lr_position_delay_mult=0.01,
        lr_position_max_steps=30_000,
        lr_feature=2.5,
        lr_opacity=50.,
        lr_scaling=5.0,
        lr_rotation=1.0,
        **kwargs
    ):
        super().__init__()
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D = compute_cov3D
        self.register_buffer('_active_sh_degree', torch.tensor(0, dtype=torch.int))
        self.max_sh_degree = sh_degree
        self.use_so3 = use_so3
        self.use_official_gaussians_render = use_official_gaussians_render
        if self.use_official_gaussians_render:
            self.RasterizationSettings = GaussianRasterizationSettings_offical
            self.gs_rasterizer = render_gs_offical
        else:
            self.RasterizationSettings = GaussianRasterizationSettings
            self.gs_rasterizer = render

        self._xyz = torch.empty(0, 3)
        self._features_dc = torch.empty(0, 1, 3)
        self._features_rest = torch.empty(0, (self.max_sh_degree + 1) ** 2 - 1, 3)
        self._scaling = torch.empty(0, 3)
        self._rotation = torch.empty(0, 3 if use_so3 else 4)
        self._opacity = torch.empty(0, 1)

        self.register_buffer('max_radii2D', torch.empty(0), persistent=False)
        self.register_buffer('xyz_gradient_accum', torch.empty(0, 1), persistent=False)
        self.register_buffer('denom', torch.empty(0, 1), persistent=False)

        self.lr_spatial_scale = 0.
        self.lr_position_init = lr_position_init
        self.lr_position_final = lr_position_final
        self.lr_position_delay_mult = lr_position_delay_mult
        self.lr_position_max_steps = lr_position_max_steps
        self.lr_feature = lr_feature
        self.lr_opacity = lr_opacity
        self.lr_scaling = lr_scaling
        self.lr_rotation = lr_rotation

        self.scaling_activation = torch.exp
        self.scaling_activation_inverse = torch.log
        self.covariance_activation = ComputeCov3D
        self.opacity_activation = torch.sigmoid
        self.opacity_activation_inverse = inverse_sigmoid
        self.rotation_activation = (lambda x: SO3.exp(x).vec()) if use_so3 else torch.nn.functional.normalize

        self.adaptive_control_cfg = utils.merge_dict(
            adaptive_control_cfg, {
                'densify_interval': [100, 500, 15000],
                'densify_grad_threshold': 0.0002,
                'densify_percent_dense': 0.01,

                'prune_interval': [100, 500, 15000],
                'prune_opacity_threshold': 0.005,
                'prune_max_screen_size': 20,
                'prune_max_percent': -1,
                'prune_percent_dense': 0.1,

                'opacity_reset_interval': [3000, 3000, -1],
            }
        )
        self.background_type = 'black'
        self._step = -1
        self.loss_funcs = LossDict(_net=self, **utils.merge_dict(loss_cfg))
        self._task: Optional[my_ext.IterableFramework] = None
        if len(kwargs) > 0:
            logging.warning(f"{self.__class__.__name__} unused parameters: {list(kwargs.keys())}")

    @property
    def active_sh_degree(self) -> int:
        return self._active_sh_degree.item()

    @active_sh_degree.setter
    def active_sh_degree(self, value):
        self._active_sh_degree = self._active_sh_degree.new_tensor(value)

    def set_from_dataset(self, dataset):
        self.background_type = dataset.background_type
        if hasattr(dataset, 'cameras_extent'):
            self.cameras_extent = getattr(dataset, 'cameras_extent')
        else:
            self.cameras_extent = ops_3d.get_center_and_diag(dataset.Tv2w[:, :3, 3])[1].item() * 1.1
        self.lr_spatial_scale = self.cameras_extent
        logging.info(f"set camera_radius to {self.cameras_extent} ")

    def create_from_pcd(self, pcd: BasicPointCloud, lr_spatial_scale: float = None):
        self.lr_spatial_scale = self.cameras_extent if lr_spatial_scale is None else lr_spatial_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        logging.info(f"Number of points at initialisation: {fused_point_cloud.shape[0]} ")

        distCUDA2 = get_C_function('simple_knn')
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        if self.use_so3:
            rots = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        else:
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, -1] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D.data = torch.zeros((self.points.shape[0]), device=self._xyz.device)

    def forward(self, scaling_modifier=1, campos: Tensor = None, **kwargs):
        points = self._xyz
        features = torch.cat((self._features_dc, self._features_rest), dim=1)
        outputs = {'points': points, 'opacity': self.opacity_activation(self._opacity)}
        if self.convert_SHs_python and campos is not None:
            outputs['colors'] = self.get_colors(features, points, campos)
        else:
            outputs['sh_features'] = features

        if self.compute_cov3D:
            outputs['covariance'] = self.covariance_activation(self.get_scaling * scaling_modifier, self._rotation)
        else:
            outputs['scales'] = self.scaling_activation(self._scaling)
            outputs['rotations'] = self.rotation_activation(self._rotation)
        return outputs

    def prepare_inputs(self, info, t=None, background: Tensor = None, scale_modifier=1.):
        Tw2v = info['Tw2v'].view(-1, 4, 4)
        Tv2c = info['Tv2c'].view(-1, 4, 4)
        campos = info['campos'].view(-1, 3)
        FoV = info['FoV'].view(-1, 2)
        if t is not None:
            t = t.view(-1)
        if info['Tw2v'].ndim == 2:
            if background is not None and background.ndim > 1:
                background = background.unsqueeze(0)
        outputs = []
        sh_degree = self.active_sh_degree  # if self.training else self.max_sh_degree
        for b in range(Tw2v.shape[0]):
            if self.use_official_gaussians_render:
                if background is None:
                    bg = Tw2v.new_zeros(3)
                else:
                    if background.numel() <= 3:
                        bg = background.view(-1).expand(3).contiguous()
                    else:
                        bg = background[b, ..., :3].view(-1, 3).mean(0)
            else:
                bg = (background[b] if background.ndim > 0 else background) if background is not None else None
            if self.use_official_gaussians_render:
                raster_settings = GaussianRasterizationSettings_offical(
                    image_width=info['size'][0],
                    image_height=info['size'][1],
                    tanfovx=math.tan(0.5 * FoV[b, 0]),
                    tanfovy=math.tan(0.5 * FoV[b, 1]),
                    scale_modifier=scale_modifier,
                    viewmatrix=Tw2v[b].transpose(-1, -2),
                    projmatrix=(Tv2c[b] @ Tw2v[b]).transpose(-1, -2),
                    sh_degree=sh_degree,
                    campos=campos[b],
                    prefiltered=False,
                    debug=False,
                    bg=bg
                )
            else:
                raster_settings = GaussianRasterizationSettings(
                    image_width=info['size'][0],
                    image_height=info['size'][1],
                    tanfovx=math.tan(0.5 * FoV[b, 0]),
                    tanfovy=math.tan(0.5 * FoV[b, 1]),
                    scale_modifier=scale_modifier,
                    Tw2v=Tw2v[b],
                    Tv2c=Tv2c[b],
                    sh_degree=sh_degree,
                    campos=campos[b],
                    is_opengl=ops_3d.get_coord_system() != 'opencv',
                    debug=False,
                )
            outputs.append((raster_settings, campos[b], bg, None if t is None else t[b]))
        return outputs

    def render(self, *args, info, background: Tensor = None, scale_modifier=1.0, **kwargs):
        outputs = {}
        inputs = self.prepare_inputs(info, None, background, scale_modifier)
        for b, (raster_settings, campos, bg, t) in enumerate(inputs):
            net_out = self(campos=campos, **kwargs)
            if 'hook' in kwargs:
                net_out = kwargs['hook'](net_out)
            outputs_b = self.gs_rasterizer(**net_out, raster_settings=raster_settings)
            images = outputs_b['images']
            if images.shape[0] == 3:
                images = torch.permute(images, (1, 2, 0))
            if not self.use_official_gaussians_render and background is not None:
                images = images + (1 - outputs_b['opacity'][..., None]) * bg.squeeze(0)
            outputs_b['images'] = images
            if b == 0:
                outputs = {k: [v] for k, v in outputs_b.items() if v is not None}
            else:
                for k, v in outputs_b.items():
                    if v is not None:
                        outputs[k].append(v)
        return {k: torch.stack(v, dim=0) if k != 'viewspace_points' else v for k, v in outputs.items()}

    def change_with_training_progress(self, step=0, num_steps=1, epoch=0, num_epochs=1):
        total_step = epoch * num_steps + step + 1
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if total_step % 1000 == 0 and self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree = self.active_sh_degree + 1
            logging.info(f"active_sh_degree={self.active_sh_degree} at step[{total_step}]")
        self._step = total_step

    def loss(self, inputs, outputs, targets, info):
        image = outputs['images']
        gt_img = targets['images'][..., :3]
        H, W, C = image.shape[-3:]
        image, gt_img = image.view(1, H, W, C), gt_img.view(1, H, W, C)
        losses = {'rgb': self.loss_funcs('image', image, gt_img), 'ssim': self.loss_funcs('ssim', image, gt_img)}
        return losses

    def construct_list_of_attributes(self):
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append('f_rest_{}'.format(i))
        attrs.append('opacity')
        for i in range(self._scaling.shape[1]):
            attrs.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            attrs.append('rot_{}'.format(i))
        # for name, opt_name in self.param_names_map.items():
        #     if name == 'xyz':
        #         continue
        #     channels = getattr(self, name)[0].numel()
        #     if channels == 1:
        #         attrs.append(opt_name)
        #     else:
        #         for i in range(channels):
        #             attrs.append(f"{opt_name}_{i}")
        return attrs

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (np.asarray(plydata.elements[0]["x"]),
             np.asarray(plydata.elements[0]["y"]),
             np.asarray(plydata.elements[0]["z"])), axis=1
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda")
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        features_extra = torch.tensor(features_extra, dtype=torch.float, device="cuda")
        self._features_rest = nn.Parameter(features_extra.transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        return plydata

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs):
        for name in self.param_names_map.keys():
            if name in state_dict:
                setattr(self, name, nn.Parameter(getattr(self, name).new_empty(state_dict[name].shape)))
                logging.debug(f'change the shape of parameters of {name}')
        for name in ['xyz_gradient_accum', 'denom', 'max_radii2D']:
            if name in state_dict:
                setattr(self, name, state_dict[name])
                logging.debug(f'change the shape of parameters of {name}')
        super().load_state_dict(state_dict, strict=strict, **kwargs)
        N = self._xyz.shape[0]
        self.max_radii2D.data = torch.zeros((N,), device=self._xyz.device)

    def get_params(self, cfg):
        lr = cfg.lr
        # yapf: off
        params_groups = [
            {'params': [self._xyz], 'lr': lr * self.lr_position_init * self.lr_spatial_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': lr * self.lr_feature, "name": "f_dc", 'fix': True},
            {'params': [self._features_rest], 'lr': lr * self.lr_feature / 20, "name": "f_rest", 'fix': True},
            {'params': [self._opacity], 'lr': lr * self.lr_opacity, "name": "opacity", 'fix': True},
            {'params': [self._scaling], 'lr': lr * self.lr_scaling, "name": "scaling", 'fix': True},
            {'params': [self._rotation], 'lr': lr * self.lr_rotation, "name": "rotation", 'fix': True},
        ]
        # yapf: on
        self.lr_scheduler = {'xyz': get_expon_lr_func(
            lr_init=lr * self.lr_position_init * self.lr_spatial_scale,
            lr_final=lr * self.lr_position_final * self.lr_spatial_scale,
            lr_delay_mult=self.lr_position_delay_mult,
            max_steps=self.lr_position_max_steps
        )}
        return params_groups

    def update_learning_rate(self, step: int = None, optimizer=None, *args, **kwargs):
        if optimizer is None:
            optimizer = self._task.optimizer
        if step is None:
            step = self._task.global_step + 1
        for group in optimizer.param_groups:
            if group.get('name', None) == 'xyz':
                group['lr'] = self.lr_scheduler['xyz'](step)
                return
        return

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def points(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def training_setup(self):
        # self.percent_dense = percent_dense
        num_points = self._xyz.shape[0]
        self.xyz_gradient_accum.data = self._xyz.new_zeros((num_points, 1))
        self.denom.data = self._xyz.new_zeros((num_points, 1))

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if isinstance(viewspace_point_tensor, Tensor):
            grad = viewspace_point_tensor.grad
        elif len(viewspace_point_tensor) == 1:
            grad = viewspace_point_tensor[0].grad
        else:
            grad = torch.zeros_like(viewspace_point_tensor[0])
            for p in viewspace_point_tensor:
                grad += p.grad
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @staticmethod
    def change_optimizer(
        optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
        tensor: Union[Tensor, Dict[str, Tensor]],
        name: Union[None, str, Sequence[str]] = None, op='replace', dim=0,
    ) -> Dict[str, Tensor]:
        """replace, prune, concat a tensor or tensor list in optimizer"""
        assert op in ['replace', 'prune', 'concat']
        optimizable_tensors = {}
        mask = None
        if name is not None:
            op_names = [name] if isinstance(name, str) else list(name)
        else:
            assert isinstance(tensor, dict)
            op_names = list(tensor.keys())
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                if ('name' not in group) or (group['name'] not in op_names):
                    continue
                old_tensor = group['params'][0]
                new_tensor = tensor[group['name']] if isinstance(tensor, dict) else tensor
                if op == 'concat':
                    group["params"][0] = nn.Parameter(torch.cat([old_tensor, new_tensor], dim).requires_grad_(True))
                elif op == 'prune':
                    mask = (slice(None),) * dim + (new_tensor,)
                    group["params"][0] = nn.Parameter(old_tensor[mask].requires_grad_(True))
                else:
                    group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

                stored_state = optimizer.state.get(old_tensor, None)
                if stored_state is None:
                    continue
                del optimizer.state[old_tensor]
                if op == 'concat':
                    stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], torch.zeros_like(new_tensor)], dim)
                    stored_state["exp_avg_sq"] = torch.cat(
                        [stored_state["exp_avg_sq"], torch.zeros_like(new_tensor)], dim
                    )
                elif op == 'prune':
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                else:  # replace
                    stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)
                optimizer.state[group['params'][0]] = stored_state
        return optimizable_tensors

    def prune_points(self, optimizer: torch.optim.Optimizer, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.change_optimizer(
            optimizer, valid_points_mask, list(self.param_names_map.values()), op='prune'
        )

        for param_name, opt_name in self.param_names_map.items():
            setattr(self, param_name, optimizable_tensors[opt_name])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, optimizer: torch.optim.Optimizer, mask=None, N=None, **kwargs):
        optimizable_tensors = self.change_optimizer(optimizer, tensor=kwargs, op='concat')
        for param_name, optim_name in self.param_names_map.items():
            setattr(self, param_name, optimizable_tensors[optim_name])

        num_points = self.points.shape[0]
        self.xyz_gradient_accum.data = self._xyz.new_zeros((num_points, 1))
        self.denom.data = self._xyz.new_zeros((num_points, 1))
        self.max_radii2D.data = self._xyz.new_zeros(num_points)
        return mask

    def densify_and_split(self, optimizer: torch.optim.Optimizer, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.points.shape[0]
        device = self.points.device
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.gt(torch.amax(self.get_scaling, dim=1), scene_extent)
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        if self.use_so3:
            rots = ops_3d.rotation.lie_to_R(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        else:
            rots = ops_3d.rotation.quaternion_to_R(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        # rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_params = {
            '_xyz': torch.bmm(rots, samples[..., None]).squeeze(-1) + self.points[selected_pts_mask].repeat(N, 1),
            '_scaling': self.scaling_activation_inverse(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        }
        for param_name, opt_name in self.param_names_map.items():
            if param_name == '_xyz' or param_name == '_scaling':
                new_params[opt_name] = new_params.pop(param_name)
            else:
                param = getattr(self, param_name)
                new_params[opt_name] = param[selected_pts_mask].repeat(N, *[1] * (param.ndim - 1))

        self.densification_postfix(optimizer, **new_params, mask=selected_pts_mask, N=N)

        prune_filter = torch.cat((selected_pts_mask, selected_pts_mask.new_zeros(N * selected_pts_mask.sum())))
        self.prune_points(optimizer, prune_filter)

    def densify_and_clone(self, optimizer: torch.optim.Optimizer, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.le(torch.amax(self.get_scaling, dim=1), scene_extent)
        )

        masked_params = {}
        for param_name, opt_name in self.param_names_map.items():
            masked_params[opt_name] = getattr(self, param_name)[selected_pts_mask]
        self.densification_postfix(optimizer, **masked_params, mask=selected_pts_mask)

    def densify(self, optimizer, max_grad: float, extent: float, densify_percent_dense=0.01):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(optimizer, grads, max_grad, densify_percent_dense * extent)
        self.densify_and_split(optimizer, grads, max_grad, densify_percent_dense * extent)

    def prune(self, optimizer, min_opacity: float, extent: float, max_screen_size: float, prune_percent_dense=0.1):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = torch.gt(self.get_scaling.amax(dim=1), prune_percent_dense * extent)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(optimizer, prune_mask)
        torch.cuda.empty_cache()

    def reset_opacity(self, optimizer):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        name = self.param_names_map['_opacity']
        self._opacity = self.change_optimizer(optimizer, opacities_new, name=name, op='replace')[name]

    @torch.no_grad()
    def adaptive_control(self, inputs, outputs, optimizer, step: int):
        radii = outputs['radii']
        viewspace_point_tensor = outputs['viewspace_points']
        cfg = self.adaptive_control_cfg
        densify_interval = self.adaptive_control_cfg['densify_interval']
        prune_interval = self.adaptive_control_cfg['prune_interval']
        max_step = max(densify_interval[2], prune_interval[2])
        step = step + 1  # start from 1
        if step < max_step:
            if radii.ndim == 2:
                radii = radii.amax(dim=0)
            mask = radii > 0
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[mask] = torch.max(self.max_radii2D[mask], radii[mask])
            self.add_densification_stats(viewspace_point_tensor, mask)

            num_0 = len(self.points)
            if utils.check_interval_v2(step, *densify_interval, close='()'):
                self.densify(
                    optimizer,
                    max_grad=cfg['densify_grad_threshold'],
                    extent=self.cameras_extent,
                    densify_percent_dense=cfg['densify_percent_dense']
                )
            num_1 = len(self.points)
            if utils.check_interval_v2(step, *prune_interval, close='()'):
                if step > cfg['opacity_reset_interval'][0] and cfg['prune_max_screen_size'] > 0:
                    size_threshold = cfg['prune_max_screen_size']
                else:
                    size_threshold = None
                self.prune(
                    optimizer,
                    min_opacity=cfg['prune_opacity_threshold'],
                    extent=self.cameras_extent,
                    max_screen_size=size_threshold,
                    prune_percent_dense=cfg['prune_percent_dense'],
                )
            num_2 = len(self.points)
            if num_0 != num_1 or num_1 != num_2:
                logging.info(f'step[{step}], there are {num_2}(+{num_1 - num_0}, -{num_1 - num_2}) points.')
            if utils.check_interval_v2(step, *cfg['opacity_reset_interval'], close='()') or \
                (self.background_type == 'white' and step == densify_interval[1]):
                self.reset_opacity(optimizer)
                logging.info(f'reset_opacity at {step}')


def convert_offical_to_ours():
    import my_ext as ext
    import argparse
    from networks import options, make
    from pathlib import Path
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-i', '--load', type=str)
    parser.add_argument('-o', '--save', type=str)
    args = parser.parse_args()
    cfg_path = args.config
    assert os.path.exists(cfg_path)
    model_dir = Path(args.load).expanduser()
    if model_dir.is_dir():
        max_iter = -1
        for path in os.listdir(model_dir.joinpath('point_cloud')):
            results = re.match(r'iteration_(\d+)', path).groups()
            if len(results) == 1:
                max_iter = max(max_iter, int(results[0]))
        if max_iter < 0:
            raise FileNotFoundError(f'Can not find a valid dir in {model_dir}/point_cloud/iteration_*')
        parts = model_dir.joinpath('point_cloud', f'iteration_{max_iter}').parts
    else:
        parts = model_dir.parts[:-1]
    ply_path = Path(*parts[:-2], 'point_cloud', parts[-1], 'point_cloud.ply')

    assert ply_path.is_file(), f"can find '{ply_path}'"
    save_path = args.save

    parser = argparse.ArgumentParser()
    ext.config.options(parser)
    options(parser)
    cfg = ext.config.make([f'-c={cfg_path}'], True, True, parser)
    net = make(cfg)  # type: GaussianSplatting # noqa
    print(net)
    net.load_ply(ply_path)
    net._rotation.data = net._rotation.data[:, (1, 2, 3, 0)]

    if save_path:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path.joinpath('official.pth')
        save_path.parent.mkdir(exist_ok=True)
        torch.save(net.state_dict(), save_path)
        print('save converted path to', save_path)


def vis_trained():
    from my_ext.utils.gui.viewer_3D import simple_3d_viewer
    from my_ext import ops_3d

    path = '/home/wan/wan_code/NeRF/results/guassian_lego.ply'
    path = '/home/wan/wan_code/NeRF/results/Gaussian/lego/last.ply'
    model = GaussianSplatting(3)
    model.load_ply(path)

    @torch.no_grad()
    def rendering(Tw2v, fovy, size):
        Tw2v = Tw2v.cuda()
        Tw2v = ops_3d.convert_coord_system(Tw2v, 'opengl', 'colmap')
        Tv2c = ops_3d.perspective(size=size, fovy=fovy).cuda()
        # print(Tv2c)
        fovx = ops_3d.fovx_to_fovy(fovy, size[1] / size[0])
        Tv2c = ops_3d.opencv.perspective(fovy, size=size).cuda()
        # print(Tv2c_2)
        # exit()
        Tw2c = Tv2c @ Tw2v
        Tv2w = torch.inverse(Tw2v)
        tanfovx = math.tan(0.5 * fovx)
        tanfovy = math.tan(0.5 * fovy)
        bg_color = [1, 1, 1]  # if dataset.background == 'white' else [0, 0, 0]
        bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        raster_settings = model.RasterizationSettings(
            image_height=size[1],
            image_width=size[0],
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=1.0,
            viewmatrix=Tw2v.T,
            projmatrix=Tw2c.T,
            sh_degree=model.max_sh_degree,
            campos=Tv2w[:3, 3],
            prefiltered=False,
            debug=False,
            **(dict(bg=bg_color) if model.use_official_gaussians_render else {})
        )
        return model.gs_rasterizer(**model(), raster_settings=raster_settings)['images']

    simple_3d_viewer(rendering)


if __name__ == '__main__':
    convert_offical_to_ours()
    # vis_trained()
    # test_compute_cov3D()
    # test_compute_cov2D()
