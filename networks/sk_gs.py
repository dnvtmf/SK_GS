import logging
import math
import random
from typing import Any, Mapping, Optional, Union
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from pytorch3d.ops import knn_points
from lietorch import SE3, SO3

import my_ext
from my_ext.ops.point_sample import FurthestSampling
from my_ext import utils, ops_3d, SH2RGB
from my_ext.blocks import MLP_with_skips

from networks.encoders import POSITION_ENCODERS
from networks.gaussian_splatting import GaussianSplatting, NETWORKS, get_expon_lr_func, BasicPointCloud
from networks.losses.SC_GS_arap_loss import cal_connectivity_from_points, cal_arap_error


def get_superpoint_features(value: Tensor, neighbor: Tensor, G: Tensor, num_sp: int):
    """ value_sp[j] = 1 / w[j] sum_{i=0}^{N} [j in neighbor[i]] G[i, j] value[i]
    w[j] = sum_{i=0}^{N} [j in neighbor[i]] G[i, j]

    Args:
        G: shape [N, K]
        neighbor: shape: [N, K] The indices of K-nearest superpoints for each point
        value: [N, C]
        num_sp: The number of superpoints
    Returns:
        Tensor: the value for superpoints, shape: [num_sp, C]
    """
    C = value.shape[-1]
    assert 0 <= neighbor.min() and neighbor.max() < num_sp
    value_sp = value.new_zeros([num_sp, C])
    value_sp = torch.scatter_reduce(
        value_sp,
        dim=0,
        index=neighbor[:, :, None].repeat(1, 1, C).view(-1, C),
        src=(value[:, None, :] * G[:, :, None]).view(-1, C),
        reduce='sum'
    )
    w = value.new_zeros([num_sp]).scatter_reduce_(dim=0, index=neighbor.view(-1), src=G.view(-1), reduce='sum')
    return value_sp / w[:, None].clamp_min(1e-5)


def find_root(father: Tensor):
    """找到一个root，使得所有关节到root的节点数尽量的少"""
    M = father.shape[0]
    # build graph
    edges = {i: [] for i in range(M)}
    for i in range(M):
        if father[i] < 0:
            continue
        else:
            j = father[i].item()
            edges[i].append(j)
            edges[j].append(i)
    visited = np.zeros(M, dtype=np.int32)
    num_edges = np.array([len(edges[i]) for i in range(M)])
    # find leaf: only one points connect to it
    que = [i for i in range(M) if num_edges[i] == 1]
    for node in que:
        visited[node] = 1
    assert len(que) > 0
    i = 0
    while i < len(que):
        now = que[i]
        i += 1
        for node in edges[now]:
            if num_edges[node] > 1:
                num_edges[node] -= 1
                visited[node] = max(visited[node], visited[now] + 1)  # noqa
                if num_edges[node] == 1:
                    que.append(node)
    # find new father
    root = que[-1]
    max_depth = visited.max()
    max_level = 0
    while 2 ** max_level < max_depth:
        max_level += 1
    parents = father.new_full((M, max_level), root)
    depth = parents.new_zeros(M)
    que = [root]
    visited[:] = 0
    visited[root] = 1
    i = 0
    while i < len(que):
        now = que[i]
        i += 1
        for node in edges[now]:
            if visited[node] == 0:
                parents[node, 0] = now
                depth[node] = depth[now] + 1
                que.append(node)
                visited[node] = 1
    for i in range(1, max_level):
        for j in range(M):
            parents[j, i] = parents[parents[j, i - 1], i - 1]
    return parents, depth, root


@torch.no_grad()
@my_ext.try_use_C_extension
def joint_discovery(joint_cost: Tensor):
    M = joint_cost.shape[0]
    connectivity = torch.eye(M, device=joint_cost.device, dtype=torch.long)  # (n_parts, n_parts)
    joint_connection = torch.full((M,), -1, device=joint_cost.device, dtype=torch.long)
    for j in range(M - 1):  # there are n_parts-1 connection
        invalid_connection_bias = connectivity * 1e10
        connected = torch.argmin(joint_cost + invalid_connection_bias)
        idx_0 = connected // M  # torch.div(connected, M, rounding_mode='trunc')
        idx_1 = connected % M
        # update connectivity
        connectivity[idx_0] = torch.maximum(connectivity[idx_0].clone(), connectivity[idx_1].clone())
        connectivity[torch.where(connectivity[idx_0] == 1)] = connectivity[idx_0].clone()
        if joint_connection[idx_0] == -1:
            joint_connection[idx_0] = idx_1
        else:
            parents = [idx_1]
            a = joint_connection[idx_1]
            while a != -1:
                parents.append(a)
                a = joint_connection[a]
            for i in range(len(parents) - 1, 0, -1):
                joint_connection[parents[i]] = parents[i - 1]
            joint_connection[idx_1] = idx_0
    return find_root(joint_connection)


class SimpleDeformationNetwork(nn.Module):
    def __init__(
        self,
        p_in_channels=3,
        t_in_channels=1,
        out_channels=6,
        width=256,
        depth=8,
        skips=(4,),
        pos_enc_p='freq',
        pos_enc_p_cfg: dict = None,
        pos_enc_t='freq',
        pos_enc_t_cfg: dict = None
    ):
        super().__init__()
        self.pos_enc_p = POSITION_ENCODERS[pos_enc_p](**utils.merge_dict(pos_enc_p_cfg, input_dim=p_in_channels))
        self.pos_enc_t = POSITION_ENCODERS[pos_enc_t](**utils.merge_dict(pos_enc_t_cfg, input_dim=t_in_channels))
        self.dynamic_net = MLP_with_skips(
            in_channels=self.pos_enc_p.output_dim + self.pos_enc_t.output_dim,
            dim_hidden=width,
            out_channels=out_channels,
            num_layers=depth,
            skips=skips,
        )

    def forward(self, points: Tensor, t: Tensor):
        p_embed = self.pos_enc_p(points)
        t_embed = self.pos_enc_t(t.view(-1, 1)).expand(*points.shape[:-1], -1)
        x_input = torch.cat([p_embed, t_embed], dim=-1)
        out = self.dynamic_net(x_input)
        return out


def skeleton_warp_v0(local_T: Tensor, global_T: Optional[Tensor], parents: Tensor, root: Tensor):
    if parents.ndim == 2:
        parents = parents[:, 0]
    N = local_T.shape[0]
    out = torch.eye(4, device=local_T.device, dtype=local_T.dtype).expand(N, 4, 4).contiguous()
    for i in range(N):
        f = i
        while f != root:
            out[i] = local_T[f] @ out[i]
            f = parents[f]
    if global_T is None:
        return out
    return (global_T[root] if global_T.ndim == 3 else global_T) @ out


def skeleton_warp(local_T: Tensor, global_T: Optional[Tensor], parents: Tensor, root: Tensor):
    N, L = parents.shape  # num_parts, num_level
    out = local_T.clone()
    out[root] = torch.eye(4, device=local_T.device, dtype=local_T.dtype)
    for level in range(L):
        out = out[parents[:, level]] @ out
    if global_T is None:
        return out
    return (global_T[root] if global_T.ndim == 3 else global_T) @ out


def skeleton_warp_SE3(local_T: SE3, global_T: Optional[SE3], parents: Tensor, root: Tensor) -> SE3:
    N, L = parents.shape  # num_parts, num_level
    out = local_T.vec().clone()
    out[root] = out.new_tensor([0, 0, 0, 0, 0, 0, 1.])
    out = SE3.InitFromVec(out)
    for level in range(L):
        out = out[parents[:, level]] * out
    if global_T is None:
        return out
    elif len(global_T.shape) > 0:
        global_T = global_T[root]
    if len(global_T.shape) == 0:
        global_T = global_T[None]
    return global_T * out


class DeformNetwork(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=59,
        pos_enc_p='freq',
        pos_enc_p_cfg: dict = None,
        pos_enc_t='freq',
        pos_enc_t_cfg: dict = None,
        is_blender=False,
        sep_rot=False,
        resnet_color=True,
        color_wrt_dir=False,
        max_d_scale=-1,
        **kwargs
    ):  # t_multires 6 for D-NeRF; 10 for HyperNeRF
        super(DeformNetwork, self).__init__()
        self.name = 'mlp'
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.pos_enc_p = POSITION_ENCODERS[pos_enc_p](**utils.merge_dict(pos_enc_p_cfg, input_dim=3))
        self.pos_enc_t = POSITION_ENCODERS[pos_enc_t](**utils.merge_dict(pos_enc_t_cfg, input_dim=1))
        self.input_ch = self.pos_enc_p.output_dim + self.pos_enc_t.output_dim

        self.resnet_color = resnet_color
        self.color_wrt_dir = color_wrt_dir
        self.max_d_scale = max_d_scale

        self.reg_loss = 0.

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(self.pos_enc_t.output_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out)
            )

            self.linear = nn.ModuleList([nn.Linear(self.pos_enc_p.output_dim + self.time_out, W)])
            for i in range(D - 1):
                if i not in self.skips:
                    self.linear.append(nn.Linear(W, W))
                else:
                    self.linear.append(nn.Linear(W + self.pos_enc_p.output_dim + self.time_out, W))

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_scaling = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)

        self.sep_rot = sep_rot
        if self.sep_rot:
            self.local_rotation = nn.Linear(W, 4)
        self.reset_parameters()

    def reset_parameters(self):
        if self.sep_rot:
            nn.init.normal_(self.local_rotation.weight, mean=0, std=1e-4)
            nn.init.zeros_(self.local_rotation.bias)

        for layer in self.linear:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
        nn.init.normal_(self.gaussian_scaling.weight, mean=0, std=1e-8)
        nn.init.normal_(self.gaussian_rotation.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.zeros_(self.gaussian_scaling.bias)
        nn.init.zeros_(self.gaussian_rotation.bias)

    def forward(self, x: Tensor, t: Tensor, **kwargs):
        t_emb = self.pos_enc_t(t.view(-1, 1)).expand(x.shape[0], self.pos_enc_t.output_dim)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.pos_enc_p(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        if self.max_d_scale > 0:
            scaling = torch.tanh(scaling) * np.log(self.max_d_scale)
        return_dict = {'d_xyz': d_xyz, 'd_rotation': rotation, 'd_scaling': scaling, 'hidden': h}
        if self.sep_rot:
            return_dict['g_rotation'] = self.local_rotation(h)
        return return_dict


@NETWORKS.register('SK_GS')
class SkeletonGaussianSplatting(GaussianSplatting):
    train_db_times: Tensor
    gs_knn_index: Tensor
    gs_knn_dist: Tensor
    # superpoints
    sp_is_init: Tensor
    p2sp: Optional[Tensor]
    sp_cache: Tensor
    """cache the results from sp_deform_net, shape: [T, M, 14]"""
    sp_weights: Tensor
    sp_knn: Tensor
    ## Skeleton
    joint_is_init: Tensor
    sk_is_init: Tensor
    joint_cost: Tensor
    joint_parents: Tensor
    joint_depth: Tensor
    joint_root: Tensor
    sk_cache: Tensor
    """cache the results from sk_deform_net, shape: [T, M, 14] """

    def __init__(
        self,
        train_schedule: dict = None,
        is_blender=True,
        num_knn=3,
        gs_knn_num=20,
        gs_knn_update_interval=(1000, 3000),
        canonical_time_id=-1,
        use_canonical_net=False,
        canonical_replace_steps=(),
        # sp stage
        num_superpoints=512,
        hyper_dim=2,
        sep_rot=True,  # separate rotation
        net_cfg=None,
        sp_prune_threshold=1e-3,
        sp_split_threshold=0.0002,
        sp_merge_threshold=0.01,
        lr_deform_scale=1.0,
        lr_feature_scale=2.5,
        lr_deform_max_steps=40000,
        warp_method='LBS',
        LBS_method='weighted_kernel',
        # init stage
        node_max_num_ratio_during_init=16,
        init_num_times=16,
        init_sampling_step=7500,
        joint_init_steps=0,
        # sp->sk
        sk_re_init_gs=False,
        sk_densify_gs=False,
        sp_guided_detach=True,
        joint_update_interval=(1000, 10_000),
        sk_momentum=0.9,
        sk_knn_num=3,
        guided_step_start=-1,
        lr_joints=0.1,
        # sk stage
        sk_feature_dim=0,
        sk_use_features=True,
        sk_deform_net_cfg: dict = None,
        # other
        max_d_scale=-1.0,
        f_s=0.1,
        annealing_steps=20_000,
        adaptive_control_cfg=None,
        test_time_interpolate=False,
        loss_arap_start_step=0,
        which_rotation='quaternion',
        **kwargs
    ):
        adaptive_control_cfg = utils.merge_dict(
            adaptive_control_cfg,
            # node_densify_interval=[5000, 5000, 25_000],
            # node_force_densify_prune_step=10_000,
            # node_enable_densify_prune=False,
            sp_adjust_interval=[5000, 5000, 25000],
            sp_merge_interval=[-1, 10_000, 20_000],
            init_opacity_reset_interval=[3000, 0, -1],
            init_densify_prune_interval=[100, 0, -1],
        )
        super().__init__(**kwargs, adaptive_control_cfg=adaptive_control_cfg)
        # train schedule
        self.stages = {}
        self.train_schedule = []
        step = 0
        for stage in ['static', 'init_fix', 'init', 'sp_fix', 'sp', 'sk_init', 'sk_fix', 'sk']:
            steps = 0 if train_schedule is None else train_schedule.get(stage, 0)
            self.stages[stage] = (step, step + steps, steps)  # start, end, len
            self.train_schedule.append((step, stage))
            step += steps
        self._R_dim = {'lie': 3, 'quaternion': 4}[which_rotation]
        if which_rotation == 'lie':
            self.to_SO3 = SO3.exp
        else:
            self.to_SO3 = SO3.InitFromVec
        self.num_knn = num_knn
        self.num_frames = 0
        self.canonical_time_id = canonical_time_id
        self.canonical_replace_steps = canonical_replace_steps
        self.test_time_interpolate = test_time_interpolate
        self.hyper_dim = hyper_dim

        self.register_buffer('train_db_times', torch.tensor(self.num_frames))
        if self.hyper_dim > 0:
            self.hyper_feature = nn.Parameter(torch.empty(0, self.hyper_dim))
            self.param_names_map['hyper_feature'] = 'hyper'
        else:
            self.hyper_feature = None

        self.gs_knn_update_interval = gs_knn_update_interval
        self.gs_knn_num = gs_knn_num
        self.register_buffer('gs_knn_index', torch.empty(0, self.gs_knn_num, dtype=torch.long), persistent=False)
        self.register_buffer('gs_knn_dist', torch.empty(0, self.gs_knn_num, dtype=torch.float), persistent=False)
        self._is_gs_knn_updated = False

        # init stage
        self.init_sampling_step = init_sampling_step
        self.init_num_times = init_num_times
        if use_canonical_net and self.canonical_time_id >= 0:
            self.canonical_net = DeformNetwork(
                **utils.merge_dict(net_cfg, sep_rot=sep_rot, is_blender=is_blender, max_d_scale=max_d_scale)
            )  # same to sk_deform_net
        else:
            self.canonical_net = None
        # sp stage
        self.f_s = f_s
        self.annealing_steps = annealing_steps

        self.sep_rot = sep_rot

        self.sp_prune_threshold = sp_prune_threshold
        self.sp_split_threshold = sp_split_threshold
        self.sp_merge_threshold = sp_merge_threshold

        self.sp_points = nn.Parameter(torch.randn(num_superpoints, 3))
        if self.hyper_dim > 0:
            self.sp_hyper_feature = nn.Parameter(torch.zeros(num_superpoints, self.hyper_dim))
        else:
            self.sp_hyper_feature = None

        assert LBS_method in ['W', 'dist', 'kernel', 'weighted_kernel']
        self.LBS_method = LBS_method
        self._sp_radius: Optional[Tensor] = None
        self._sp_weight: Optional[Tensor] = None
        self.sp_W: Optional[Tensor] = None
        if self.LBS_method == 'W':
            self.sp_W = nn.Parameter(torch.zeros(0, self.num_superpoints))
            self.param_names_map['sp_W'] = 'sp_W'
        if self.LBS_method == 'weighted_kernel' or self.LBS_method == 'kernel':
            self._sp_radius = nn.Parameter(torch.randn(num_superpoints))
        if self.LBS_method == 'weighted_kernel':
            self._sp_weight = nn.Parameter(torch.zeros(num_superpoints))
        self.sp_deform_net = DeformNetwork(
            **utils.merge_dict(net_cfg, sep_rot=sep_rot, is_blender=is_blender, max_d_scale=max_d_scale))

        assert warp_method in ['largest', 'LBS', 'LBS_c']
        self.warp_method = warp_method
        if self.warp_method == 'largest':
            self.register_buffer('p2sp', torch.empty(0, dtype=torch.int32))
        else:
            self.p2sp = None

        self.register_buffer('sp_is_init', torch.tensor(False))
        self.register_buffer('sp_weights', torch.empty(0, self.num_knn))  # [N, K]
        self.register_buffer('sp_knn', torch.empty(0, self.num_knn, dtype=torch.long))  # [N, K]
        self.register_buffer('sp_cache', torch.empty(self.num_frames, self.num_superpoints, 14 if self.sep_rot else 10))
        self.node_max_num_ratio_during_init = node_max_num_ratio_during_init

        # Cached nn_weight to speed up
        self.cached_nn_weight = False
        self.nn_weight, self.nn_dist, self.nn_idxs = None, None, None

        ## superpoints -> skeleton
        self.sk_re_init_gs = sk_re_init_gs  # re-initialize Gaussians in skeleton-stage?
        self.sk_densify_gs = sk_densify_gs
        self.sp_guided_detach = sp_guided_detach
        self.joint_update_interval = joint_update_interval
        self.joint_init_steps = joint_init_steps
        self.sk_momentum = sk_momentum
        self.sk_knn_num = sk_knn_num
        self.guided_step_start = guided_step_start

        M = self.num_superpoints
        self.joints = nn.Parameter(torch.empty(M, 3))
        self.joint_pos = nn.Parameter(torch.empty(M, M, 3))
        self.register_buffer('joint_is_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('joint_cost', torch.zeros(M, M))
        self._joint_pair = None
        ## skeleton stage
        self.register_buffer('sk_is_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('sk_W_is_init', torch.tensor(False, dtype=torch.bool))
        self.sk_deform_net_cfg = sk_deform_net_cfg
        self.sk_use_features = sk_use_features
        self.sk_feature_dim = sk_feature_dim
        self.sk_feature = nn.Parameter(torch.randn(M, sk_feature_dim)) if sk_feature_dim > 0 else None
        self.sk_dims = [self._R_dim, 4, 3]
        self.sk_deform_net = SimpleDeformationNetwork(
            **utils.merge_dict(sk_deform_net_cfg, out_channels=self.sk_dims, p_in_channels=3 + self.sk_feature_dim)
        )
        self.register_buffer('joint_parents', torch.full((M, 1), -1, dtype=torch.int32))
        self.register_buffer('joint_depth', torch.zeros(M, dtype=torch.int32))
        self.register_buffer('joint_root', torch.arange(M, dtype=torch.int32))
        self.register_buffer('sk_cache', torch.empty((self.num_frames, M, sum(self.sk_dims))))
        self.global_tr = nn.Parameter(torch.zeros(self.num_frames, 7))
        self.reset_parameters()
        # other
        self.lr_feature_scale = lr_feature_scale
        self.lr_deform_scale = lr_deform_scale
        self.lr_deform_max_steps = lr_deform_max_steps
        self.lr_joints = lr_joints
        self.lr_scheduler = {}
        self.scales_all_same = True
        self.time_interval = 0.05
        self.loss_arap_start_step = loss_arap_start_step

    def reset_parameters(self):
        for m in self.sk_deform_net.dynamic_net.last:
            nn.init.zeros_(m.bias)
            nn.init.normal_(m.weight, mean=0, std=1e-6)

    @property
    def kernel_radius(self):
        return torch.exp(self._sp_radius)

    @property
    def kernel_weight(self):
        return torch.sigmoid(self._sp_weight)

    @property
    def num_superpoints(self):
        return self.sp_points.shape[0]

    @property
    def get_scaling(self):
        scales = self._scaling
        if 0 < self._step < self.stages['sp_fix'][0]:
            scales = scales.mean(dim=(0, 1) if self.scales_all_same else 1, keepdim=True).expand_as(scales)
        return self.scaling_activation(scales)

    @property
    def device(self):
        return self._xyz.device

    def set_from_dataset(self, dataset):
        super().set_from_dataset(dataset)
        self.num_frames = dataset.num_frames  # the number of frames
        M = self.num_superpoints
        self.train_db_times = dataset.times[dataset.camera_ids == dataset.camera_ids[0]]
        assert self.num_frames == len(self.train_db_times)
        self.global_tr = nn.Parameter(torch.zeros(self.num_frames, 7))
        assert self.canonical_time_id < self.num_frames
        self.sp_cache = torch.zeros(self.num_frames, M, 14 if self.sep_rot else 10)
        self.sk_cache = torch.zeros(self.num_frames, M, sum(self.sk_dims))
        self.time_interval = 1. / dataset.num_frames

    def get_params(self, cfg):
        self.lr_spatial_scale = 5.
        params_groups = super().get_params(cfg)
        lr = self.lr_deform_scale * cfg.lr * self.lr_spatial_scale * self.lr_position_init
        if self.canonical_net is not None:
            params_groups.append({'params': list(self.canonical_net.parameters()), 'lr': lr, 'name': 'canonical'})
        params_groups.append({'params': list(self.sp_deform_net.parameters()), 'lr': lr, 'name': 'sp_deform'})
        params_groups.append({'params': [self.sp_points], 'lr': lr, 'name': 'sp_points'})
        if self._sp_radius is not None:
            params_groups.append({'params': [self._sp_radius], 'lr': lr, 'name': 'sp_radius'})
        if self._sp_weight is not None:
            params_groups.append({'params': [self._sp_weight], 'lr': lr, 'name': 'sp_weight'})
        if self.sp_W is not None:
            params_groups.append({'params': [self.sp_W], 'lr': lr, 'name': 'sp_W'})
        if self.hyper_dim > 0:
            lr_f = cfg.lr * self.lr_feature_scale
            params_groups.extend([
                {'params': [self.hyper_feature], 'lr': lr_f, 'name': 'hyper', 'fix': True},
                {'params': [self.sp_hyper_feature], 'lr': lr_f, 'name': 'sp_hyper', 'fix': True}
            ])

        params_groups.extend([
            {'params': list(self.sk_deform_net.parameters()), 'lr': lr, 'name': 'sk_deform'},
            {'params': [self.joint_pos], 'lr': lr, 'name': 'joint_pos'},
            {'params': [self.global_tr], 'lr': lr * 0, 'name': 'global_tr'},
            {'params': [self.joints], 'lr': lr * self.lr_joints, 'name': 'joints'},
        ])
        if self.sk_feature is not None:
            params_groups.append({'params': [self.sk_feature], 'lr': lr, 'name': 'sk_feature'})
        self.lr_scheduler['deform'] = get_expon_lr_func(
            lr, cfg.lr * self.lr_position_final * self.lr_deform_scale, 0,
            self.lr_position_delay_mult, self.lr_deform_max_steps
        )
        assert sum(len(g['params']) for g in params_groups) == len(list(self.parameters()))
        return params_groups

    def update_learning_rate(self, optimizer=None, *args, **kwargs):
        if optimizer is None:
            optimizer = self._task.optimizer
        if self._step <= self.stages['sp_fix'][0]:
            step = self._step
        elif self._step <= self.stages['sp'][1]:
            step = (self._step - self.stages['sp_fix'][0])
        else:
            step = (self._step - self.stages['sk_init'][0])
        for group in optimizer.param_groups:
            if group.get('name', None) in ['sp_deform', 'sk_deform', 'canonical']:
                group['lr'] = self.lr_scheduler['deform'](step)
            elif group.get('name', None) == 'xyz':
                group['lr'] = self.lr_scheduler['xyz'](step)
            # elif group.get('name', None) in [ 'joints', 'global_tr' ]:
            #     group['lr'] = 0
        return

    def create_from_pcd(self, pcd: BasicPointCloud, lr_spatial_scale: float = None):
        points = self.init_superpoints(True, False, torch.from_numpy(pcd.points.astype(np.float32)), True)
        points = points.detach().cpu().numpy()
        new_pcd = BasicPointCloud(points, SH2RGB(np.zeros_like(points)), np.zeros_like(points))
        super().create_from_pcd(new_pcd, lr_spatial_scale)
        N = len(self._xyz)
        self.hyper_feature = nn.Parameter(torch.full([N, self.hyper_dim], -1e-2, device=self.device))
        if self.sp_W is not None:
            self.sp_W = nn.Parameter(torch.ones([N, self.num_superpoints]))
        if self.p2sp is not None:
            self.p2sp = self.p2sp.new_zeros(N)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs):
        N = state_dict['_xyz'].shape[0]
        M = state_dict['sp_points'].shape[0]
        self.sp_points = nn.Parameter(torch.randn(M, 3))
        self.joints = nn.Parameter(torch.randn(M, 3))
        if self._sp_radius is not None:
            self._sp_radius = nn.Parameter(torch.randn(M))
        if self._sp_weight is not None:
            self._sp_weight = nn.Parameter(torch.zeros(M))
        if self.sp_W is not None:
            self.sp_W = nn.Parameter(torch.empty((N, M)))
        # if self.p2sp is not None:
        #     self.p2sp = self.p2sp.new_zeros(N)
        for key in ['joint_parents', 'joint_root', 'global_tr', 'joint_pos', 'joint_cost', 'joint_depth', 'sk_cache',
                    'sp_cache', 'sp_weights', 'sp_knn', 'p2sp', 'sp_hyper_feature', 'train_db_times']:
            if key in state_dict:
                if isinstance(getattr(self, key), nn.Parameter):
                    setattr(self, key, nn.Parameter(state_dict[key]))
                else:
                    setattr(self, key, state_dict[key])
        super().load_state_dict(state_dict, strict, **kwargs)

    @torch.no_grad()
    def init_superpoints(self, force=False, use_hyper=True, points=None, return_sp=False):
        if not self.training:
            return
        if self.sp_is_init and not force:
            return
        times = torch.linspace(0, 1, self.init_num_times, device=self.device)
        points = self.points if points is None else points
        if use_hyper:
            trans_samp = [self.sp_deform_net(points, times[i])['d_xyz'] for i in range(self.init_num_times)]
            trans_samp = torch.stack(trans_samp, dim=1)
            hyper_pcl = (trans_samp + points[:, None]).reshape(points.shape[0], -1)
        else:
            hyper_pcl = None
        if return_sp or self.canonical_net is None:
            init_pcl = points
        else:
            init_pcl = self.sp_deform_net(points, self.train_db_times[self.canonical_time_id])['d_xyz'] + points
            self.sp_deform_net.load_state_dict(self.canonical_net.state_dict())
            logging.info('[red]Apply canonical_net')
        # Initialize Superpoints
        pcl_to_samp = init_pcl if hyper_pcl is None else hyper_pcl
        init_nodes_idx = FurthestSampling(pcl_to_samp.detach()[None].cuda(), self.num_superpoints)[0].to(self.device)
        self.sp_points.data = init_pcl[init_nodes_idx]
        nn.init.constant_(self.sp_hyper_feature, 1e-2)
        scene_range = init_pcl.max() - init_pcl.min()
        if self._sp_radius is not None:
            self._sp_radius.data = torch.log(.1 * scene_range + 1e-7) * scene_range.new_ones([self.num_superpoints])
        if self._sp_weight is not None:
            self._sp_weight.data = torch.zeros_like(self._sp_radius)
        if return_sp:
            return self.sp_points[..., :3]
        opt = self._task.optimizer
        optimizable_tensors = self.change_optimizer(
            opt, {opt_name: getattr(self, name)[init_nodes_idx] for name, opt_name in self.param_names_map.items()},
            op='replace'
        )
        for param_name, opt_name in self.param_names_map.items():
            if opt_name in optimizable_tensors:
                setattr(self, param_name, optimizable_tensors[opt_name])
        opt.zero_grad(set_to_none=True)
        self._xyz.data.copy_(self.sp_points[..., :3])
        self.active_sh_degree = 0
        self.xyz_gradient_accum = self.xyz_gradient_accum.new_zeros((len(self.points), 1))
        self.denom = self.denom.new_zeros((len(self.points), 1))
        self.max_radii2D.data = self.max_radii2D.new_zeros((self.points.shape[0]))

        self.sp_is_init = self.sp_is_init.new_tensor(True)
        logging.info(f'[red]Control node initialized with {self.sp_points.shape[0]} from {init_pcl.shape[0]} points.')
        return init_nodes_idx

    def get_smooth_scale(self, lr_final=1e-15, lr_delay_steps=0.01, lr_delay_mult=1.0):
        lr_init = self.f_s
        max_steps = self.annealing_steps
        step = self._step if self._step <= self.stages['sp_fix'][0] else self._step - self.stages['sp_fix'][0]
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
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    def init_stage(self, x, t, use_canonical_net=False):
        if not self.sp_deform_net.is_blender:
            t_noise = torch.randn_like(t) * self.time_interval * self.get_smooth_scale()
            t = t + t_noise
        if use_canonical_net:
            d_xyz = self.canonical_net(x.detach(), t)['d_xyz']
        else:
            d_xyz = self.sp_deform_net(x.detach(), t)['d_xyz']
        return d_xyz, d_xyz.new_tensor(0), d_xyz.new_tensor(0)

    def calc_LBS_weight(self, points: Tensor, sp_points: Tensor, feature=None, sp_feature=None, K=None, temperature=1.):
        ## calculate knn
        if feature is not None and sp_feature is not None:
            points = torch.cat([points.detach(), feature], dim=-1)
            sp_points = torch.cat([sp_points.detach(), sp_feature], dim=-1)
        K = self.num_knn if K is None else K
        nn_dist, indices, _ = knn_points(points[None], sp_points[None], None, None, K=K)  # N, K
        nn_dist, indices = nn_dist[0], indices[0]  # N, K
        ## calculate weights
        if self._sp_radius is not None:
            radius = self.kernel_radius[indices]  # N, K
            weights = torch.exp(-nn_dist / (2 * radius ** 2))  # N, K
            if self._sp_weight is not None:
                weights = weights * self.kernel_weight[indices]
            weights = weights + 1e-7
            weights = weights / weights.sum(dim=-1, keepdim=True)  # N, K
        elif self.sp_W is not None:
            weights = torch.gather(self.sp_W, dim=1, index=indices).softmax(dim=-1)
        else:  # LBS_method = dist
            weights = torch.softmax(-nn_dist / temperature, dim=-1)
        if not self.sk_is_init:
            self.sp_weights = weights.detach()
            self.sp_knn = indices.detach()
        return weights, indices

    def warp(self, points, sp_points, sp_t, sp_r, sp_rot, sp_scale, weights, indices, method='LBS'):
        """

        Args:
            points (Tensor): the position of Gaussians, shape: [P, 3]
            sp_points (Tensor): the postion of  superpoints, shape: [M, 3]
            sp_t (Tensor | SE3): the translation or transform maxtrix predicted by superpoints
            sp_r (Tensor | None): the rotation predicted by superpoints
            sp_rot (Tensor | None): the residual direction of Gaussians predicted by superpoints
            sp_scale (Tensor | None): the residual scale of Gaussians predicted by superpoints
            weights (Tensor): LBS weights, shape: [P, K]
            indices (Tensor): LBS indices, ie the index of KNN, shape: [P, K]
            method (str): LBS_c, LBS, largest

        Returns:
            (Tensor, Optional[Tensor], Optional[Tensor], Tensor)

            - d_points: shape: [P, 3]
            - d_rotation: shape: [P, 4]
            - d_scale: shape: [P, 3]
            - sp_t: The transformation matrix of superpoints, represented by quaternion, shape: [M, 4]
        """
        if isinstance(sp_t, SE3):
            sp_r = sp_t.vec()[..., 3:]
            spT = sp_t
        else:
            assert sp_r is not None
            if method == 'LBS_c':
                sp_t = sp_t + sp_points + SO3.InitFromVec(sp_r).act(-sp_points)
            spT = SE3.InitFromVec(torch.cat([sp_t, sp_r], dim=-1))

        if isinstance(spT, Tensor):
            if method == 'LBS_c' or method == 'LBS':
                d_points = (ops_3d.apply(points[:, None], spT[indices]) * weights[..., None]).sum(dim=1) - points
            else:
                d_points = ops_3d.apply(points, spT[self.p2sp]) - points
        else:  # SE3
            if method == 'LBS_c' or method == 'LBS':
                d_points = (spT[indices].act(points[:, None]) * weights[..., None]).sum(dim=1) - points
            else:
                d_points = spT[self.p2sp].act(points) - points
        # else:
        #     d_points = (sp_t[indices] * weights[..., None]).sum(dim=1)
        if sp_rot is not None:
            d_rotation = (sp_rot[indices] * weights[..., None]).sum(dim=1)
        elif sp_r is not None:
            d_rotation = (sp_r[indices] * weights[..., None]).sum(dim=1)
        else:
            d_rotation = None
        d_scales = (sp_scale[indices] * weights[..., None]).sum(dim=1) if sp_scale is not None else None
        if isinstance(spT, SE3):
            spT = spT.vec()
        return d_points, d_rotation, d_scales, spT

    def sp_stage(
        self, points: Tensor, t, time_id: int = None, use_canonical_net=False, sp_points: Tensor = None, **kwargs
    ):
        points = points.detach()
        sp_points = self.sp_points if sp_points is None else sp_points
        rot_bias = points.new_tensor([0, 0, 0, 1.])
        # Calculate nn weights: [N, K]
        if not self.sp_deform_net.is_blender:
            t_noise = torch.randn_like(t) * self.time_interval * self.get_smooth_scale()
            t = t + t_noise
        if use_canonical_net:
            outs = self.canonical_net(sp_points.detach(), t)
            weights, indices = self.sp_weights, self.sp_knn
        else:
            weights, indices = self.calc_LBS_weight(points, sp_points, self.hyper_feature, self.sp_hyper_feature)
            outs = self.sp_deform_net(sp_points.detach(), t)
        d_xyz, d_scale = outs['d_xyz'], outs['d_scaling']
        d_rot = ops_3d.quaternion.normalize(outs['d_rotation'] + rot_bias)
        g_rot = ops_3d.quaternion.normalize(outs['g_rotation'] + rot_bias) if self.sep_rot else None

        if self.warp_method == 'largest' and self.training:
            self.p2sp = torch.gather(indices, -1, weights.argmax(dim=-1, keepdim=True))[:, 0]
        d_points, d_rotation, d_scales, spT = self.warp(
            points, sp_points, d_xyz, d_rot, g_rot, d_scale,
            weights, indices, self.warp_method
        )
        return d_points, d_rotation, d_scales, spT, g_rot, d_scale, weights, indices

    @torch.no_grad()
    def init_joint_pos(self, force=False):
        if self.joint_is_init and not force:
            return
        self.joint_is_init = self.joint_is_init.new_tensor(True)
        logging.info('init joint_pos')
        sp_points = self.sp_points[..., :3]
        self.joint_pos.data.copy_((sp_points[:, None] + sp_points[None, :]) * 0.5)

    @torch.no_grad()
    def init_joint_pos_v2(self, knn, knn_weights, force=False, set_cost=False):
        """对(a, b), 从b对应的高斯中选取所有时间中距离最小的一个"""
        if self.joint_is_init and not force:
            return
        logging.info('begin init joint_pos')
        points = self.points
        p2sp = torch.gather(knn, -1, knn_weights.argmax(dim=-1, keepdim=True))[:, 0]
        T = SE3.InitFromVec(self.sp_cache[..., :7])
        for a in range(self.num_superpoints):
            points_a = T[:, a:a + 1].act(points[None, :])  # [T, N, 3]
            points_b = T[:, p2sp].act(points[None, :])  # [T, N, 3]
            dist = torch.pairwise_distance(points_a, points_b)  # [T, N]
            dist = dist.mean(dim=0)  # [N]
            for b in range(self.num_superpoints):
                mask = p2sp.eq(b)
                index = torch.nonzero(mask).squeeze(-1)
                if index.numel() > 0:
                    value, idx = dist[mask].min(dim=0)
                    if set_cost:
                        self.joint_cost[a, b] = value
                    self.joint_pos[a, b] = points[index[idx]]
        self.joint_is_init = self.joint_is_init.new_tensor(True)
        logging.info('init joint_pos')

    def init_joint(self, progress_func=None):
        if self.joint_is_init:
            return
        self.joint_is_init = self.joint_is_init.new_tensor(True)
        logging.info('Begin init joint')
        self.init_joint_pos(force=True)
        # self.init_joint_pos_v2(force=True)

        sp_R = SO3.InitFromVec(self.sp_cache[..., 3:7])  # [T, M]
        joint_rot = sp_R.inv()[:, None] * sp_R[:, :, None]  # [T, M, M], R_b^-1 R_a
        sp_delta_tr = self.sp_cache[..., :7]
        optimizer = torch.optim.Adam([self.joint_pos], lr=1.0e-3)
        loss_meter = my_ext.DictMeter(float2str=utils.float2str)

        with torch.enable_grad():
            for i in range(self.joint_init_steps):
                tid = random.randrange(self.num_frames)
                joint_loss, all_loss = self.find_joint_loss(sp_delta_tr[tid], joint_rot, tid, True)
                loss_dict = {'best': joint_loss, 'all': all_loss}

                tatol_loss = sum(loss_dict.values())
                loss_dict['total'] = tatol_loss
                loss_meter.update(loss_dict)
                tatol_loss.backward()  # noqa
                optimizer.step()
                optimizer.zero_grad(True)
                if (i + 1) % 100 == 0:
                    logging.info(f"step [{i + 1}]: loss: {loss_meter.average}")
                    loss_meter.reset()
                if progress_func is not None:
                    progress_func()
        logging.info('Init joint')
        return

    @torch.enable_grad()
    def init_sk_deform(self, progress_func=None):
        param_groups = list(self.sk_deform_net.parameters()) + [self.joints, self.global_tr]
        if self._sp_radius is not None:
            param_groups.append(self._sp_radius)
        if self._sp_weight is not None:
            param_groups.append(self._sp_weight)
        if self.sp_W is not None:
            param_groups.append(self.sp_W)
        optimizer = torch.optim.Adam([
            {'params': param_groups, 'lr': 1e-3},
            # {'params': [self.joints], 'lr': 1e-4},
        ], lr=1e-3)
        state = self.training
        self.train()
        points_c = self.points.detach()
        loss_meter = my_ext.DictMeter(float2str=my_ext.utils.str_utils.float2str)
        for step in range(self.joint_init_steps):
            tid = random.randrange(self.num_frames)
            t = self.train_db_times[tid]
            with torch.no_grad():
                if self.sep_rot:
                    sp_tr, sp_d_rot, sp_d_scale = self.sp_cache[tid].split([7, 4, 3], dim=-1)
                else:
                    sp_tr, sp_d_scale = self.sp_cache[tid].split([7, 3], dim=-1)
                    sp_d_rot = sp_tr[:, 3:]
                sp_tr = SE3.InitFromVec(sp_tr)
                points_t1 = points_c + self.warp(
                    points_c, self.sp_points,
                    sp_tr, None, None, None,
                    self.sp_weights, self.sp_knn, self.warp_method
                )[0]
            d_xyz, _, _, sk_tr, sk_d_rot, sk_d_scale, _, _, _ = self.sk_stage(points_c, t, tid)
            sk_tr = SE3.InitFromVec(sk_tr)
            points_t2 = points_c + d_xyz
            cmp_t = (sp_tr.inv() * sk_tr).log().norm(dim=-1).mean()
            cmp_t = self.loss_funcs('cmp_t', cmp_t)
            cmp_p = self.loss_funcs('cmp_p', F.mse_loss, points_t1, points_t2)
            cmp_r = self.loss_funcs('cmp_r', F.mse_loss, sk_d_rot, sp_d_rot)
            cmp_s = self.loss_funcs('cmp_s', F.mse_loss, sk_d_scale, sp_d_scale)
            loss_meter.update({'cmp_t': cmp_t, 'cmp_p': cmp_p, 'cmp_r': cmp_r, 'cmp_s': cmp_s})
            loss = cmp_t + cmp_r + cmp_s + cmp_p
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(True)
            if step % 100 == 0:
                logging.info(f"step[{step}]: loss: {loss_meter.average}")
                loss_meter.reset()
            if progress_func is not None:
                progress_func()
        with torch.no_grad():
            loss_meter.reset()
            for tid in range(self.num_frames):
                t = self.train_db_times[tid]
                if self.sep_rot:
                    sp_tr, sp_d_rot, sp_d_scale = self.sp_cache[tid].split([7, 4, 3], dim=-1)
                else:
                    sp_tr, sp_d_scale = self.sp_cache[tid].split([7, 3], dim=-1)
                    sp_d_rot = sp_tr[:, 3:]
                sp_tr = SE3.InitFromVec(sp_tr)
                points_t1 = points_c + self.warp(
                    points_c, self.sp_points, sp_tr, None, None, None,
                    self.sp_weights, self.sp_knn, self.warp_method
                )[0]
                d_xyz, _, _, sk_tr, sk_d_rot, sk_d_scale, _, _, _ = self.sk_stage(points_c, t, tid)
                sk_tr = SE3.InitFromVec(sk_tr)
                points_t2 = points_c + d_xyz
                cmp_t = (sp_tr.inv() * sk_tr).log().norm(dim=-1).mean()
                cmp_t = self.loss_funcs('cmp_t', cmp_t)
                cmp_p = self.loss_funcs('cmp_p', F.mse_loss, points_t1, points_t2)
                cmp_r = self.loss_funcs('cmp_r', F.mse_loss, sk_d_rot, sp_d_rot)
                cmp_s = self.loss_funcs('cmp_s', F.mse_loss, sk_d_scale, sp_d_scale)
                loss_meter.update({
                    'total': cmp_t + cmp_r + cmp_s + cmp_p,
                    'cmp_p': cmp_p, 'cmp_t': cmp_t, 'cmp_r': cmp_r, 'cmp_s': cmp_s
                })
            logging.info(f'final loss: {loss_meter.average}')
        self.sk_W_is_init = self.sk_W_is_init.new_tensor(True)
        self.train(state)

    @torch.no_grad()
    def init_skeleton(self, progress_func=None):
        if self.sk_is_init:
            return
        if progress_func is None and self._task is not None:
            progress = getattr(self._task, 'progress', None)  # type: Optional[my_ext.Progress]
        else:
            progress = None

        ## cache the data from sp stage
        for i in range(len(self.train_db_times)):
            out = self.sp_deform_net(self.sp_points, self.train_db_times[i].cuda().view(-1, 1))
            d_xyz, d_scale = out['d_xyz'], out['d_scaling']
            rot_bias = d_xyz.new_tensor([0., 0., 0., 1.])
            d_rot = ops_3d.quaternion.normalize(out['d_rotation'] + rot_bias)
            if self.warp_method == 'LBS_c':
                spR = SO3.InitFromVec(d_rot)
                spT = SE3.InitFromVec(torch.cat([d_xyz + self.sp_points + spR.act(-self.sp_points), spR.vec()], dim=-1))
            elif self.warp_method == 'LBS':
                spT = SE3.InitFromVec(torch.cat([d_xyz, d_rot], dim=-1))
            else:  # elif self.warp_method == 'largest':
                spT = SE3.InitFromVec(torch.cat([d_xyz, d_rot], dim=-1))
            if self.sep_rot:
                g_rot = ops_3d.quaternion.normalize(out['g_rotation'] + rot_bias)
                self.sp_cache[i] = torch.cat([spT.vec(), g_rot, d_scale], dim=-1)
            else:
                self.sp_cache[i] = torch.cat([spT.vec(), d_scale], dim=-1)
        self.sp_weights, self.sp_knn = self.calc_LBS_weight(
            self.points, self.sp_points, self.hyper_feature, self.sp_hyper_feature)
        ## init joint
        if progress is not None:
            progress.pause('train')
            progress.add_task('joint', total=self.joint_init_steps)
            progress.start('joint')
            progress_func = lambda: progress.step('joint')
        self.init_joint(progress_func)
        if progress is not None:
            progress.stop('joint')
            progress.start('train')
        self.update_joint()
        self.global_tr.data.copy_(self.sp_cache[:, self.joint_root, :7])
        ## assign W
        self.joints.data.copy_(self.sp_points)
        a, b, mask = self.joint_pair
        self.joints[mask] = self.joint_pos[a, b]
        if True or self.stages['sk_init'][2] == 0:
            if progress is not None:
                progress.pause('train')
                progress.add_task('sk_deform', total=self.joint_init_steps)
                progress.start('sk_deform')
                progress_func = lambda: progress.step('sk_deform')
            self.init_sk_deform(progress_func)
            if progress is not None:
                progress.stop('sk_deform')
                progress.start('train')
        self.sk_is_init = self.sk_is_init.new_tensor(True)
        if self._task is not None:
            self._task.metric_manager.reset()
            logging.info('reset metric_manager')
            self._task.evaluation('sk init')
            self._task.save_checkpoint('sk_init.pth', use_prefix=False, force=True, manage=False)
        logging.info('[red]Init skeleton')

    def kinematic(self, joints: Tensor, t: Tensor, g_tr: Tensor = None, time_id: int = None, sk_r_delta: Tensor = None):
        """运动学模型"""
        ## get joint rotation
        if self.training or not self.test_time_interpolate:
            x_input = joints if self.sk_feature is None else torch.cat([joints, self.sk_feautre], dim=1)
            sk_r, d_rot, d_scale = self.sk_deform_net(x_input, t)
            if sk_r.shape[-1] == 4:
                sk_r = ops_3d.quaternion.normalize(sk_r + sk_r.new_tensor([0., 0., 0., 1.]))
            if self.training and time_id is not None:
                with torch.no_grad():
                    self.sk_cache[time_id] = torch.cat([sk_r, d_rot, d_scale], dim=-1)
        elif time_id is None:
            w, idx1, idx2 = utils.get_interpolate_weight(self.train_db_times, t)
            sk_r, d_rot, d_scale = torch.lerp(self.sk_cache[idx1], self.sk_cache[idx2], w).split(self.sk_dims, -1)
            sk_r = ops_3d.quaternion.normalize(sk_r)
        else:
            sk_r, d_rot, d_scale = self.sk_cache[time_id].split(self.sk_dims, dim=-1)
        sk_r = self.to_SO3(sk_r)
        if sk_r_delta is not None:
            sk_r = (SO3.exp(sk_r_delta) if sk_r_delta.shape[-1] == 3 else SO3.InitFromVec(sk_r_delta)) * sk_r
        ## skeleton transforms
        sk_t = joints + sk_r.act(-joints)
        sk_tr = SE3.InitFromVec(torch.cat([sk_t, sk_r.vec()], dim=-1))
        if g_tr is None:
            g_tr = None
        elif isinstance(g_tr, SE3):
            pass
        elif g_tr.shape[-1] == 6:
            g_tr = SE3.exp(g_tr)  # .matrix()
        elif g_tr.shape[-1] == 7:
            g_tr = SE3.InitFromVec(g_tr)  # .matrix()
        elif g_tr.shape[-2:] == (4, 4):
            g_tr = SE3.InitFromVec(ops_3d.rigid.Rt_to_quaternion(g_tr))
        else:
            raise ValueError(f'g_tr got shape {g_tr.shape}')
        # sk_T = skeleton_warp(sk_tr.matrix(), g_tr.matrix(), self.joint_parents, self.joint_root)
        # sk_T = SE3.InitFromVec(ops_3d.rigid.Rt_to_quaternion(sk_T))
        sk_T = skeleton_warp_SE3(sk_tr, g_tr, self.joint_parents, self.joint_root)
        return sk_T, d_rot, d_scale

    def sk_stage(
        self, points: Tensor, t: Tensor, time_id: Union[Tensor, int, None] = None, sk_r_delta=None, detach=False,
        **kwargs
    ):
        points = points.detach()
        time_id = time_id.item() if isinstance(time_id, Tensor) else time_id
        # get global transform
        if not self.sk_is_init:
            sp_points = self.sp_points[..., :3]
            sp_out = self.sp_deform_net(sp_points, t)
            d_rot = ops_3d.quaternion.normalize(sp_out['d_rotation'] + sp_points.new_tensor([0, 0, 0, 1.]))
            d_rot = SO3.InitFromVec(d_rot)
            if self.warp_method == 'LBS_c':
                d_xyz = d_rot.act(-sp_points) + sp_points + sp_out['d_xyz']
            else:
                d_xyz = sp_out['d_xyz']
            g_tr = SE3.InitFromVec(torch.cat([d_xyz, d_rot.vec()], dim=-1))
        elif time_id is None:
            w, t1_idx, t2_idx = utils.get_interpolate_weight(self.train_db_times, t)
            g_tr = torch.lerp(self.global_tr[t1_idx], self.global_tr[t2_idx], w).view(-1)
        else:
            g_tr = self.global_tr[time_id].view(-1)
        ## Kinematic transform
        if self.sk_is_init:
            joints = self.joints
        else:
            joints = self.sp_points[..., :3].clone().detach()
            a, b, mask = self.joint_pair
            joints[mask] = self.joint_pos[a, b]
        # if detach:
        #     joints, g_tr = joints.detach(), g_tr.detach()
        with torch.no_grad() if detach else nullcontext():
            sk_T, sk_d_rot, sk_d_scale = self.kinematic(joints, t, g_tr, time_id, sk_r_delta)

        ## Linear Blend Skinning
        weights, indices = self.calc_LBS_weight(points, joints)
        # if detach:
        # weights, indices = weights.detach(), indices.detach()
        d_xyz = (sk_T[indices].act(points[:, None]) * weights[..., None]).sum(dim=1) - points
        d_rot = (sk_d_rot[indices] * weights[..., None]).sum(dim=1)
        d_scale = (sk_d_scale[indices] * weights[..., None]).sum(dim=1)
        return d_xyz, d_rot, d_scale, sk_T.vec(), sk_d_rot, sk_d_scale, g_tr, weights, indices

    def get_now_stage(self, stage=None, now_step: int = None):
        now_step = self._step if now_step is None else now_step
        if stage is None:
            for stage, (start, end, num) in self.stages.items():
                if start < now_step <= end:
                    return stage
        return stage

    def forward(self, t: Tensor = None, campos: Tensor = None, stage=None, time_id: int = None, **kwargs):
        stage = self.get_now_stage(stage)
        outputs = {'opacity': self.opacity_activation(self._opacity)}
        sh_features = torch.cat((self._features_dc, self._features_rest), dim=1)
        t = t.view(-1, 1)

        points, scales, rotations = self._xyz, self._scaling, self._rotation
        if stage == 'static':
            d_xyz, d_rotation, d_scaing = 0, 0, 0
        elif stage == 'init' or stage == 'init_fix':
            d_xyz, d_rotation, d_scaing = self.init_stage(points, t)
            scales = scales.mean(dim=(0, 1) if self.scales_all_same else 1, keepdim=True).expand_as(scales)
            if stage == 'init_fix':
                d_xyz, d_rotation, d_scaing = d_xyz.detach(), d_rotation.detach(), d_scaing.detach()
        elif stage == 'sp_fix' or stage == 'sp':
            d_xyz, d_rotation, d_scaing, spT, sp_d_rot, sp_d_scale, knn_w, knn_i = self.sp_stage(points, t, time_id)
            if stage == 'sp_fix':
                d_xyz, d_rotation, d_scaing = d_xyz.detach(), d_rotation.detach(), d_scaing.detach()
            outputs.update(_spT=spT, _knn_w=knn_w, _knn_i=knn_i, _sp_rot=sp_d_rot, _sp_scale=sp_d_scale)
        else:  # sk / sk_init
            self.init_skeleton()
            if stage == 'sk_init':
                points, scales, rotations = points.detach(), scales.detach(), rotations.detach()
                sh_features = sh_features.detach()
                outputs['opacity'].detach_()
                d_xyz, d_rotation, d_scaing, skT, sk_d_rot, sk_d_scale, g_tr, knn_w, knn_i = self.sk_stage(
                    points, t, time_id=time_id, detach=False, **kwargs)
            else:
                d_xyz, d_rotation, d_scaing, skT, sk_d_rot, sk_d_scale, g_tr, knn_w, knn_i = self.sk_stage(
                    points, t, time_id=time_id, detach=stage == 'sk_fix', **kwargs)
            outputs.update(_skT=skT, _knn_w=knn_w, _knn_i=knn_i, _sk_rot=sk_d_rot, _sk_scale=sk_d_scale,
                           _d_xyz=d_xyz, _d_rot=d_rotation, _d_scale=d_scaing)
        outputs['points'] = points + d_xyz
        if self.convert_SHs_python and campos is not None:
            outputs['colors'] = self.get_colors(sh_features, points, campos)
        else:
            outputs['sh_features'] = sh_features

        if self.compute_cov3D:
            outputs['covariance'] = self.covariance_activation(scales * kwargs.get('scaling_modifier', 1.0), rotations)
            assert False
        else:
            outputs['scales'] = self.scaling_activation(scales) + d_scaing
            outputs['rotations'] = self.rotation_activation(rotations + d_rotation)
        return outputs

    def render(
        self,
        *args,
        t: Tensor = None,
        info,
        background: Tensor = None,
        time_id=None,
        scale_modifier=1.0,
        stage=None,
        **kwargs
    ):
        stage = self.get_now_stage(stage)
        inputs = self.prepare_inputs(info, t, background, scale_modifier)
        outputs = {}
        for b, (raster_settings, campos, bg, t) in enumerate(inputs):
            net_out = self(t=t, campos=campos, stage=stage, time_id=time_id)
            if 'hook' in kwargs:
                net_out = kwargs['hook'](net_out)
            outputs_b = {}
            for k in list(net_out.keys()):  # type: str
                if k.startswith('_'):
                    outputs_b[k] = net_out.pop(k)
            outputs_b.update(self.gs_rasterizer(**net_out, raster_settings=raster_settings))
            images = torch.permute(outputs_b['images'], (1, 2, 0))
            if not self.use_official_gaussians_render and background is not None:
                images = images + (1 - outputs_b['opacity'][..., None]) * bg.squeeze(0)
            outputs_b['images'] = images
            outputs_b['points'] = net_out['points']
            if b == 0:
                outputs = {k: [v] for k, v in outputs_b.items() if v is not None}
            else:
                for k, v in outputs_b.items():
                    if v is not None:
                        outputs[k].append(v)
        outputs = {k: torch.stack(v, dim=0) if k != 'viewspace_points' else v for k, v in outputs.items()}  # noqa
        outputs['stage'] = stage
        return outputs

    @torch.no_grad()
    def update_joint(self, verbose=True, use_hyper=False):
        if self.sk_knn_num > 0:
            cost = self.joint_cost.clone()
            sp_points = self.sp_points
            if self.hyper_dim > 0 and use_hyper:
                sp_points = torch.cat([self.sp_points, self.sp_hyper_feature], dim=-1)
            sp_dist = torch.cdist(sp_points, sp_points)
            knn_dist, _ = torch.kthvalue(sp_dist, min(self.num_superpoints, self.sk_knn_num + 1), dim=-1, keepdim=True)
            cost[sp_dist > knn_dist] += cost.max().abs() + 1
        else:
            cost = self.joint_cost
        self.joint_parents, self.joint_depth, root = joint_discovery(cost)
        self.joint_root = self.joint_depth.new_tensor(root)

        mask = torch.ones(self.num_superpoints, device=self.joint_parents.device, dtype=torch.bool)
        mask[self.joint_root] = False
        a = torch.arange(self.num_superpoints, device=self.joint_parents.device)[mask]
        b = self.joint_parents[mask, 0]
        self._joint_pair = (a, b, mask)
        if verbose:
            logging.info('update joint')

    @property
    def joint_pair(self):
        if self._joint_pair is None:
            mask = torch.ones_like(self.joint_parents[:, 0], dtype=torch.bool)
            mask[self.joint_root] = 0
            a = torch.arange(self.num_superpoints, device=self.joint_parents.device)[mask]
            b = self.joint_parents[mask, 0]
            self._joint_pair = (a, b, mask)
        return self._joint_pair

    def find_joint_loss(self, sp_tr: Tensor, joint_rot: SO3, time_id, update_joint=True):
        self.init_joint_pos()
        sp_tr = sp_tr.detach()
        # if self.sk_tr_batch_size <= 0:
        time_id = time_id if isinstance(time_id, Tensor) else torch.tensor(time_id, device=sp_tr.device)
        time_id = time_id.view(-1)
        T = (SE3.InitFromVec(sp_tr) if isinstance(sp_tr, Tensor) else sp_tr)[None]  # [1, M]
        # else:
        #     time_id = torch.randint(0, self.num_frames, size=(self.sk_tr_batch_size,), device=sp_tr.device)
        #     T = SE3.InitFromVec(self.sp_delta_tr[time_id])  # [B, M]
        # rot = SO3.InitFromVec(self.joint_rot[time_id]) if joint_rot is None else joint_rot[time_id]  # [B, M, M]
        rot = joint_rot[time_id]  # [B, M, M]
        translate = self.joint_pos[None] - rot.act(self.joint_pos[None])  # [B, M, M]
        local_tr = SE3.InitFromVec(torch.cat([translate, rot.vec()], dim=-1))
        joint_T = T[:, None, :] * local_tr
        joint_dist = (T[:, :, None].inv() * joint_T).log().norm(dim=-1).mean(dim=0)

        # T = (T[None, :].inv() * T[:, None]).matrix()
        # joint_dist = T[..., :3, 3] - (self.joint_pos - ops_3d.apply(self.joint_pos, T[..., :3, :3]))
        # joint_dist = joint_dist.norm(dim=-1, keepdim=False)  # [M, M]

        joint_pos_t = T[:, None, :].act(self.joint_pos[None])  # [B, M, M, 3]
        joint_dist = joint_dist + (joint_pos_t - joint_pos_t.transpose(1, 2)).norm(dim=-1).mean(dim=0)  # [M, M]

        with torch.no_grad():
            self.joint_cost = self.joint_cost * self.sk_momentum + joint_dist * (1. - self.sk_momentum)
            if update_joint:
                self.update_joint(verbose=False)
        a, b, _ = self.joint_pair
        sp_joint_loss = (joint_dist[a, b] + joint_dist[b, a]) * 0.5
        return sp_joint_loss.mean(), joint_dist.mean()

    def loss_joint_discovery(self, sp_T: Tensor, sp_T_c: Tensor = None, update_joint=True):
        """discovery joints during traning"""
        self.init_joint_pos()
        sp_T = sp_T.detach() if self.sp_guided_detach else sp_T
        if sp_T_c is not None:
            sp_T = sp_T @ torch.inverse(sp_T_c)
        sp_T = ops_3d.rigid.quaternion_to_Rt(sp_T)
        if self.canonical_time_id < 0:
            T = (torch.inverse(sp_T[None, :]) @ sp_T[:, None])
            joint_dist = T[..., :3, 3] - (self.joint_pos - ops_3d.apply(self.joint_pos, T[..., :3, :3]))
        else:
            # tb = ops_3d.apply(self.joint_pos, sp_T[:, None]) - ops_3d.apply(self.joint_pos, sp_T[None, :, :3, :3])
            # joint_dist = tb - sp_T[None, :, :3, 3]
            tb = ops_3d.apply(self.joint_pos, sp_T[None, :]) - ops_3d.apply(self.joint_pos, sp_T[:, None, :3, :3])
            joint_dist = tb - sp_T[:, None, :3, 3]
        joint_dist = joint_dist.norm(dim=-1, keepdim=False)  # [M, M]

        joint_pos_t = ops_3d.apply(self.joint_pos, sp_T)  # [M, M, 3]
        joint_dist = joint_dist + (joint_pos_t - joint_pos_t.transpose(0, 1)).norm(dim=-1)  # [M, M]

        with torch.no_grad():
            if self.training:
                self.joint_cost = self.joint_cost * self.sk_momentum + joint_dist * (1. - self.sk_momentum)
                if update_joint or self._joint_pair is None:
                    self.update_joint()
        a, b, _ = self.joint_pair
        sp_joint_loss = (joint_dist[a, b] + joint_dist[b, a]) * 0.5
        return sp_joint_loss.mean(), joint_dist.mean()
        # return joint_dist.mean()

    def loss_weight_sparsity(self, weight: Tensor, eps=1e-7):
        return -(weight * torch.log(weight + eps) + (1 - weight) * torch.log(1 - weight + eps)).mean()

    def update_gs_knn(self, force=False):
        if self._is_gs_knn_updated:
            return
        self._is_gs_knn_updated = True
        if not (force or (self.gs_knn_index.shape[0] != self.points.shape[0]) or
                utils.check_interval(self._step, *self.gs_knn_update_interval, force_end=False)):
            return
        from pykdtree.kdtree import KDTree
        points = self.points.detach().cpu().numpy()
        kdtree = KDTree(points)
        knn_dist, knn_index = kdtree.query(points, k=self.gs_knn_num + 1)
        self.gs_knn_index = torch.from_numpy(knn_index.astype(np.int32)).to(self.gs_knn_index)
        self.gs_knn_dist = torch.from_numpy(knn_dist).to(self.gs_knn_dist)
        logging.info('update guassian knn')

    def loss_weight_smooth(self, weight: Tensor):
        self.update_gs_knn()
        return (weight[:, None] - weight[self.gs_knn_index]).abs().mean()

    def loss_points_arap(self, points_t: Tensor):
        # self.update_gs_knn()
        points_c = self.points
        # indices = self.gs_knn_index[:, 1:]
        nn_dist, indices, _ = knn_points(points_t[None], points_t[None], K=self.gs_knn_num + 1)
        indices = indices[0, :, 1:]
        dict_c = (points_c[:, None] - points_c[indices]).square().sum(dim=-1)
        dict_t = (points_t[:, None] - points_t[indices]).square().sum(dim=-1)
        return (dict_c - dict_t).abs().mean()

    def loss_sp_arap(self, sp_se3: SE3):
        sp_points_c = self.sp_points[..., :3]
        sp_points_t = sp_se3.act(sp_points_c)
        with torch.no_grad():
            sp_dist = torch.cdist(sp_points_c, sp_points_c)
            k_dist, knn = torch.topk(sp_dist, dim=1, k=min(self.num_superpoints, self.sk_knn_num + 1), largest=False)
            knn = knn[:, 1:]
        # loss = F.mse_loss(sp_tr[:, None].repeat(1, num_knn, 1), sp_tr[knn])
        loss = (sp_se3[:, None].inv() * sp_se3[knn]).log().norm(dim=-1).mean()
        dist_c = (sp_points_c[:, None] - sp_points_c[knn]).square().sum(dim=-1)
        dist_t = (sp_points_t[:, None] - sp_points_t[knn]).square().sum(dim=-1)
        arap_ct_loss = (dist_c - dist_t).abs().mean()
        return loss, arap_ct_loss

    def loss_arap(self, t=None, delta_t=0.05, t_samp_num=2, points: Tensor = None):
        if points is None:
            points = self.sp_points
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(points.shape[0], t_samp_num, 1)  # M, T, 1
        x = points[:, None, :].repeat(1, t_samp_num, 1).detach()
        dx = self.sp_deform_net(x=x.reshape(-1, 3), t=t_samp.reshape(-1, 1))['d_xyz']
        nodes_t = points[:, None, :3].detach() + dx.reshape(-1, t_samp_num, 3)  # M, T, 3
        hyper_nodes = nodes_t[:, 0]  # M, 3
        ii, jj, knn, weight = cal_connectivity_from_points(hyper_nodes, K=10)  # connectivity of control nodes
        error = cal_arap_error(nodes_t.permute(1, 0, 2), ii, jj, knn)
        return error

    def loss_elastic(self, t=None, delta_t=0.005, K=2, t_samp_num=8, points: Tensor = None, hyper: Tensor = None):
        if points is None:
            points = self.sp_points
            hyper = self.sp_hyper_feature
        num_points = points.shape[0]
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(num_points, t_samp_num, 1)
        x = points[:, None, :].repeat(1, t_samp_num, 1).detach()
        node_trans = self.sp_deform_net(x=x.reshape(-1, 3), t=t_samp.reshape(-1, 1))['d_xyz']
        nodes_t = points[:, None, :3].detach() + node_trans.reshape(-1, t_samp_num, 3)  # M, T, 3

        # Calculate weights of nodes NN
        nn_weight, nn_idx = self.calc_LBS_weight(points, points, hyper, hyper, K=K + 1)
        nn_weight, nn_idx = nn_weight[:, 1:], nn_idx[:, 1:]  # M, K

        # Calculate edge deform loss
        edge_t = (nodes_t[nn_idx] - nodes_t[:, None]).norm(dim=-1)  # M, K, T
        edge_t_var = edge_t.var(dim=2)  # M, K
        edge_t_var = edge_t_var / (edge_t_var.detach() + 1e-5)
        arap_loss = (edge_t_var * nn_weight).sum(dim=1).mean()
        return arap_loss

    def loss_acc(self, t=None, delta_t=.005, points: Tensor = None):
        if points is None:
            points = self.sp_points
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t = torch.stack([t - delta_t, t, t + delta_t])
        t = t[None, :, None].expand(points.shape[0], 3, 1)
        x = points[:, None, :].repeat(1, 3, 1).detach()
        node_trans = self.sp_deform_net(x=x.reshape(-1, 3), t=t.reshape(-1, 1))['d_xyz']
        nodes_t = points[:, None, :3].detach() + node_trans.reshape(-1, 3, 3)  # M, 3, 3
        acc = (nodes_t[:, 0] + nodes_t[:, 2] - 2 * nodes_t[:, 1]).norm(dim=-1)  # M
        acc = acc / (acc.detach() + 1e-5)
        acc_loss = acc.mean()
        return acc_loss

    def loss_guided_sp(self, losses, time_id, outputs, t):
        sp_tr = outputs['_spT'][..., :7].squeeze(0).detach()
        sp_d_rot = outputs['_sp_rot'].squeeze(0).detach()
        sp_d_scale = outputs['_sp_scale'].squeeze(0).detach()
        a, b, mask = self.joint_pair
        joints = torch.zeros_like(self.sp_points)
        joints[a] = self.joint_pos[a, b]
        sk_tr, sk_d_rot, sk_d_scale = self.kinematic(joints, t, sp_tr[self.joint_root], time_id)
        guided_loss = (SE3.InitFromVec(sp_tr).inv() * sk_tr).log().norm(dim=-1).mean()
        losses['cmp_t'] = self.loss_funcs('cmp_t', guided_loss)
        # losses['cmp_t'] = self.loss_funcs('cmp_t', F.mse_loss, sk_tr.vec(), sp_tr)
        losses['cmp_r'] = self.loss_funcs('cmp_r', F.mse_loss, sk_d_rot, sp_d_rot)
        losses['cmp_s'] = self.loss_funcs('cmp_s', F.mse_loss, sk_d_scale, sp_d_scale)
        return

    def loss_guided_sk(self, losses, time_id, outputs):
        sp_tr = self.sp_cache[time_id]
        if self.sep_rot:
            sp_d_tr, sp_d_rot, sp_d_scale = sp_tr.split([7, 4, 3], dim=-1)
        else:
            sp_d_tr, sp_d_scale = sp_tr.split([7, 3], dim=-1)
            sp_d_rot = sp_d_tr[:, 3:]
        sk_tr, sk_d_rot, sk_d_scale = outputs['_skT'].squeeze(0), outputs['_sk_rot'], outputs['_sk_scale']

        guided_loss = (SE3.InitFromVec(sp_d_tr).inv() * SE3.InitFromVec(sk_tr)).log().norm(dim=-1).mean()
        losses['cmp_t'] = self.loss_funcs('cmp_t', guided_loss)
        losses['cmp_r'] = self.loss_funcs('cmp_r', F.mse_loss, sk_d_rot.squeeze(0), sp_d_rot)
        losses['cmp_s'] = self.loss_funcs('cmp_s', F.mse_loss, sk_d_scale.squeeze(0), sp_d_scale)

    def loss_guided_sk_v2(self, losses, time_id, outputs):
        sk_d_xyz, sk_d_rot, sk_d_scale = outputs['_d_xyz'], outputs['_d_rot'], outputs['_d_scale']
        with torch.no_grad():
            if self.sep_rot:
                sp_tr, sp_d_rot, sp_d_scale = self.sp_cache[time_id].split([7, 4, 3], dim=-1)
            else:
                sp_tr, sp_d_scale = self.sp_cache[time_id].split([7, 3], dim=-1)
                sp_d_rot = sp_tr[:, 3:]
            sp_weights, sp_knn = self.sp_weights, self.sp_knn
            points = self.points
            if self.warp_method == 'LBS' or self.warp_method == 'LBS_c':
                sp_d_xyz = (SE3.InitFromVec(sp_tr)[sp_knn].act(points[:, None]) * sp_weights[..., None]).sum(dim=1)
                sp_d_xyz = sp_d_xyz - points
            else:  # self.warp_method == 'largest'
                sp_d_xyz = SE3.InitFromVec(sp_tr)[self.p2sp].act(points) - points
            sp_d_rot = (sp_d_rot[sp_knn] * sp_weights[..., None]).sum(dim=1)
            sp_d_scale = (sp_d_scale[sp_knn] * sp_weights[..., None]).sum(dim=1)

        losses['cmp_t'] = self.loss_funcs('cmp_t', F.mse_loss, sk_d_xyz.squeeze(0), sp_d_xyz)
        losses['cmp_r'] = self.loss_funcs('cmp_r', F.mse_loss, sk_d_rot.squeeze(0), sp_d_rot)
        losses['cmp_s'] = self.loss_funcs('cmp_s', F.mse_loss, sk_d_scale.squeeze(0), sp_d_scale)

    def loss_reconstruct(self, losses, outputs):
        # if self.loss_funcs.w('re_tr'):
        #     re_sp_tr = get_superpoint_features(outputs['_tr'], Neighbor, W, self.num_superpoints)
        #     losses['re_tr'] = self.loss_funcs('re_tr', F.mse_loss(re_sp_tr, outputs['_sp_tr']))
        if self.loss_funcs.w('re_pos') > 0:
            weights, indices = outputs['_knn_w'][0], outputs['_knn_i'][0]
            sp_se3 = SE3.InitFromVec(outputs['_spT'][0])
            re_sp = get_superpoint_features(outputs['points'][0], indices, weights, self.num_superpoints)
            sp = sp_se3.act(self.sp_points)
            losses['re_pos'] = self.loss_funcs('re_pos', F.mse_loss(sp, re_sp))
        return losses

    def loss_canonical_net(self, points, t, stage='init'):
        if stage == 'sp' and len(self.canonical_replace_steps) == 0:
            return 0
        tc = self.train_db_times[self.canonical_time_id]
        if stage == 'init':
            with torch.no_grad():
                points_c = self.init_stage(self._xyz, tc)[0] + self._xyz
            points_t = self.init_stage(points_c, t, use_canonical_net=True)[0] + points_c
        else:
            with torch.no_grad():
                d_xyz, d_rotation, d_scaing, spT, sp_d_rot, sp_d_scale, knn_w, knn_i = self.sp_stage(self._xyz, tc)
                points_c = d_xyz + self._xyz
                sp_points_c = SE3.InitFromVec(spT).act(self.sp_points)
            points_t = self.sp_stage(points_c, t, sp_points=sp_points_c, use_canonical_net=True)[0] + points_c
        return F.mse_loss(points_t, points.detach())

    def loss(self, inputs, outputs, targets, info):
        self._is_gs_knn_updated = False
        time_id = inputs['time_id'].item() if 'time_id' in inputs else None
        t = inputs['t'].view(-1)
        stage = outputs['stage']

        losses = {}
        image = outputs['images']
        gt_img = targets['images'][..., :3]
        H, W, C = image.shape[-3:]
        image, gt_img = image.view(1, H, W, C), gt_img.view(1, H, W, C)
        losses['rgb'] = self.loss_funcs('image', image, gt_img)
        losses['ssim'] = self.loss_funcs('ssim', image, gt_img)
        if stage == 'sp':
            losses['elastic'] = self.loss_funcs('elastic', self.loss_elastic, t, self.time_interval)
            losses['acc'] = self.loss_funcs('acc', self.loss_acc, t, 3.0 * self.time_interval)
            losses['arap'] = self.loss_funcs('arap', self.loss_arap)
        if stage == 'sp' or stage == 'init':
            if self.canonical_net is not None and self._step <= max(self.canonical_replace_steps) + 5:
                losses['c_net'] = self.loss_funcs('c_net', self.loss_canonical_net, outputs['points'][0], t, stage)
        if stage == 'init':
            if self.points.shape[0] <= self.num_superpoints:
                points, hyper = self.points, self.hyper_feature
            else:
                index = torch.randperm(self.points.shape[0], device=self.device)[:self.num_superpoints]
                points, hyper = self.points[index], self.hyper_feature[index]
            losses['elastic'] = self.loss_funcs('elastic', self.loss_elastic, t, self.time_interval, points=points,
                                                hyper=hyper)
            losses['acc'] = self.loss_funcs('acc', self.loss_acc, t, 3.0 * self.time_interval, points=points)
            losses['arap'] = self.loss_funcs('arap', self.loss_arap, points=points)
            losses['arap_p'] = self.loss_funcs('p_arap_ct_init', self.loss_points_arap, outputs['points'][0])
        if stage == 'sp' and self.training:
            with torch.no_grad():
                cache = torch.cat([outputs[k] for k in ['_spT', '_sp_rot', '_sp_scale'] if k in outputs], dim=-1)
                self.sp_cache[time_id] = cache.squeeze(0)
            self.loss_reconstruct(losses, outputs)
        if stage == 'sp' and self.loss_funcs.w('joint') > 0 and self._step >= self.joint_update_interval[1]:
            spT = outputs['_spT'][0]
            update_joint = utils.check_interval_v2(self._step, *self.joint_update_interval, close='[)')
            best_cost, all_cost = self.loss_joint_discovery(spT, None, update_joint)
            losses['joint'] = self.loss_funcs('joint', best_cost)
            losses['joint_all'] = self.loss_funcs('joint_all', all_cost)
            if self.loss_funcs.w('jp_dist') > 0:
                a, b, mask = self.joint_pair
                with torch.no_grad():
                    sp_points_t = ops_3d.apply(self.sp_points[..., :3], spT)
                joints = ops_3d.apply(self.joint_pos[a, b], spT[b])
                jd_dist = F.mse_loss(joints, sp_points_t[a]) + F.mse_loss(joints, sp_points_t[b])
                losses['jp_dist'] = self.loss_funcs('jp_dist', jd_dist)
        if stage == 'sp' and self._step >= self.loss_arap_start_step >= 0:
            if self.loss_funcs.w('sp_arap_t') > 0:
                arap_t, arap_ct = self.loss_sp_arap(SE3.InitFromVec(outputs['_spT'][0]))
                losses['arap_t'] = self.loss_funcs('sp_arap_t', arap_t)
                losses['arap_ct'] = self.loss_funcs('sp_arap_ct', arap_ct)
        if stage == 'sp' or stage == 'sk':
            losses['sparse'] = self.loss_funcs('sparse', self.loss_weight_sparsity, outputs['_knn_w'])
            losses['smooth'] = self.loss_funcs('smooth', self.loss_weight_smooth, outputs['_knn_w'][0])
        if stage == 'sp' and self._step > self.guided_step_start >= 0:
            self.loss_guided_sp(losses, time_id, outputs, inputs['t'])
        if stage == 'sk_init' and self.training:
            # self.loss_guided_sk(losses, time_id, outputs)
            self.loss_guided_sk_v2(losses, time_id, outputs)
            losses['rgb'].detach_()
            losses['ssim'].detach_()
        return {k: v for k, v in losses.items() if isinstance(v, Tensor)}

    def update_skeleton_shapes(self):
        M = self.num_superpoints
        if M == self.joint_parents.shape[0]:
            return
        self.sk_cache = self.sk_cache.new_zeros(self.num_frames, M, self.sk_cache.shape[-1])
        if self.sk_feature is not None:
            self.sk_feature = nn.Parameter(self.sk_feature.new_zeros(M, self.sk_feature_dim))
        self.update_joint()

    @torch.no_grad()
    def superpoint_prune_split(self, optimizer: torch.optim.Optimizer):
        ## prune
        weight, knn = self.calc_LBS_weight(self.points, self.sp_points, self.hyper_feature, self.sp_hyper_feature)
        W = torch.scatter_reduce(weight.new_zeros(self.num_superpoints), 0, knn.view(-1), weight.view(-1), 'sum')
        mask = torch.ge(W, self.sp_prune_threshold)
        num_pruned = self.num_superpoints - mask.sum().item()
        if num_pruned > 0:
            names = ['sp_points', "joints"]
            if self.hyper_dim > 0:
                names.append('sp_hyper_feature')
            tensors = self.change_optimizer(optimizer, mask, names, op='prune')
            for name, tensor in tensors.items():
                setattr(self, name, tensor)
            if self._sp_radius is not None:
                self._sp_radius = self.change_optimizer(optimizer, mask, 'sp_radius', op='prune')['sp_radius']
            if self._sp_weight is not None:
                self._sp_weight = self.change_optimizer(optimizer, mask, 'sp_weight', op='prune')['sp_weight']
            if self.sp_W is not None:
                self.sp_W = self.change_optimizer(optimizer, mask, ['sp_W'], op='prune', dim=1)['sp_W']
            if self.hyper_dim > 0:
                self.sp_hyper_feature = self.change_optimizer(optimizer, mask, 'sp_hyper', op='prune')['sp_hyper']
            # self.sk_W = self.change_optimizer(optimizer, mask, ['sk_W'], op='prune', dim=1)['sk_W']
            self.sp_cache = self.sp_cache[:, mask, :]
            self.joint_cost = self.joint_cost[mask][:, mask]
            self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune')['joint_pos']
            self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune', dim=1)['joint_pos']
            # if self.sk_tr_batch_size >= 0:
            #     self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=1)['joint_rot']
            #     self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=2)['joint_rot']
            # logging.info(f'prune superpoints {self.num_superpoints} to {self.sp_points.shape[0]}')
            ## split
            weight, knn = self.calc_LBS_weight(self.points, self.sp_points, self.hyper_feature, self.sp_hyper_feature)
            W = W[mask]

        # mask = torch.ge((self.sp_grad_accum / self.sp_grad_denom.clamp_min(1e-6)), self.sp_split_threshold)[mask]
        p_grad = torch.nan_to_num(self.xyz_gradient_accum / self.denom, 0)
        sp_grad = torch.index_add(p_grad.new_zeros(self.num_superpoints), 0, knn.view(-1), (p_grad * weight).view(-1))
        sp_weight = torch.index_add(p_grad.new_zeros(self.num_superpoints), 0, knn.view(-1), weight.view(-1))
        mask = torch.ge((sp_grad / sp_weight.clamp_min(1e-6)), self.sp_split_threshold)

        W_9 = torch.kthvalue(W, int(0.9 * len(W)))[0]
        mask = torch.logical_or(mask, W.ge(2. * W_9))

        num_split = mask.sum().item()
        if num_split > 0:
            weight_sum = torch.scatter_reduce(
                weight.new_zeros(self.num_superpoints), dim=0, index=knn.view(-1), src=weight.view(-1), reduce='sum')
            weight = weight / weight_sum.clamp_min(1e-6)[knn]
            new_points = torch.scatter_reduce(
                torch.zeros_like(self.sp_points),
                dim=0,
                index=knn.view(-1, 1).expand(-1, 3),
                src=(self.points[:, None, :] * weight[:, :, None]).view(-1, 3),
                reduce='sum'
            )
            self.sp_points = self.change_optimizer(optimizer, new_points[mask], 'sp_points', op='concat')['sp_points']
            self.joints = self.change_optimizer(optimizer, new_points[mask], 'joints', op='concat')['joints']
            if self.hyper_dim > 0:
                self.sp_hyper_feature = self.change_optimizer(
                    optimizer, self.sp_hyper_feature[mask], 'sp_hyper', op='concat')['sp_hyper']
            if self._sp_radius is not None:
                self._sp_radius = self.change_optimizer(
                    optimizer, self._sp_radius[mask], 'sp_radius', op='concat')['sp_radius']
            if self._sp_weight is not None:
                self._sp_weight = self.change_optimizer(
                    optimizer, self._sp_weight[mask], 'sp_weight', op='concat')['sp_weight']
            if self.sp_W is not None:
                self.sp_W = self.change_optimizer(optimizer, {'sp_W': self.sp_W[:, mask]}, op='concat', dim=1)['sp_W']
            self.sp_cache = torch.cat([self.sp_cache, self.sp_cache[:, mask]], dim=1)
            self.joint_pos = self.change_optimizer(
                optimizer, self.joint_pos[mask], 'joint_pos', op='concat')['joint_pos']
            self.joint_pos = self.change_optimizer(
                optimizer, self.joint_pos[:, mask], 'joint_pos', op='concat', dim=1)['joint_pos']
            # if self.sk_tr_batch_size >= 0:
            #     self.joint_rot = self.change_optimizer(
            #         optimizer, self.joint_rot[:, mask, :], 'joint_rot', op='concat', dim=1)['joint_rot']
            #     self.joint_rot = self.change_optimizer(
            #         optimizer, self.joint_rot[:, :, mask], 'joint_rot', op='concat', dim=2)['joint_rot']
            self.joint_cost = torch.cat([
                torch.cat([self.joint_cost, self.joint_cost[:, mask]], dim=1),
                torch.cat([self.joint_cost[mask], self.joint_cost[:, mask][mask, :]], dim=1)
            ], dim=0)
        if self.p2sp is not None and (num_pruned != 0 or num_split != 0):
            # weights, indices = self.calc_LBS_weight(x=points)
            self.p2sp = torch.gather(knn, -1, weight.argmax(dim=-1, keepdim=True))[:, 0]
        logging.info(f'prune & split superpoints {self.num_superpoints} (-{num_pruned}, +{num_split}) '
                     f'at step {self._step}')
        self.update_skeleton_shapes()

    def all_merge(self, mask, min_index):
        merge_a = torch.nonzero(mask)[:, 0]
        merge_b = min_index[mask]
        merge = torch.arange(self.num_superpoints, device=mask.device)

        def find_merge_root(x):
            if merge[x] != x:
                merge[x] = find_merge_root(merge[x])
            return merge[x]

        for a, b in zip(merge_a, merge_b):
            a = find_merge_root(a)
            b = find_merge_root(b)
            if a != b:
                merge[a] = b
        for i in range(self.num_superpoints):
            merge[i] = find_merge_root(i)
        mask = merge == torch.arange(self.num_superpoints, device=merge.device)
        return mask, merge

    def non_overlap_merge(self, min_diff: Tensor, min_index: Tensor) -> (Tensor, Tensor):
        """不重叠的合并"""
        merge = torch.arange(self.num_superpoints, device=min_diff.device)
        merged = torch.zeros_like(merge, dtype=torch.bool)
        for i in min_diff.argsort().tolist():
            if min_diff[i] >= self.sp_merge_threshold:
                break
            j = min_index[i]
            if merged[i] or merged[j]:
                continue
            merge[i] = j
            merged[i] = True
            merged[j] = True
        mask = merge == torch.arange(self.num_superpoints, device=merge.device)
        return mask, merge

    @torch.no_grad()
    def superpoint_merge(self, optimizer: torch.optim.Optimizer):
        M = self.num_superpoints
        for i, t in enumerate(self.train_db_times):
            outs = self.sp_deform_net(self.sp_points, t)
            d_xyz, d_scale = outs['d_xyz'], outs['d_scaling']
            rot_bias = d_xyz.new_tensor([0, 0, 0, 1.])
            d_rot = ops_3d.quaternion.normalize(outs['d_rotation'] + rot_bias)
            if self.warp_method == 'LBS_c':
                spR = SO3.InitFromVec(d_rot)
                spT = SE3.InitFromVec(torch.cat([d_xyz + self.sp_points + spR.act(-self.sp_points), spR.vec()], dim=-1))
            elif self.warp_method == 'LBS':
                spT = SE3.InitFromVec(torch.cat([d_xyz, d_rot], dim=-1))
            else:  # elif self.warp_method == 'largest':
                spT = SE3.InitFromVec(torch.cat([d_xyz, d_rot], dim=-1))
            if self.sep_rot:
                g_rot = ops_3d.quaternion.normalize(outs['g_rotation'] + rot_bias)
                self.sp_cache[i] = torch.cat([spT.vec(), g_rot, d_scale], dim=-1)
            else:
                self.sp_cache[i] = torch.cat([spT.vec(), d_scale], dim=-1)

        dist = torch.cdist(self.sp_points, self.sp_points[..., :3])  # [N, M]
        knn = torch.topk(dist, k=min(self.num_superpoints, self.num_knn + 1), dim=-1, largest=False)[1][:, 1:]
        tr_diff = (self.sp_cache[:, :, None] - self.sp_cache[:, knn]).norm(dim=-1)  # [T, M, K]
        tr_diff = tr_diff.mean(dim=0)  # [M, K]
        min_diff, min_index = torch.min(tr_diff, dim=1)
        mask = min_diff.lt(self.sp_merge_threshold)
        num_merge = mask.sum().item()
        if num_merge == 0:
            logging.info(f'merge superpoints {self.num_superpoints} (-0) at step {self._step}')
            return
        min_index = knn[torch.arange(self.num_superpoints, device=min_index.device), min_index]
        # mask, merge = self.all_merge(mask, min_index)
        mask, merge = self.non_overlap_merge(min_diff, min_index)

        self.sp_points = self.change_optimizer(optimizer, mask, ['sp_points'], op='prune')['sp_points']
        self.joints = self.change_optimizer(optimizer, mask, ['joints'], op='prune')['joints']
        if self.sp_W is not None:
            new_W = torch.scatter_reduce(torch.zeros_like(self.sp_W), 1, merge.expand_as(self.sp_W), self.sp_W, 'sum')
            self.sp_W = self.change_optimizer(optimizer, new_W[:, mask], ['sp_W'], op='replace', dim=1)['sp_W']
        if self._sp_radius is not None:
            self._sp_radius = self.change_optimizer(
                optimizer, self._sp_radius[mask], ['sp_radius'], op='replace')['sp_radius']
        if self._sp_weight is not None:
            self._sp_weight = self.change_optimizer(
                optimizer, self._sp_weight[mask], ['sp_weight'], op='replace')['sp_weight']

        if self.hyper_dim > 0:
            self.sp_hyper_feature = self.change_optimizer(
                optimizer, self.sp_hyper_feature[mask], 'sp_hyper', op='replace')['sp_hyper']
        # new_W = torch.scatter_reduce(torch.zeros_like(self.sk_W), 1, merge.expand_as(self.sk_W), self.sk_W, 'sum')
        # self.sk_W = self.change_optimizer(optimizer, new_W[:, mask], ['sk_W'], op='replace', dim=1)['sk_W']
        self.sp_cache = self.sp_cache[:, mask, :]
        self.joint_cost = self.joint_cost[mask, :][:, mask]
        self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune')['joint_pos']
        self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune', dim=1)['joint_pos']
        # if self.sk_tr_batch_size >= 0:
        #     self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=1)['joint_rot']
        #     self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=2)['joint_rot']
        num_reduce = M - self.num_superpoints
        if self.p2sp is not None and num_reduce != 0:
            weights, indices = self.calc_LBS_weight(
                self.points, self.sp_points, self.hyper_feature, self.sp_hyper_feature)
            self.p2sp = torch.gather(indices, -1, weights.argmax(dim=-1, keepdim=True))[:, 0]
        logging.info(f'merge superpoints {self.num_superpoints} (-{num_reduce}) at step {self._step}')
        self.update_skeleton_shapes()

    @torch.no_grad()
    def cal_node_importance(self, x: Tensor, K=None, weights=None, feature=None):
        # Calculate the weights of Gaussians on nodes as importance
        if self.hyper_dim > 0:
            x = torch.cat([x, feature], dim=-1)
        K = self.K if K is None else K
        nn_weight, nn_idxs = self.calc_LBS_weight(
            x[..., :3], self.sp_points, self.hyper_feature, self.sp_hyper_feature, K=K)  # [N, K]
        node_importance = torch.zeros_like(self.sp_points[:, 0]).view(-1)
        node_edge_count = torch.zeros_like(self.sp_points[:, 0]).view(-1)
        avg_affected_x = torch.zeros_like(self.sp_points)
        weights = torch.ones_like(x[:, 0]) if weights is None else weights
        node_importance.index_add_(dim=0, index=nn_idxs.view(-1), source=(nn_weight * weights[:, None]).view(-1))
        node_edge_count.index_add_(dim=0, index=nn_idxs.view(-1), source=nn_weight.view(-1))
        avg_affected_x.index_add_(
            dim=0, index=nn_idxs.view(-1),
            source=((nn_weight * weights[:, None]).view(-1, 1) *
                    x[:, None].expand(*nn_weight.shape, x.shape[-1]).reshape(-1, x.shape[-1])))
        avg_affected_x = avg_affected_x / node_importance[:, None]
        node_importance = node_importance / (node_edge_count + 1e-7)
        return node_importance, avg_affected_x, node_edge_count

    @torch.no_grad()
    def node_prune_split(self, optimizer: torch.optim.Optimizer):
        if not self.sp_is_init:
            logging.info('No need to densify nodes before initialization.')
            return
        max_grad = self.densify_threshold
        x_grad = self.xyz_gradient_accum / self.denom
        x_grad[x_grad.isnan()] = 0.
        K = self.num_knn
        weights = x_grad.norm(dim=-1)

        # Calculate the avg importance and coordinate
        node_avg_xgradnorm, node_avg_x, node_edge_count = self.cal_node_importance(
            x=self.points, K=K, weights=weights, feature=self.hyper_feature)

        # Picking pts to densify
        selected_pts_mask = torch.logical_and(
            node_avg_xgradnorm > max_grad, node_avg_x.isnan().logical_not().all(dim=-1))

        pruned_pts_mask = node_edge_count.eq(0)
        if selected_pts_mask.sum() > 0 or pruned_pts_mask.sum() > 0:
            print(f'Add {selected_pts_mask.sum()} nodes and prune {pruned_pts_mask.sum()} nodes. ', end='')
        else:
            return

        # Densify
        if selected_pts_mask.sum() > 0:
            self.sp_points = self.change_optimizer(optimizer, node_avg_x[selected_pts_mask], 'sp_points', op='concat')[
                'sp_points']
            self._sp_radius = self.change_optimizer(
                optimizer, self._sp_radius[selected_pts_mask], 'sp_radius', op='concat')['sp_radius']
            if self._sp_weight is not None:
                self._sp_weight = self.change_optimizer(
                    optimizer, self._sp_weight[selected_pts_mask], 'node_weight', op='concat')['node_weight']

        # Prune
        if pruned_pts_mask.shape[0] < self.sp_points.shape[0]:
            pruned_pts_mask = torch.cat(
                [pruned_pts_mask, pruned_pts_mask.new_zeros([self.sp_points.shape[0] - pruned_pts_mask.shape[0]])])
        if pruned_pts_mask.sum() > 0:
            pruned_pts_mask = ~pruned_pts_mask
            self.sp_points = self.change_optimizer(optimizer, pruned_pts_mask, 'sp_points', op='prune')['sp_points']
            self._sp_radius = self.change_optimizer(
                optimizer, pruned_pts_mask, 'sp_radius', op='prune')['sp_radius']
            if self._sp_weight is not None:
                self._sp_weight = self.change_optimizer(
                    optimizer, pruned_pts_mask, 'node_weight', op='prune')['node_weight']

    def change_with_training_progress(self, step=0, num_steps=1, epoch=0, num_epochs=1):
        total_step = epoch * num_steps + step + 1
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if total_step > self.stages['sp_fix'][0] and self.active_sh_degree < self.max_sh_degree and (
            total_step - self.stages['sp_fix'][0]) % 1000 == 0:
            self.active_sh_degree = self.active_sh_degree + 1
            logging.info(f'increase active sh degree to {self.active_sh_degree} at step {total_step}')
        self._step = total_step

    def hook_before_train_step(self):
        if self.canonical_net is not None and self._step > self.stages['sp_fix'][0] and \
            (self._step in self.canonical_replace_steps):
            with torch.no_grad():
                tc = self.train_db_times[self.canonical_time_id]
                d_xyz, d_rotation, d_scaing, spT, sp_d_rot, sp_d_scale, knn_w, knn_i = self.sp_stage(self._xyz, tc)
                points_c = d_xyz + self._xyz
                sp_points_c = SE3.InitFromVec(spT).act(self.sp_points)
                self._xyz.data = points_c
                self.sp_points.data = sp_points_c
                self.sp_deform_net.load_state_dict(self.canonical_net.state_dict())
                logging.info(f"Replace canonical_net at step {self._step}")

    @torch.no_grad()
    def hook_after_train_step(self):
        if self._step == self.stages['sp_fix'][0]:
            self.sp_points.data[:, :3] = self.points
            super().create_from_pcd(self._task._pcd, self.lr_spatial_scale)  # noqa
            self.hyper_feature = nn.Parameter(
                torch.full([self._xyz.shape[0], self.hyper_dim], -1e-2, device=self.sp_points.device))
            self.to(self.sp_points.device)
            new_params = {v: getattr(self, k) for k, v in self.param_names_map.items()}
            if self.sp_W is not None:
                _, p2sp = my_ext.cdist_top(self.points, self.joints)
                scale = math.log(9 * (self.num_knn - 1))
                new_params['sp_W'] = F.one_hot(p2sp, self.num_superpoints).float() * scale  # [0.9, 0.1/(K-1), ...]
            new_params = self.change_optimizer(self._task.optimizer, new_params, op='replace')
            for param_name, opt_name in self.param_names_map.items():
                setattr(self, param_name, new_params[opt_name])
            self.active_sh_degree = 0
            self.training_setup()
            logging.info('Finish control points initialization')
            self._task.save_model('init.pth')
        elif self.sk_re_init_gs and self._step == self.stages['sk_fix'][0]:
            self.create_from_pcd(self._task._pcd, self.lr_spatial_scale)  # noqa
            self.to(self.joints.device)
            new_params = {v: getattr(self, k) for k, v in self.param_names_map.items()}
            M = self.joints.shape[0]
            scene_range = self.points.max() - self.points.min()
            if self.sp_W is not None:
                _, p2sp = my_ext.cdist_top(self.points, self.joints)
                scale = math.log(9 * (self.num_knn - 1))
                new_params['sp_W'] = F.one_hot(p2sp, M).float() * scale  # [0.9, 0.1/(K-1), ...]
            if self._sp_radius is not None:
                self._sp_radius.data = torch.log(.1 * scene_range + 1e-7) * scene_range.new_ones([M])
            if self._sp_weight is not None:
                self._sp_weight.data = torch.zeros_like(self._sp_radius)
            new_params = self.change_optimizer(self._task.optimizer, new_params, op='replace')
            for param_name, opt_name in self.param_names_map.items():
                setattr(self, param_name, new_params[opt_name])
            self.training_setup()
            logging.info('Finish re-initialize of gaussians')

    def prune_points(self, optimizer: torch.optim.Optimizer, mask):
        super().prune_points(optimizer, mask)
        if self.p2sp is not None:
            self.p2sp = self.p2sp[~mask]

    def densification_postfix(self, optimizer, mask=None, N=None, **kwargs):
        super().densification_postfix(optimizer, **kwargs, N=N, mask=mask)
        if self.p2sp is not None:
            if N is None:
                self.p2sp = torch.cat([self.p2sp, self.p2sp[mask]], dim=0)
            else:
                self.p2sp = torch.cat([self.p2sp, self.p2sp[mask].repeat(N)], dim=0)

    def adaptive_control_init_stage(self, inputs, outputs, optimizer, step: int):
        if step < self.init_sampling_step:
            radii = outputs['radii']
            if radii.ndim == 2:
                radii = radii.amax(dim=0)
            mask = radii > 0
            self.add_densification_stats(outputs['viewspace_points'], mask)
            # if step % self.adaptive_control_cfg['densify_interval'][0] == 0 or step == self.stages['init_fix'][1] - 1:
            if utils.check_interval_v2(step, *self.adaptive_control_cfg['init_densify_prune_interval']):
                size_threshold = 20 if step > self.adaptive_control_cfg['opacity_reset_interval'][0] else None
                if self.sp_deform_net.is_blender:
                    grad_max = self.adaptive_control_cfg['densify_grad_threshold']
                else:
                    if self.points.shape[0] > self.num_superpoints * self.node_max_num_ratio_during_init:
                        grad_max = torch.inf
                    else:
                        grad_max = self.adaptive_control_cfg['densify_grad_threshold']
                self.densify(optimizer, grad_max, self.cameras_extent,
                             self.adaptive_control_cfg['densify_percent_dense'])
                self.prune(optimizer, self.adaptive_control_cfg['prune_opacity_threshold'],
                           self.cameras_extent, size_threshold, self.adaptive_control_cfg['prune_percent_dense'])
                logging.info(f'Node after densify and prune, there are {len(self.points)} points at step {step}')
            # if (step >= 100 and (step - 1) % self.adaptive_control_cfg['opacity_reset_interval'][0] == 0) or \
            #     (self.background_type == 'white' and step == self.adaptive_control_cfg['densify_interval'][1]):
            if utils.check_interval_v2(step, *self.adaptive_control_cfg['init_opacity_reset_interval']):
                self.reset_opacity(optimizer)
                logging.info(f'reset opacity at init step {step}')
        elif step == self.init_sampling_step:
            self.init_superpoints(True, True)
            optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def adaptive_control(self, inputs, outputs, optimizer, step: int):
        step = step + 1
        if step <= self.stages['sp_fix'][0]:
            self.adaptive_control_init_stage(inputs, outputs, optimizer, step)
            return
        if step <= self.stages['sk_init'][0]:
            step = step - self.stages['sp_fix'][0]
        elif step > self.stages['sk_fix'][0]:
            if not self.sk_densify_gs:
                return
            step = step - self.stages['sk_fix'][0]
        else:
            return
        if step >= self.adaptive_control_cfg['densify_interval'][2]:
            return
        radii = outputs['radii']
        viewspace_point_tensor = outputs['viewspace_points']
        if radii.ndim == 2:
            radii = radii.amax(dim=0)
        mask = radii > 0
        # Keep track of max radii in image-space for pruning
        self.max_radii2D[mask] = torch.max(self.max_radii2D[mask], radii[mask])
        self.add_densification_stats(viewspace_point_tensor, mask)

        is_sp_stage = self.stages['sp'][0] < self._step <= self.stages['sp'][1]
        if is_sp_stage and utils.check_interval_v2(step, *self.adaptive_control_cfg['sp_adjust_interval'], close='[)'):
            self.superpoint_prune_split(optimizer)
            # logging.info(f"superpoint_prune_split {self.num_superpoints} at step {step}")
        if is_sp_stage and utils.check_interval_v2(step, *self.adaptive_control_cfg['sp_merge_interval'], close='[)'):
            self.superpoint_merge(optimizer)
        # node_interval = self.adaptive_control_cfg['node_densify_interval']
        # if utils.check_interval_v2(step, *node_interval, close='()') and \
        #     step > self.stages['sp'][0] or step == self.adaptive_control_cfg['node_force_densify_prune_step']:
        #     self.node_prune_split(optimizer)
        #     logging.info(f"superpoint_prune_split {self.num_superpoints} at step {step}")

        if utils.check_interval_v2(step, *self.adaptive_control_cfg['densify_interval'], close='()'):
            num0 = len(self.points)
            self.densify(
                optimizer,
                self.adaptive_control_cfg['densify_grad_threshold'],
                self.cameras_extent,
                self.adaptive_control_cfg['densify_percent_dense'],
            )
            num1 = len(self.points)
            if step > self.adaptive_control_cfg['opacity_reset_interval'][1]:
                size_threshold = self.adaptive_control_cfg['prune_max_screen_size']
            else:
                size_threshold = None
            self.prune(
                optimizer,
                self.adaptive_control_cfg['prune_opacity_threshold'],
                self.cameras_extent,
                size_threshold,
                self.adaptive_control_cfg['prune_percent_dense'],
            )
            num2 = len(self.points)
            logging.info(f'densify & prune: {num2} (+{num1 - num0}, -{num1 - num2}) points at step {step}')
        if (step > 1 and (step - 1) % self.adaptive_control_cfg['opacity_reset_interval'][0] == 0) or (
            self.background_type == 'white' and step == self.adaptive_control_cfg['densify_interval'][1]):
            self.reset_opacity(optimizer)
            logging.info(f'reset opacity at step {step}')


def test_skeleton_warp():
    from my_ext.utils.test_utils import get_run_speed
    print()
    M = 512
    joint_cost = torch.randn(M, M).cuda()
    parents, depth, root = joint_discovery(joint_cost)
    # print(parents[:, 0], root)
    sk_T = SE3.exp(torch.randn(M, 3).cuda()).matrix()
    sp_T = SE3.exp(torch.zeros(M, 3).cuda()).matrix()

    T1 = skeleton_warp_v0(sk_T, sp_T, parents, root)
    T2 = skeleton_warp(sk_T, sp_T, parents, root)
    error = ((T1 - T2).abs().view(M, -1).max())
    assert error < 1e-5

    get_run_speed((sk_T, sp_T, parents, root), None, skeleton_warp_v0, skeleton_warp)
