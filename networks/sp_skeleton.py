import logging
import math
import random
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import my_ext as ext
from networks.encoders import POSITION_ENCODERS
from networks.gaussian_splatting import GaussianSplatting, NERF_NETWORKS
from my_ext import get_C_function, utils, ops_3d
from my_ext.blocks import MLP_with_skips
from my_ext.ops_3d.lietorch import SE3, SO3


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
@ext.try_use_C_extension
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


class SkeletonDeformationNetwork(nn.Module):
    def __init__(
        self,
        out_channels,
        width=256,
        depth=8,
        skips=(4,),
        pos_enc_t='freq',
        pos_enc_t_cfg: dict = None
    ):
        super().__init__()
        self.pos_enc_t = POSITION_ENCODERS[pos_enc_t](**utils.merge_dict(pos_enc_t_cfg, input_dim=1))
        self.dynamic_net = MLP_with_skips(
            in_channels=self.pos_enc_t.output_dim,
            dim_hidden=width,
            out_channels=out_channels,
            num_layers=depth,
            skips=skips,
        )

    def forward(self, t: Tensor):
        t_embed = self.pos_enc_t(t.view(-1, 1))
        out = self.dynamic_net(t_embed)
        return out


@NERF_NETWORKS.register('SK_GS')
class SuperpointSkeletonGaussianSplatting(GaussianSplatting):
    ## Gaussians
    gs_knn_index: Tensor
    gs_knn_dist: Tensor
    train_db_times: Tensor
    ## Superpoints
    sp_is_init: Tensor
    sp_delta_tr: Tensor
    sp_weights: Tensor
    sp_knn: Tensor
    p2sp: Optional[Tensor]
    ## Skeleton
    joint_is_init: Tensor
    sk_is_init: Tensor
    joint_cost: Tensor
    joint_parents: Tensor
    joint_depth: Tensor
    joint_root: Tensor

    skeleton_r: Tensor
    """The rotation matrix of skeleton, shape: [T, B, 3]"""

    def __init__(
        self,
        ## Gaussians
        learn_gaussian=True,
        gs_knn_num=20,
        gs_knn_update_interval=(1000, 3000),
        canonical_time_id: int = -1,
        ## Superpoint
        sp_deform_net_cfg: dict = None,
        num_superpoints=1000,
        sp_knn_num=5,
        sp_prune_threshold=1e-3,
        sp_split_threshold=0.0002,
        sp_merge_threshold=0.01,
        sp_lbs=True,
        test_time_interpolate=False,
        ## Skeleton
        sk_use_features=True,
        sk_feature_dim=0,
        sk_deform_net_cfg: dict = None,
        sk_knn_num=10,
        sk_tr_batch_size=-1,
        joint_update_interval=(1000, 10_000),
        joint_init_steps=1000,
        momentum=0.9,
        ## other
        train_schedule=((0, 'static'), (3000, 'sp'), (30_000, 'skeleton')),
        ## loss
        sp_guided_steps_range=(0, 10_000),
        sp_guided_detach=True,
        loss_arap_start_step=-1,
        adaptive_control_cfg=None,
        **kwargs
    ):
        adaptive_control_cfg = utils.merge_dict(
            adaptive_control_cfg,
            sp_adjust_interval=[5000, 5000, 25000],
            sp_merge_interval=[-1, 10_000, 20_000]
        )
        super().__init__(**kwargs, adaptive_control_cfg=adaptive_control_cfg)
        self.test_time_interpolate = test_time_interpolate
        self.train_schedule = train_schedule
        assert all(stage in ['static', 'sp', 'skeleton', 'canonical'] for steps, stage in train_schedule)
        ## 3D Gaussians
        self._Rt_dim = 6
        self.which_rotation = 'so3'
        self.learn_gaussian = learn_gaussian
        self.gs_knn_update_interval = gs_knn_update_interval
        self.gs_knn_num = gs_knn_num
        self.register_buffer('gs_knn_index', torch.empty(0, self.gs_knn_num, dtype=torch.long), persistent=False)
        self.register_buffer('gs_knn_dist', torch.empty(0, self.gs_knn_num, dtype=torch.float), persistent=False)

        self.num_frames = 0
        self.register_buffer('train_db_times', torch.zeros(self.num_frames))

        self.canonical_time_id = canonical_time_id
        self._canonical_aux = None
        ## superpoints
        self.num_superpoints = num_superpoints
        self.sp_points = nn.Parameter(torch.zeros(self.num_superpoints, 3))  # [M, 3]
        self.sp_deform_net = SimpleDeformationNetwork(**utils.merge_dict(sp_deform_net_cfg, out_channels=6))
        self.sp_W = nn.Parameter(torch.empty(0, self.num_superpoints))  # [N, M]
        self.param_names_map['sp_W'] = 'sp_W'
        self.sp_knn_num = sp_knn_num
        self.sp_no_lbs = not sp_lbs
        self.register_buffer('sp_weights', torch.empty(0, sp_knn_num))  # [N, K]
        self.register_buffer('sp_knn', torch.empty(0, sp_knn_num, dtype=torch.long))  # [N, K]
        self.register_buffer('sp_delta_tr', torch.empty(self.num_frames, self.num_superpoints, 6))
        self.register_buffer('sp_is_init', torch.tensor(False, dtype=torch.bool))
        if self.sp_no_lbs:
            self.register_buffer('p2sp', torch.empty(0, self.num_superpoints))
        else:
            self.p2sp = None
        self.sp_prune_threshold = sp_prune_threshold
        self.sp_split_threshold = sp_split_threshold
        self.sp_merge_threshold = sp_merge_threshold
        self._sp_prune_split_temp_tensors = tuple()
        ## skeleton
        M = self.num_superpoints
        self.sk_knn_num = sk_knn_num
        self.joint_update_interval = joint_update_interval
        self.joint_init_steps = joint_init_steps
        self.momentum = momentum
        self.register_buffer('sk_is_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('joint_is_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('sk_W_is_init', torch.tensor(False, dtype=torch.bool))

        self.joint_pos = nn.Parameter(torch.empty(M, M, 3))
        self.sk_W = nn.Parameter(torch.empty(0, self.num_superpoints))  # [N, M]
        self.param_names_map['sk_W'] = 'sk_W'
        self.sk_tr_batch_size = sk_tr_batch_size
        if self.sk_tr_batch_size >= 0:
            self.joint_rot = nn.Parameter(torch.empty(self.num_frames, M, M, 3))
        # self.LBS_alpha = nn.Parameter(torch.tensor(1.))
        self.sk_deform_net_cfg = sk_deform_net_cfg
        self.sk_use_features = sk_use_features
        self.sk_feature_dim = sk_feature_dim
        self.sk_feature = nn.Parameter(torch.randn(M, sk_feature_dim)) if sk_feature_dim > 0 else None
        if sk_use_features:
            self.sk_deform_net = SimpleDeformationNetwork(
                **utils.merge_dict(sk_deform_net_cfg, out_channels=3, p_in_channels=3 + self.sk_feature_dim)
            )
        else:
            self.sk_deform_net = SkeletonDeformationNetwork(
                **utils.merge_dict(sk_deform_net_cfg, out_channels=3 * M)
            )
        self.register_buffer('joint_cost', torch.zeros(M, M))
        self.register_buffer('joint_parents', torch.full((M, 1), -1, dtype=torch.int32))
        self.register_buffer('joint_depth', torch.zeros(M, dtype=torch.int32))
        self.register_buffer('joint_root', torch.arange(M, dtype=torch.int32))
        self.register_buffer('skeleton_r', torch.empty((self.num_frames, M, 3)))
        self._joint_pair = None
        self.global_tr = nn.Parameter(torch.zeros(self.num_frames, 6))
        ## loss
        self.sp_guided_steps_range = sp_guided_steps_range
        self.sp_guided_detach = sp_guided_detach
        self.loss_arap_start_step = loss_arap_start_step
        # self.timer = utils.TimeWatcher(False)
        self.reset_parameters()

    def reset_parameters(self, ):
        # nn.init.normal_(self.sk_deform_net.dynamic_net.last.weight, 0, 1e-5)
        nn.init.zeros_(self.sk_deform_net.dynamic_net.last.weight)
        if self.sk_deform_net.dynamic_net.last.bias is not None:
            nn.init.constant_(self.sk_deform_net.dynamic_net.last.bias, 0)
        # nn.init.constant_(self.sp_deform_net.last.weight, 0)
        nn.init.normal_(self.sp_deform_net.dynamic_net.last.weight, 0, 1e-5)
        if self.sp_deform_net.dynamic_net.last.bias is not None:
            nn.init.constant_(self.sp_deform_net.dynamic_net.last.bias, 0)

    def get_params(self, cfg, include_deform=True):  # noqa
        if self.learn_gaussian:
            params_groups = super().get_params(cfg)
        else:
            params_groups = []
        lr = cfg.lr
        params_groups.extend([
            {'params': self.sp_deform_net.parameters(), 'lr': lr, 'name': 'sp_deform'},
            {'params': [self.sp_W], 'lr': lr, 'name': 'sp_W'},
            {'params': [self.sp_points], 'lr': lr, 'name': 'sp_points'},
        ])
        params_groups.extend([
            {'params': [self.sk_W], 'lr': lr, 'name': 'sk_W'},
            {'params': [self.joint_pos], 'lr': lr, 'name': 'joint_pos'},
            # {'params': [self.LBS_alpha], 'lr': lr, 'name': 'LBS_alpha'},
            {'params': self.sk_deform_net.parameters(), 'lr': lr, 'name': 'sk_deform'},
            {'params': [self.global_tr], 'lr': lr * 0.1, 'name': 'global_tr'},
        ])
        if self.sk_tr_batch_size >= 0:
            params_groups.append({'params': [self.joint_rot], 'lr': lr, 'name': 'joint_rot'})
        if self.sk_feature_dim > 0:
            params_groups.append({'params': [self.sk_feature], 'lr': lr, 'name': 'sk_feature'})
        return params_groups

    def set_from_dataset(self, dataset):
        super().set_from_dataset(dataset)
        self.num_frames = dataset.num_frames  # the number of frames
        M = self.num_superpoints
        self.register_buffer('sp_delta_tr', torch.zeros([self.num_frames, M, self._Rt_dim]))
        self.train_db_times = dataset.times[dataset.camera_ids == dataset.camera_ids[0]]
        assert self.num_frames == len(self.train_db_times)
        self.register_buffer('skeleton_r', torch.zeros(self.num_frames, M, 3))
        # self.skeleton_r = nn.Parameter(torch.zeros(self.num_frames, M , 3))
        if self.sk_tr_batch_size >= 0:
            self.joint_rot = nn.Parameter(torch.zeros(self.num_frames, M, M, 3))
        self.global_tr = nn.Parameter(torch.zeros(self.num_frames, 6))
        assert self.canonical_time_id < self.num_frames

    def create_from_pcd(self, pcd, lr_spatial_scale: float = None):
        super().create_from_pcd(pcd, lr_spatial_scale)
        self.sp_W = nn.Parameter(torch.empty(len(self._xyz), self.num_superpoints))
        self.sk_W = nn.Parameter(torch.empty(len(self._xyz), self.num_superpoints))

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs):
        N = state_dict['_xyz'].shape[0]
        M = self.num_superpoints = state_dict['sp_points'].shape[0]
        self.num_frames = state_dict['sp_delta_tr'].shape[0]
        self.sp_no_lbs = 'p2sp' in state_dict
        if M != self.joint_pos.shape[0]:
            self.joint_pos = nn.Parameter(torch.zeros(M, M, 3))
            if self.sk_tr_batch_size >= 0:
                self.joint_rot = nn.Parameter(torch.zeros(self.num_frames, M, M, 3))
            self.joint_cost = torch.zeros(M, M)

            self.joint_depth = torch.zeros(M, dtype=torch.int32)
            self.joint_root = torch.arange(M, dtype=torch.int32)

            self.skeleton_r = torch.zeros(self.num_frames, M, 3)
            if not self.sk_use_features:
                self.sk_deform_net = SkeletonDeformationNetwork(
                    **utils.merge_dict(self.sk_deform_net_cfg, out_channels=3 * M)
                )
        for name in [
            'xyz_gradient_accum', 'denom', 'max_radii2D', 'train_db_times',
            'sp_grad_accum', 'sp_grad_denom',
            'sp_delta_tr', 'sp_weights', 'sp_knn', 'p2sp',
            'joint_parents', 'joint_root',
        ]:
            if name in state_dict:
                setattr(self, name, state_dict[name])
                logging.debug(f'change the shape of parameters of {name}')
        for name in ['sp_points', 'W', 'sk_feature', 'global_tr']:
            if name in state_dict:
                setattr(self, name, nn.Parameter(state_dict[name]))
                logging.debug(f'change the shape of Parameter of {name}')
        super().load_state_dict(state_dict, strict, **kwargs)
        assert self.canonical_time_id < self.num_frames

    def tr_to_R(self, tr: Tensor):
        return SE3.exp(tr).matrix()[..., :3, :3]

    def tr_to_T(self, tr: Tensor):
        return SE3.exp(tr).matrix()

    def T_to_tr(self, T: Tensor):
        return SE3.InitFromVec(ops_3d.rigid.Rt_to_quaternion(T)).data

    def update_gs_knn(self, force=False):
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

    @torch.no_grad()
    def init_superpoint(self, points: Tensor):
        if self.sp_is_init:
            return
        sp_idx = get_C_function('FurthestSampling')(points.contiguous()[None], self.num_superpoints)
        sp_idx = sp_idx.view(-1)  # shape: [M]
        self.sp_points.data.copy_(points[sp_idx])  # [M, 3]
        _, p2sp = ext.cdist_top(points, self.sp_points)
        scale = math.log(9 * (self.sp_knn_num - 1))
        self.sp_W.data.copy_(F.one_hot(p2sp, self.num_superpoints).float() * scale)  # [0.9, 0.1/(K-1), ...]
        self.sp_is_init = self.sp_is_init.new_tensor(True)
        if self.sp_no_lbs:
            self.p2sp = p2sp
        logging.info('init superpoint')

    def LBS_warp(self, points: Tensor, rotations: Tensor, sp_tr, knn: Tensor, weights: Tensor):
        if self.sp_no_lbs:
            delta_tr = sp_tr[self.p2sp]
            delta_T = SE3.exp(delta_tr)  # [N, ?]
            points = delta_T.act(points)
        else:
            delta_tr = torch.sum(sp_tr[knn] * weights[..., None], dim=1)  # [N, 6]
            delta_T = SE3.exp(delta_tr)  # [N, ?]
            points = torch.sum(SE3.exp(sp_tr)[knn].act(points[:, None]) * weights[..., None], dim=1)
            # points = delta_T.act(points)
        rotations = (SO3.InitFromVec(delta_T.vec()[..., 3:]) * SO3.InitFromVec(rotations)).vec()
        return points, rotations, delta_tr

    def sp_stage(self, points: Tensor, rotations: Tensor, t: Tensor, time_id: Tensor = None, **kwargs):
        """ superpoint-based warp
        Args:
            points: shape [P, 3] [x, y, z]
            rotations: shape: [P, 4] quaternion [w, x, y, z]
            t: shape [1] currernt time
            time_id: shape: [1] the id of frame/time
        Returns:
           (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
               points: shape [P, 3]
               rotations: shape [P, 3]
               delta_tr: [M, 6] Rt for points
               sp_tr: [M, 6] Rt for superpoints
               weights: [N, K]
               knn: [N, K]
        """
        self.init_superpoint(points)

        if self.training or not self.test_time_interpolate:
            sp_tr = self.sp_deform_net(self.sp_points, t)
        elif time_id is None:
            time_id = torch.searchsorted(self.train_db_times, t).item()
            if time_id == len(self.train_db_times):
                t1_idx, t2_idx = len(self.train_db_times) - 2, len(self.train_db_times) - 1
            elif time_id == 0:
                t1_idx, t2_idx = time_id, time_id + 1
            else:
                t1_idx, t2_idx = time_id - 1, time_id
            t1, t2 = self.train_db_times[t1_idx], self.train_db_times[t2_idx]
            w = (t - t1) / (t2 - t1)
            sp_tr = torch.lerp(self.sp_delta_tr[t1_idx], self.sp_delta_tr[t2_idx], w)
        else:
            sp_tr = self.sp_delta_tr[time_id]  # [M, 3+?]

        with torch.no_grad():
            # if self.canonical_time_id >= 0:
            #     sp_tr_c = self.sp_deform_net(self.sp_points, self.train_db_times[self.canonical_time_id])
            #     sp_points = SE3.exp(sp_tr_c).act(self.sp_points)
            # else:
            sp_points = self.sp_points
            dist = torch.cdist(points, sp_points)  # [N, M]
            knn_dist, knn_index = torch.topk(dist, k=min(self.num_superpoints, self.sp_knn_num), dim=-1, largest=False)
        weights = torch.gather(self.sp_W, dim=1, index=knn_index).softmax(dim=-1)
        if self.training and self.sp_no_lbs:
            self.p2sp = torch.gather(knn_index, -1, weights.argmax(dim=-1, keepdim=True))[:, 0]  # [N]
        points, rotations, delta_tr = self.LBS_warp(points, rotations, sp_tr, knn_index, weights)
        self.sp_weights = weights.detach()
        self.sp_knn = knn_index.detach()
        return points, rotations, delta_tr, sp_tr, weights, knn_index

    def canonical_stage(self, time_id):
        if self._canonical_aux is None:
            with torch.no_grad():
                # cache old
                for i in range(len(self.train_db_times)):
                    self.sp_delta_tr[i] = self.sp_deform_net(self.sp_points, self.train_db_times[i].cuda().view(-1, 1))
                dist = torch.cdist(self.points, self.sp_points)  # [N, M]
                knn_dist, knn_index = torch.topk(dist, k=self.sp_knn_num, dim=-1, largest=False)
                weights = torch.gather(self.sp_W, dim=1, index=knn_index).softmax(dim=-1)
                p2sp = self.p2sp.clone() if self.sp_no_lbs else None
                self._canonical_aux = (self.points.clone(), self.get_rotation.clone(), weights, knn_index, p2sp)
                # assign
                t = self.train_db_times[self.canonical_time_id]
                points, rotations, _, sp_tr, _, _ = self.sp_stage(self.points, self.get_rotation, t)
                self._xyz.data.copy_(points)
                self._rotation.data.copy_(rotations)
                sp_points = SE3.exp(sp_tr).act(self.sp_points)
                self.sp_points.data.copy_(sp_points)
                # new parameters
                # self.sp_deform_net.dynamic_net.last.weight.data.zero_()
                nn.init.normal_(self.sp_deform_net.dynamic_net.last.weight.data, 0, 1e-4)
                nn.init.zeros_(self.sp_deform_net.dynamic_net.last.bias.data)
                self.sp_is_init.zero_()
                self.init_superpoint(points)
            logging.info(f'make time[{self.canonical_time_id}] {t} as canonical')
        sp_tr = self.sp_delta_tr[time_id]  # [M, 3+?]
        points, rotations, weights, knn_index, p2sp = self._canonical_aux
        if self.sp_no_lbs:
            delta_tr = sp_tr[p2sp]
            delta_T = SE3.exp(delta_tr)  # [N, ?]
            points = delta_T.act(points)
        else:
            delta_tr = torch.sum(sp_tr[knn_index] * weights[..., None], dim=1)  # [N, 6]
            delta_T = SE3.exp(delta_tr)  # [N, ?]
            points = torch.sum(SE3.exp(sp_tr)[knn_index].act(points[:, None]) * weights[..., None], dim=1)
            # points = delta_T.act(points)
        rotations = (SO3.InitFromVec(delta_T.vec()[..., 3:]) * SO3.InitFromVec(rotations)).vec()
        return points, rotations, delta_tr, sp_tr

    @torch.no_grad()
    def init_joint_pos(self, force=False, use_gs=False):
        if self.joint_is_init and not force:
            return
        logging.info('init joint_pos')
        self.joint_is_init = self.joint_is_init.new_tensor(True)
        if self.canonical_time_id >= 0:
            sp_tr_c = self.sp_deform_net(self.sp_points, self.train_db_times[self.canonical_time_id])
            sp_tr_c = SE3.exp(sp_tr_c)
            sp_points = sp_tr_c.act(self.sp_points)
        else:
            sp_tr_c = None
            sp_points = self.sp_points
        self.joint_pos.data.copy_((sp_points[:, None] + sp_points[None, :]) * 0.5)
        if not use_gs:
            return
        W = torch.scatter(torch.zeros_like(self.sp_W), 1, self.sp_knn, self.sp_weights)
        sp_se3 = SE3.exp(self.sp_delta_tr)
        for a in range(self.num_superpoints):
            for b in range(self.num_superpoints):
                w = W[:, a] + W[:, b]
                mask = w.ge(0.5)
                if mask.sum() == 0:
                    continue
                points = self.points[mask]
                p_a = sp_se3[:, a:a + 1].act(points[None])
                p_b = sp_se3[:, b:b + 1].act(points[None])
                dist = (p_a - p_b).norm(dim=-1).mean(dim=0)
                if self.canonical_time_id >= 0:
                    p = points[torch.argmin(dist)]
                    self.joint_pos[a, b] = (sp_tr_c[a].act(p) + sp_tr_c[b].act(p)) * 0.5
                else:
                    self.joint_pos[a, b] = points[torch.argmin(dist)]
        return

    def init_joint(self, progress_func=None):
        if self.joint_is_init:
            return
        self.joint_is_init = self.joint_is_init.new_tensor(True)
        logging.info('Begin init joint')
        self.init_joint_pos(True)

        sp_R = SO3.exp(self.sp_delta_tr[..., 3:])  # [T, M]
        joint_rot = sp_R.inv()[:, None] * sp_R[:, :, None]  # [T, M, M], R_b^-1 R_a
        if self.canonical_time_id >= 0:
            sp_se3_c = SE3.exp(self.sp_delta_tr[self.canonical_time_id])
            sp_se3_c_inv = sp_se3_c.inv()
            sp_so3_c = SO3.InitFromVec(sp_se3_c.vec()[..., 3:])
            sp_so3_c_inv = sp_so3_c.inv()

            joint_rot = sp_so3_c[None, None] * joint_rot * sp_so3_c_inv[None, None]
            sp_delta_tr = SE3.exp(self.sp_delta_tr) * sp_se3_c_inv[None]
            sp_points = self.sp_points.clone()
            self.sp_points.data.copy_(sp_se3_c.act(sp_points))
        else:
            sp_points = self.sp_points
            sp_delta_tr = self.sp_delta_tr
        optimizer = torch.optim.Adam([{
            'params': [self.joint_pos],  # + list(self.sk_deform_net.parameters()),
            'lr': 1.0e-3
        }], lr=1.0e-3)
        loss_meter = ext.DictMeter()
        with torch.enable_grad():
            for i in range(self.joint_init_steps):
                tid = random.randrange(self.num_frames)
                joint_loss, all_loss = self.find_joint_loss(sp_delta_tr[tid], tid, True, joint_rot)
                loss_dict = {'best': joint_loss, 'all': all_loss}

                # loss_dict['bone'] = self.loss_bone(self.sp_delta_tr[tid])
                # t = self.train_db_times[tid]
                # a, b, mask = self.joint_pair
                # joints = sp_xyz.clone().detach()
                # joints[mask] = self.joint_pos[a, b]
                # g_tr = self.sp_delta_tr[tid, self.joint_root]
                # if self.canonical_time_id >= 0:
                #     g_tr = SE3.exp(g_tr) * sp_se3_c_inv[self.joint_root]
                # sk_tr = self.kinematic(joints, t, g_tr, tid)
                # if self.canonical_time_id >= 0:
                #     sp_tr = SE3.exp(self.sp_delta_tr[tid])
                #     sk_tr = SE3.exp(sk_tr) * sp_se3_c_inv
                #     loss_dict['guided'] = (sk_tr.inv() * sp_tr).log().norm(-1).mean()
                # else:
                #     loss_dict['guided'] = F.mse_loss(sk_tr, self.sp_delta_tr[tid])
                # break
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
        if self.canonical_time_id >= 0:
            self.sp_points.data.copy_(sp_points)
        logging.info('Init joint')
        return

    @torch.enable_grad()
    def init_sk_deform(self, progress_func=None):
        with torch.no_grad():
            # dist = torch.cdist(self.points, self.sp_points)
            # knn = torch.topk(dist, k=min(self.num_superpoints, self.sp_knn_num), dim=-1, largest=False)[1]
            knn = self.sp_knn
            points, rotations = self._canonical_aux

            nn.init.zeros_(self.sk_deform_net.dynamic_net.last.bias)
            nn.init.normal_(self.sk_deform_net.dynamic_net.last.weight, 0, 1e-5)
            scale = math.log(9 * (self.sp_knn_num - 1))
            self.sk_W.data.copy_(F.one_hot(knn[:, 0], self.num_superpoints).float() * scale)
            points_c = self.points
            rotations_c = self._rotation
        optimizer = torch.optim.Adam(
            list(self.sk_deform_net.parameters()) + [self.sk_W, self.global_tr, self.sp_points], lr=1e-3,
        )
        state = self.training
        self.train()
        loss_meter = ext.DictMeter(float2str=ext.utils.str_utils.float2str)
        for step in range(self.joint_init_steps):
            tid = random.randrange(self.num_frames)
            t = self.train_db_times[tid]
            with torch.no_grad():
                p1, r1 = self.LBS_warp(points, rotations, self.sp_delta_tr[tid], self.sp_knn, self.sp_weights)[:2]

            sk_tr = self.kinematic(self.sp_points, t, self.global_tr[tid], tid)
            weights = torch.gather(self.sk_W, dim=1, index=knn).softmax(dim=-1)
            p2, r2 = self.LBS_warp(points_c, rotations_c, sk_tr, knn, weights)[:2]

            cmp_p = F.mse_loss(p1, p2)
            cmp_r = 0  # (SO3.InitFromVec(r1).inv() * SO3.InitFromVec(r2)).log().norm(dim=-1).mean()
            loss_meter.update({'cmp_p': cmp_p, 'cmp_r': cmp_r})
            loss = cmp_p + cmp_r
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(True)
            if step % 100 == 0:
                logging.info(f"step[{step}]: loss: {loss_meter.average}")
                loss_meter.reset()
            if progress_func is not None:
                progress_func()
        with torch.no_grad():
            losses = 0
            for tid in range(self.num_frames):
                t = self.train_db_times[tid]
                p1, r1 = self.LBS_warp(points, rotations, self.sp_delta_tr[tid], self.sp_knn, self.sp_weights)[:2]

                sk_tr = self.kinematic(self.sp_points, t, self.global_tr[tid], tid)
                weights = torch.gather(self.sk_W, dim=1, index=knn).softmax(dim=-1)
                p2, r2 = self.LBS_warp(points_c, rotations_c, sk_tr, knn, weights)[:2]
                losses = losses + F.mse_loss(p1, p2)
            logging.info(f'final loss: {losses.item() / self.num_frames:.6f}')
        # self.sp_knn = knn
        self.sk_W_is_init = self.sk_W_is_init.new_tensor(True)
        self.train(state)

    @torch.no_grad()
    def init_skeleton(self, progress_func=None):
        if self.sk_is_init:
            return
        ## cache the data from sp stage
        for i in range(len(self.train_db_times)):
            self.sp_delta_tr[i] = self.sp_deform_net(self.sp_points, self.train_db_times[i].cuda().view(-1, 1))
        dist = torch.cdist(self.points, self.sp_points)  # [N, M]
        _, self.sp_knn = torch.topk(dist, k=min(self.num_superpoints, self.sp_knn_num), dim=-1, largest=False)
        self.sp_weights = torch.gather(self.sp_W, dim=1, index=self.sp_knn).softmax(dim=-1)
        self._canonical_aux = (self.points.clone(), self.get_rotation.clone())
        ## init joint
        self.init_joint(progress_func)
        self.update_joint()
        ## make self.canonical_time as canonical
        if self.canonical_time_id < 0:
            self.global_tr.data.copy_(self.sp_delta_tr[:, self.joint_root])
            self.sk_is_init = self.sk_is_init.new_tensor(True)
            logging.info('Init skeleton')
            return
        t = self.train_db_times[self.canonical_time_id]
        points, rotations, _, sp_tr, _, _ = self.sp_stage(self.points, self.get_rotation, t)
        self._xyz.data.copy_(points)
        self._rotation.data.copy_(self.rotation_activation(rotations))
        sp_tr_c = SE3.exp(sp_tr)
        root_tr_inv = sp_tr_c[self.joint_root].inv()
        for i in range(self.num_frames):
            self.global_tr[i] = (SE3.exp(self.sp_delta_tr[i, self.joint_root]) * root_tr_inv).log()
        ## assign W
        a, b, mask = self.joint_pair
        self.sp_points[self.joint_root] = sp_tr_c[self.joint_root].act(self.sp_points[self.joint_root])
        self.sp_points[mask] = self.joint_pos[a, b]
        # # if self.sp_no_lbs:
        # # nearest = torch.cdist(self.points, self.sp_points).argmin(dim=1)
        # # scale = math.log(9 * (self.num_superpoints - 1))
        # # self.W.data.copy_(F.one_hot(nearest, self.num_superpoints).float() * scale)  # [0.9, 0.1/(K-1), ...]
        # # else:
        # #     W = torch.full_like(self.W, -10)
        # #     W = torch.scatter(W, 1, self.sp_knn, torch.gather(self.W, 1, self.sp_knn))
        # #     self.W.data.copy_(W)
        # # self.sp_points.data.copy_(sp_tr_c.act(self.sp_points))
        # # self.sk_deform_net.dynamic_net.last.weight.data.zero_()
        # # self.sk_deform_net.dynamic_net.last.bias.data.zero_()
        # if not self.sk_W_is_init:
        #     self.sk_W.data.copy_(self.sp_W)
        #     self.sk_W_is_init = self.sk_W_is_init.new_tensor(True)
        self.init_sk_deform(progress_func)
        self.sk_is_init = self.sk_is_init.new_tensor(True)
        logging.info('Init skeleton')

    def kinematic(
        self,
        joint: Tensor,
        t: Tensor,
        g_tr: Tensor = None,
        time_id: int = None,
        sk_r_delta: Tensor = None,
        return_se3=True,
    ):
        """运动学模型"""
        ## joint rotation
        if self.training or not self.test_time_interpolate:
            if self.sk_use_features:
                x_input = joint if self.sk_feature is None else torch.cat([joint, self.sk_feautre], dim=1)
                sk_r = self.sk_deform_net(x_input, t).view(self.num_superpoints, 3)
            else:
                sk_r = self.sk_deform_net(t).view(self.num_superpoints, 3)

            if self.training and time_id is not None:
                self.skeleton_r[time_id] = sk_r
        elif time_id is None:
            time_id = torch.searchsorted(self.train_db_times, t).item()
            if time_id == len(self.train_db_times):
                t1_idx, t2_idx = len(self.train_db_times) - 2, len(self.train_db_times) - 1
            elif time_id == 0:
                t1_idx, t2_idx = time_id, time_id + 1
            else:
                t1_idx, t2_idx = time_id - 1, time_id
            t1, t2 = self.train_db_times[t1_idx], self.train_db_times[t2_idx]
            w = (t - t1) / (t2 - t1)
            sk_r = torch.lerp(self.skeleton_r[t1_idx], self.skeleton_r[t2_idx], w).view(-1)
        else:
            sk_r = self.skeleton_r[time_id]
        sk_r = SO3.exp(sk_r)
        if sk_r_delta is not None:
            sk_r = (SO3.exp(sk_r_delta) if sk_r_delta.shape[-1] == 3 else SO3.InitFromVec(sk_r_delta)) * sk_r
        ## skeleton transforms
        sk_t = joint + sk_r.act(-joint)
        sk_tr = SE3.InitFromVec(torch.cat([sk_t, sk_r.vec()], dim=-1))
        if g_tr is None:
            g_tr = None
        elif isinstance(g_tr, SE3):
            g_tr = g_tr.matrix()
        elif g_tr.shape == (6,):
            g_tr = SE3.exp(g_tr).matrix()
        elif g_tr.shape == (7,):
            g_tr = SE3.InitFromVec(g_tr).matrix()
        elif g_tr.shape == (4, 4):
            g_tr = g_tr
        else:
            raise ValueError(f'g_tr got shape {g_tr.shape}')
        sk_T = skeleton_warp(sk_tr.matrix(), g_tr, self.joint_parents, self.joint_root)
        # sk_T = skeleton_warp_v0(sk_tr.matrix(), g_tr, self.joint_parents, self.joint_root)
        sk_tr = SE3.InitFromVec(ops_3d.rigid.Rt_to_quaternion(sk_T))
        return sk_tr.log() if return_se3 else sk_tr

    def skeleton_stage(
        self, points: Tensor, rotations: Tensor, t: Tensor, time_id: Tensor = None, sk_r_delta: Tensor = None, **kwargs
    ):
        # self.timer.start()
        # if self.training:
        self.init_skeleton()
        # weights = self.sp_weights
        # if time_id is None:
        #     tid = torch.searchsorted(self.train_db_times, t).item()
        #     if tid == len(self.train_db_times):
        #         t1_idx, t2_idx = len(self.train_db_times) - 2, len(self.train_db_times) - 1
        #     elif tid == 0:
        #         t1_idx, t2_idx = tid, tid + 1
        #     else:
        #         t1_idx, t2_idx = tid - 1, tid
        #     t1, t2 = self.train_db_times[t1_idx], self.train_db_times[t2_idx]
        #     w = (t - t1) / (t2 - t1)
        #     sk_tr = torch.lerp(self.sp_delta_tr[t1_idx], self.sp_delta_tr[t2_idx], w)
        # else:
        #     sk_tr = self.sp_delta_tr[time_id]  # [M, 3+?]
        # points_2, rotations_2, _ = self.LBS_warp(*self._canonical_aux, sk_tr, self.sp_knn, weights)

        # self.timer.log('init skeleton')
        ## global transform
        time_id = time_id.item() if isinstance(time_id, Tensor) else time_id
        if not self.sk_is_init:
            g_tr = self.sp_deform_net(self.sp_points, t)[self.joint_root]
            # sp_point_root = self.sp_points[self.joint_root][None]
            # g_tr_t = self.sp_deform_net(sp_point_root, t)[0]
            # g_tr_c = self.sp_deform_net(self.sp_points, self.train_db_times[self.canonical_time_id])
            # g_tr = SE3.exp(g_tr_t) * SE3.exp(g_tr_c[0]).inv()
            #
            # delta_tr = torch.sum(g_tr_c[self.sp_knn] * self.sp_weights[..., None], dim=1)  # [N, 6]
            # delta_T = SE3.exp(delta_tr)  # [N, ?]
            # points = delta_T.act(points)
            # rotations = (SO3.InitFromVec(delta_T.vec()[..., 3:]) * SO3.InitFromVec(rotations)).vec()
        elif time_id is None:
            time_id = torch.searchsorted(self.train_db_times, t).item()
            if time_id == len(self.train_db_times):
                t1_idx, t2_idx = len(self.train_db_times) - 2, len(self.train_db_times) - 1
            elif time_id == 0:
                t1_idx, t2_idx = time_id, time_id + 1
            else:
                t1_idx, t2_idx = time_id - 1, time_id
            t1, t2 = self.train_db_times[t1_idx], self.train_db_times[t2_idx]
            w = (t - t1) / (t2 - t1)
            g_tr = torch.lerp(self.global_tr[t1_idx], self.global_tr[t2_idx], w).view(-1)
        else:
            g_tr = self.global_tr[time_id].view(-1)
        # self.timer.log('sp')
        ## Kinematic transform
        if self.sk_is_init:
            joints = self.sp_points
        else:
            joints = self.sp_points.clone().detach()
            a, b, mask = self.joint_pair
            joints[mask] = self.joint_pos[a, b]
        sk_tr = self.kinematic(joints, t, g_tr, time_id, sk_r_delta)
        # self.timer.log('kinematic')
        ## Linear Blend Skinning
        # # w = (self.W / self.LBS_alpha).softmax(dim=1)  # [N, M]
        # if self.sk_is_init:
        #     w = self.W.softmax(dim=1)  # [N, M]
        # else:
        #     w = torch.scatter(torch.zeros_like(self.W), 1, self.sp_knn, self.sp_weights)
        # tr_c = w @ sk_tr  # [N, 7]
        # Tc = SE3.exp(tr_c)
        # points = torch.sum(SE3.exp(sk_tr[None, :]).act(points[:, None]) * w[..., None], dim=1)
        # # points = Tc.act(points)  # shape: [N, 3]
        # # rotations = (SO3.exp(tr_c[:, 3:]) * SO3.InitFromVec(rotations)).vec()
        # rotations = (SO3.InitFromVec(Tc.vec()[..., 3:]) * SO3.InitFromVec(rotations)).vec()
        # # self.timer.log('lbs')
        weights = torch.gather(self.sk_W, dim=1, index=self.sp_knn).softmax(dim=-1)
        points, rotations, tr_c = self.LBS_warp(points, rotations, sk_tr, self.sp_knn, weights)
        # print(F.mse_loss(points, points_2))
        return points, rotations, tr_c, sk_tr, g_tr, weights

    def forward(
        self,
        t: Tensor = None,
        scaling_modifier=1,
        campos: Tensor = None,
        time_id: int = None,
        stage=None,
        **kwargs
    ):
        outputs = {}
        stage = self.get_now_stage(stage)
        opacity = self.opacity_activation(self._opacity)
        sh_features = torch.cat((self._features_dc, self._features_rest), dim=1)
        t = t.view(-1, 1)
        if isinstance(time_id, Tensor):
            time_id = time_id.item()
        points, scales, rotations = self._xyz, self._scaling, self._rotation
        if stage == 'static':
            pass
        elif stage == 'sp':
            points, rotations, tr, sp_tr, sp_weights, sp_knn = self.sp_stage(points, rotations, t, time_id)
            outputs.update(_tr=tr, _sp_tr=sp_tr, _sp_w=sp_weights, _sp_knn=sp_knn)
        elif stage == 'skeleton':
            points, rotations, tr, sk_tr, g_tr, lbs_w = self.skeleton_stage(points, rotations, t, time_id, **kwargs)
            outputs.update(_tr=tr, _g_tr=g_tr, _sk_tr=sk_tr, _lbs_w=lbs_w)
        elif stage == 'canonical':
            points, rotations, tr, sp_tr, sp_weights, sp_knn = self.sp_stage(points, rotations, t, time_id)
            outputs.update(_tr=tr, _sp_tr=sp_tr, _sp_w=sp_weights, _sp_knn=sp_knn)
            points_c, rotations_c, tr_c, sp_tr_c = self.canonical_stage(time_id)
            outputs.update(_points_c=points_c, _rotations_c=rotations_c, _tr_c=tr_c, _sp_tr_c=sp_tr_c)
        else:
            raise ValueError(f"Error stage {stage}")
        # if stage != 'canonical' and self._canonical_aux is not None:
        #     self._canonical_aux = None
        if stage == 'skeleton' and self.sp_guided_detach and \
            self.sp_guided_steps_range[0] <= self._step <= self.sp_guided_steps_range[1]:
            opacity, sh_features, scales = opacity.detach(), sh_features.detach(), scales.detach()
        outputs['opacity'] = opacity
        outputs['points'] = points
        if self.convert_SHs_python and campos is not None:
            outputs['colors'] = self.get_colors(sh_features, points, campos)
        else:
            outputs['sh_features'] = sh_features

        if self.compute_cov3D:
            outputs['covariance'] = self.covariance_activation(scales, scaling_modifier, rotations)
            assert False
        else:
            outputs['scales'] = self.scaling_activation(scales)
            outputs['rotations'] = self.rotation_activation(rotations)
        return outputs

    def get_now_stage(self, stage=None, now_step: int = None):
        now_step = self._step if now_step is None else now_step
        if stage is None:
            for step, _stage in self.train_schedule:
                if now_step > step:  # self._step = Train.current_step + 1
                    stage = _stage
            if stage is None:
                stage = self.train_schedule[-1][1]
        return stage

    def render(
        self,
        *args,
        t: Tensor = None,
        info,
        background: Tensor = None,
        time_id=None,
        scale_modifier=1.0,
        stage: str = None,
        **kwargs
    ):
        outputs = {}
        inputs = self.prepare_inputs(info, t, background, scale_modifier)
        for b, (raster_settings, campos, bg, t) in enumerate(inputs):
            net_out = self(t=t, campos=campos)
            if 'hook' in kwargs:
                net_out = kwargs['hook'](net_out)

            for name in list(net_out.keys()):
                if name.startswith('_'):
                    outputs[name] = net_out.pop(name)
            outputs_b = self.gs_rasterizer(**net_out, raster_settings=raster_settings)
            images = torch.permute(outputs_b['images'], (1, 2, 0))
            if not self.use_official_gaussians_render and background is not None:
                images = images + (1 - outputs_b['opacity'][..., None]) * bg.squeeze(0)
            outputs_b['images'] = images
            outputs_b['points'] = net_out['points']
            outputs_b['rotations'] = net_out['rotations']
            outputs_b['scales'] = net_out['scales']
            if info['Tw2v'].ndim == 2:
                outputs_b['stage'] = stage
                return outputs_b
            elif b == 0:
                outputs = {k: [v] for k, v in outputs_b.items() if v is not None}
            else:
                for k, v in outputs_b.items():
                    if v is not None:
                        outputs[k].append(v)
        outputs = {k: torch.stack(v, dim=0) if k != 'viewspace_points' else v for k, v in outputs.items()}
        outputs['stage'] = stage
        return outputs

    @torch.no_grad()
    def update_joint(self, verbose=True):
        if self.sk_knn_num > 0:
            cost = self.joint_cost.clone()
            sp_dist = torch.cdist(self.sp_points, self.sp_points)
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

    def find_joint_loss(self, sp_tr: Tensor, time_id, update_joint=True, joint_rot: SO3 = None):
        self.init_joint_pos()
        sp_tr = sp_tr.detach()
        if self.sk_tr_batch_size <= 0:
            time_id = time_id if isinstance(time_id, Tensor) else torch.tensor(time_id, device=sp_tr.device)
            time_id = time_id.view(-1)
            T = (SE3.exp(sp_tr) if isinstance(sp_tr, Tensor) else sp_tr)[None]  # [1, M]
        else:
            time_id = torch.randint(0, self.num_frames, size=(self.sk_tr_batch_size,), device=sp_tr.device)
            T = SE3.exp(self.sp_delta_tr[time_id])  # [B, M]
        rot = SO3.exp(self.joint_rot[time_id]) if joint_rot is None else joint_rot[time_id]  # [B, M, M]
        translate = self.joint_pos[None] - rot.act(self.joint_pos[None])  # [B, M, M]
        local_tr = SE3.InitFromVec(torch.cat([translate, rot.vec()], dim=-1))
        joint_T = T[:, None, :] * local_tr
        joint_dist = (T[:, :, None].inv() * joint_T).log().norm(dim=-1).mean(dim=0)

        joint_pos_t = T[:, None, :].act(self.joint_pos[None])  # [B, M, M, 3]
        # for a in range(self.num_superpoints):
        #     for b in range(self.num_superpoints):
        #         joint_ab = self.joint_pos[a, b][None]
        #         rot_ab = rot[:, a, b]
        #         t_ab = joint_ab - rot_ab.act(joint_ab)
        #         lT_ab = SE3.InitFromVec(torch.cat([t_ab, rot_ab.vec()], dim=-1))
        #         T_a_ = T[:, b] * lT_ab
        #         T_a = T[:, a]
        #         error_ab = (T_a.inv() * T_a_).log().norm(dim=-1).mean()
        #         assert (error_ab - joint_dist[a, b]).abs().max() < 1e-5
        #         j_t = T[:, b].act(joint_ab)
        #         assert (joint_pos_t[:, a, b] - j_t).abs().max() < 1e-5
        joint_dist = joint_dist + (joint_pos_t - joint_pos_t.transpose(1, 2)).norm(dim=-1).mean(dim=0)  # [M, M]

        with torch.no_grad():
            self.joint_cost = self.joint_cost * self.momentum + joint_dist * (1. - self.momentum)
            if update_joint:
                self.update_joint(verbose=False)
        a, b, _ = self.joint_pair
        sp_joint_loss = (joint_dist[a, b] + joint_dist[b, a]) * 0.5
        return sp_joint_loss.mean(), joint_dist.mean()

    def loss_bone(self, sp_tr):
        a, b, mask = self.joint_pair
        joints = self.joint_pos[a, b]
        joints_t = SE3.exp(sp_tr[b]).act(joints)
        bone_mask = self.joint_parents[:, 0] != self.joint_root
        a = torch.nonzero(bone_mask)[:, 0]
        b = self.joint_parents[bone_mask, 0]
        a = torch.where(a >= self.joint_root, a - 1, a)
        b = torch.where(b >= self.joint_root, b - 1, b)
        dist_c = (joints[a] - joints[b]).square().sum(dim=-1)
        dist_t = (joints_t[a] - joints_t[b]).square().sum(dim=-1)
        return (dist_c - dist_t).square().sum()
        # return F.mse_loss(dist_c, dist_t)

    def loss_joint_discovery(self, sp_se3: SE3, sp_se3_c: SE3 = None, update_joint=True):
        """在训练过程中发现joints"""
        self.init_joint_pos()
        sp_se3 = sp_se3.detach() if self.sp_guided_detach else sp_se3
        if sp_se3_c is not None:
            sp_se3 = sp_se3 * sp_se3_c.inv()
        T = sp_se3[None, :].inv() * sp_se3[:, None]
        joint_R_t = SO3.InitFromVec(T.vec()[..., 3:])
        joint_dist = T.translation()[..., :3] - (self.joint_pos - joint_R_t.act(self.joint_pos))  # [M, M, 3]
        joint_dist = joint_dist.norm(dim=-1, keepdim=False)  # [M, M]

        joint_pos_t = sp_se3[None, :].act(self.joint_pos)  # [M, M, 3]
        joint_dist = joint_dist + (joint_pos_t - joint_pos_t.transpose(0, 1)).norm(dim=-1)  # [M, M]

        with torch.no_grad():
            if self.training:
                self.joint_cost = self.joint_cost * self.momentum + joint_dist * (1. - self.momentum)
                if update_joint:
                    self.update_joint()
        a, b, _ = self.joint_pair
        sp_joint_loss = (joint_dist[a, b] + joint_dist[b, a]) * 0.5
        return sp_joint_loss.mean(), joint_dist.mean()
        # return joint_dist.mean()

    def loss_canonical(self):
        t = self.train_db_times[self.canonical_time_id]
        # sp_tr = self.sp_deform_net(self.sp_points, t)
        # return F.mse_loss(sp_tr, torch.zeros_like(sp_tr))
        # sp_points_t = SE3.exp(sp_tr).act(self.sp_points)
        # return F.mse_loss(self.sp_points, sp_points_t)
        points_c = self.points
        points_t, rotations_t, _, sp_tr, _, _ = self.sp_stage(points_c, self.get_rotation, t)
        sp_points_t = SE3.exp(sp_tr).act(self.sp_points)
        return F.mse_loss(points_c, points_t) + F.mse_loss(self.sp_points, sp_points_t)

    def time_smooth_loss(self, tr_t: Tensor, scales_t: Tensor, time_id):
        # TODO: 非端点, 左右随机各抽取一个点, 插值得到当前时间的结果; 端点, 抽取两个点, 使用当前结果与较远的一个插值中间的
        if time_id == self.canonical_time_id:
            return F.mse_loss(tr_t, torch.zeros_like(tr_t))
        loss = 0
        cnt = 0
        if isinstance(time_id, Tensor):
            time_id = time_id.item()
        if time_id > 0:
            loss = F.mse_loss(tr_t, self.sp_delta_tr[time_id - 1])
            cnt += 1
        if time_id + 1 < self.num_frames:
            loss = F.mse_loss(tr_t, self.sp_delta_tr[time_id + 1]) + loss
            if self.predict_ds_sp:
                loss = F.mse_loss(scales_t, self.superpoint_delta_s[time_id + 1]) + loss
            cnt += 1
        assert not loss.isinf() and cnt > 0, \
            f"{time_id}, {tr_t}, {self.sp_delta_tr[time_id - 1]}, {self.sp_delta_tr[time_id + 1]}"
        return loss / cnt

    def loss_static_reg(self, tr: Tensor):
        """try to make point do not move"""
        return tr.abs().sum(dim=-1).mean()  # F.l1_loss(tr, torch.zeros_like(tr))

    def loss_weight_sparsity(self, weight: Tensor, eps=1e-7):
        return -(weight * torch.log(weight + eps) + (1 - weight) * torch.log(1 - weight + eps)).mean()

    def loss_weight_smooth(self, weight: Tensor):
        self.update_gs_knn()
        return (weight[:, None] - weight[self.gs_knn_index]).abs().mean()

    def loss_sp_arap(self, sp_se3: SE3):
        sp_points_c = self.sp_points
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

    def loss_sp_time_smooth(self, sp_tr: Tensor, t: Tensor, num_knn=4, lambda_w=2000):
        sp_xyz = self.superpoint_xyz
        with torch.no_grad():
            sp_dist = torch.cdist(sp_xyz, sp_xyz)
            k_dist, knn = torch.topk(sp_dist, dim=1, k=num_knn + 1, largest=False)
            knn = knn[:, 1:]
            weight = torch.exp(-k_dist[:, 1:] * lambda_w)
        t2 = (t + torch.randn_like(t) * 1 / self.num_frames).clamp(0, 1)
        t2_embed = self.pos_enc_t(t2.view(-1, 1)).expand(self.num_superpoints, -1)
        sp_inputs = torch.cat([self.pos_enc_p(self.superpoint_xyz), t2_embed], dim=-1)
        sp_tr_2, delta_s = self.sp_deform_net(sp_inputs).split([self._Rt_dim, 3 * self.predict_ds_sp], dim=-1)
        sp_tr_1 = SE3.exp(sp_tr)
        sp_tr_2 = SE3.exp(sp_tr_2)
        sp_xyz_t1 = sp_tr_1.act(sp_xyz)
        sp_xyz_t2 = sp_tr_2.act(sp_xyz)
        delta_1 = sp_xyz_t1[knn] - sp_xyz_t1[:, None, :]
        delta_2 = sp_xyz_t2[knn] - sp_xyz_t2[:, None, :]

        sp_R_1 = SO3.InitFromVec(sp_tr_1.vec()[..., 3:])
        sp_R_2 = SO3.InitFromVec(sp_tr_2.vec()[..., 3:])
        delta_2 = (sp_R_1 * sp_R_2.inv())[:, None].act(delta_2)
        local_rigid = torch.sum(weight * torch.norm(delta_1 - delta_2, dim=-1), dim=-1)
        return local_rigid.mean()

    def loss_local_rigid(self, sp_tr: Tensor, t: Tensor):
        """邻近Gaussian保持相对不变"""
        num_knn = 20
        lambda_w = 2000
        knn = 0
        weight = 0

        points, rotations = self.points.detach(), self.get_rotation().detach()
        t2 = (t + torch.randn_like(t) * 1 / self.num_frames).clamp(0, 1)
        t2_embed = self.pos_enc_t(t2.view(-1, 1)).expand(self.num_superpoints, -1)
        sp_inputs = torch.cat([self.pos_enc_p(self.superpoint_xyz), t2_embed], dim=-1)
        sp_tr_2, delta_s = self.sp_deform_net(sp_inputs).split([self._Rt_dim, 3 * self.predict_ds_sp], dim=-1)

        delta_T1 = SE3.exp(sp_tr)[self.p2sp]  # [N, 6]
        points_t1 = delta_T1.act(points)
        rotations_t1 = SO3.InitFromVec(delta_T1.vec()[..., 3:]) * SO3.InitFromVec(rotations)

        delta_T2 = SE3.exp(sp_tr_2)[self.p2sp]  # [N, 6]
        points_t2 = delta_T2.act(points)
        rotations_t2 = SO3.InitFromVec(delta_T2.vec()[..., 3:]) * SO3.InitFromVec(rotations)

        delta_1 = points_t1[:, None, :] - points_t1[knn]
        delta_2 = points_t2[:, None, :] - points_t2[knn]
        delta_2 = (rotations_t1 * rotations_t2.inv())[:, None].act(delta_2)
        local_rigid = torch.sum(weight * torch.norm(delta_1 - delta_2, dim=-1), dim=-1)
        return local_rigid.mean()

    def loss_guided_sk(self, losses, time_id, outputs):
        sp_tr = self.sp_delta_tr[time_id]
        sk_tr = outputs['_sk_tr']

        # if self.loss_funcs.w('guided') > 0:
        # guided_loss = (SE3.exp(sp_tr).inv() * SE3.exp(sk_tr)).log().norm(-1).mean()
        losses['guided'] = self.loss_funcs('guided', F.mse_loss, sk_tr, sp_tr)
        # losses['gj'] = self.loss_funcs('gj', F.mse_loss(self.sp_points, joint.detach()))

        if self.loss_funcs.w('cmp_r') > 0:
            if self.sp_no_lbs:
                dT_sp = SE3.exp(sp_tr)[self.p2sp]
            else:
                dT_sp = SE3.exp(torch.sum(sp_tr[self.sp_knn] * self.sp_weights[..., None], dim=1))  # [N, ?]
            rot_sp = SO3.InitFromVec(dT_sp.vec()[..., 3:]) * SO3.InitFromVec(self._canonical_aux[1])
            rot_sk = SO3.InitFromVec(outputs['rotations'])
            losses['cmp_r'] = self.loss_funcs('cmp_r', (rot_sp.inv() * rot_sk).log().norm(dim=-1).mean())
            # losses['cmp_r'] = self.loss_funcs('cmp_r', F.mse_loss(rot_sp.log(), rot_sk.log()))
            # losses['cmp_r'] = self.loss_funcs('cmp_r', F.mse_loss(rot_sp, rot_sk))

        if self.loss_funcs.w('cmp_p') > 0:
            # point_sp = dT_sp.act(self._canonical_aux[0])
            point_sp = self._canonical_aux[0]
            # assert not SE3.exp(sp_tr)[self.sp_knn].act(point_sp).isnan().any(), f"{time_id}"
            if self.sp_no_lbs:
                point_sp = SE3.exp(sp_tr)[self.p2sp].act(point_sp)
            else:
                point_sp = SE3.exp(sp_tr)[self.sp_knn].act(point_sp[:, None])
                point_sp = torch.sum(point_sp * self.sp_weights[..., None], dim=1)
            point_sk = outputs['points']
            losses['cmp_p'] = self.loss_funcs('cmp_p', F.mse_loss(point_sk, point_sp))

    def loss_guided_sp(self, losses, time_id, t, outputs: dict, sp_se3: SE3, sp_tr_c: Tensor = None):
        with torch.no_grad():
            if self.canonical_time_id >= 0:
                if sp_tr_c is None:
                    sp_tr_c = self.sp_deform_net(self.sp_points, self.train_db_times[self.canonical_time_id])
                sp_se3_c = SE3.exp(sp_tr_c)
                sp_se3_c_inv = sp_se3_c.inv()
                # if self.sp_no_lbs:
                #     points_c = sp_se3_c[self.p2sp].act(self.points)
                # else:
                #     points_c = sp_se3_c[self.sp_knn].act(self.points[:, None])
                #     points_c = torch.sum(points_c * self.sp_weights[..., None], dim=1)
                # sp_rot_c = SE3.exp(torch.sum(sp_tr_c[self.sp_knn] * self.sp_weights[..., None], dim=1))
                # rotations_c = SO3.InitFromVec(sp_rot_c.vec()[..., 3:]) * SO3.InitFromVec(self.get_rotation)
                points_c, rotations_c, _ = self.LBS_warp(
                    self.points, self.get_rotation, sp_tr_c, self.sp_knn, self.sp_weights)
                rotations_c = SO3.InitFromVec(rotations_c)
                joints = sp_se3_c.act(self.sp_points)
            else:
                joints = self.sp_points
                rotations_c = SO3.InitFromVec(self.get_rotation)
                points_c = self.points
        if not self.sk_W_is_init:
            self.sk_W.data.copy_(self.sp_W)
            self.sk_W_is_init = self.sk_W_is_init.new_tensor(True)

        sp_se3 = sp_se3.detach() if self.sp_guided_detach else sp_se3
        # g_tr = sp_se3[self.joint_root]
        if self.canonical_time_id >= 0:
            # g_tr = g_tr * sp_se3_c_inv[self.joint_root]
            sp_se3 = sp_se3 * sp_se3_c_inv
        # g_tr = SE3.exp(self.global_tr[time_id])
        g_tr = sp_se3[self.joint_root]
        joints = joints.clone()
        a, b, mask = self.joint_pair
        joints[mask] = self.joint_pos[a, b]
        # sk_tr = self.kinematic(joints, t, g_tr, time_id)
        sk_r = SO3.exp(self.sk_deform_net(joints, t))
        sk_t = joints + sk_r.act(-joints)
        sk_se3 = SE3.InitFromVec(torch.cat([sk_t, sk_r.vec()], dim=-1))
        temp = joints.new_zeros(self.num_superpoints, 7)
        temp[a] = (sp_se3[b] * sk_se3[a]).vec()
        temp[self.joint_root] = g_tr.vec()
        sk_se3 = SE3.InitFromVec(temp)
        weights = torch.gather(self.sk_W, dim=1, index=self.sp_knn).softmax(dim=-1)

        if self.loss_funcs.w('guided') > 0:
            guided_loss = (sp_se3.inv() * sk_se3).log().norm(dim=-1).mean()
            # guided_loss = F.mse_loss(sk_tr, sp_tr)
            losses['guided'] = self.loss_funcs('guided', guided_loss)
            # losses['gj'] = self.loss_funcs('gj', F.mse_loss(self.sp_points, joint.detach()))
        if self.loss_funcs.w('cmp_r') > 0:
            rot_sp = outputs['rotations'].detach() if self.sp_guided_detach else outputs['rotations']
            rot_sp = SO3.InitFromVec(rot_sp)
            rot_sk = sk_se3.log()
            rot_sk = SE3.exp(torch.sum(rot_sk[self.sp_knn, 3:] * weights[..., None], dim=1))  # [N, ?]
            rot_sk = rot_sk * rotations_c
            losses['cmp_r'] = self.loss_funcs('cmp_r', (rot_sp.inv() * rot_sk).log().norm(dim=-1).mean())
        if self.loss_funcs.w('cmp_p') > 0:
            point_sp = outputs['points'].detach() if self.sp_guided_detach else outputs['points']
            point_sk = torch.sum(sk_se3[self.sp_knn].act(points_c[:, None]) * weights[:, :, None], dim=1)
            losses['cmp_p'] = self.loss_funcs('cmp_p', F.mse_loss(point_sk, point_sp))

    def loss(self, inputs, outputs, targets):
        # self.timer.log('render')
        time_id = inputs['time_id'].item() if 'time_id' in inputs else None
        stage = outputs['stage']

        losses = {}
        if 'images' in targets:
            image = outputs['images']
            gt_img = targets['images'][..., :3]
            H, W, C = image.shape[-3:]
            image, gt_img = image.view(-1, H, W, C), gt_img.view(-1, H, W, C)
            losses['rgb'] = self.loss_funcs('image', image, gt_img)
            losses['ssim'] = self.loss_funcs('ssim', image, gt_img)
        if not self.training:
            return losses
        if stage == 'canonical':
            losses['cmp_p'] = F.mse_loss(outputs['_points_c'], outputs['points'])
            loss_rot = SO3.InitFromVec(outputs['rotations']).inv() * SO3.InitFromVec(outputs['_rotations_c'])
            loss_rot = loss_rot.log().norm(dim=-1).mean()
            # loss_rot = F.mse_loss(outputs['rotations'], outputs['_rotations_c'])
            losses['cmp_r'] = loss_rot
            return losses
        # self.timer.log('loss_rgb')
        if stage == 'sp':
            Neighbor, W = outputs['_sp_knn'], outputs['_sp_w']
            sp_se3 = SE3.exp(outputs['_sp_tr'])
            if self.loss_funcs.w('re_sp_tr'):
                re_sp_tr = get_superpoint_features(outputs['_tr'], Neighbor, W, self.num_superpoints)
                losses['re_sp_tr'] = self.loss_funcs('re_sp_tr', F.mse_loss(re_sp_tr, outputs['_sp_tr']))
            if self.loss_funcs.w('re_sp_pos') > 0:
                re_sp = get_superpoint_features(outputs['points'], Neighbor, W, self.num_superpoints)
                sp = sp_se3.act(self.sp_points)
                losses['re_sp_pos'] = self.loss_funcs('re_sp_pos', F.mse_loss(sp, re_sp))
            self.sp_delta_tr[time_id] = outputs['_sp_tr'].detach()

            if self.canonical_time_id >= 0:
                with torch.no_grad():
                    sp_tr_c = self.sp_deform_net(self.sp_points, self.train_db_times[self.canonical_time_id])
                    sp_se3_c = SE3.exp(sp_tr_c)
            else:
                sp_tr_c = None
                sp_se3_c = None
            if self.loss_funcs.w('joint') > 0 and self._step >= self.joint_update_interval[1]:
                update_joint = utils.check_interval(self._step, *self.joint_update_interval)
                if self.sk_tr_batch_size >= 0:
                    best_cost, all_cost = self.find_joint_loss(outputs['_sp_tr'], time_id, update_joint)
                else:
                    best_cost, all_cost = self.loss_joint_discovery(sp_se3, sp_se3_c, update_joint)
                losses['joint'] = self.loss_funcs('joint', best_cost)
                losses['joint_all'] = self.loss_funcs('joint_all', all_cost)
                if self.loss_funcs.w('jp_dist') > 0:
                    a, b, mask = self.joint_pair
                    with torch.no_grad():
                        sp_points_t = sp_se3.act(self.sp_points)
                    joints = sp_se3[b].detach().act(self.joint_pos[a, b])
                    jd_dist = F.mse_loss(joints, sp_points_t[a]) + F.mse_loss(joints, sp_points_t[b])
                    losses['jp_dist'] = self.loss_funcs('jp_dist', jd_dist)
            if self._step >= self.loss_arap_start_step:
                if self.loss_funcs.w('sp_arap_t') > 0:
                    arap_t, arap_ct = self.loss_sp_arap(sp_se3)
                    losses['arap_t'] = self.loss_funcs('sp_arap_t', arap_t)
                    losses['arap_ct'] = self.loss_funcs('sp_arap_ct', arap_ct)
            losses['sparse'] = self.loss_funcs('sparse', self.loss_weight_sparsity, W)
            losses['smooth'] = self.loss_funcs('smooth', self.loss_weight_smooth, W)
            if self.canonical_time_id >= 0:
                losses['c'] = self.loss_funcs('c', self.loss_canonical)
            if self.training and self.sp_guided_steps_range[0] <= self._step <= self.sp_guided_steps_range[1]:
                self.loss_guided_sp(losses, time_id, inputs['t'], outputs, sp_se3, sp_tr_c)
        if stage == 'skeleton':
            losses['sparse'] = self.loss_funcs('sparse', self.loss_weight_sparsity, outputs['_lbs_w'])
            losses['smooth'] = self.loss_funcs('smooth', self.loss_weight_smooth, outputs['_lbs_w'])
            if self.training and self.sp_guided_steps_range[0] <= self._step <= self.sp_guided_steps_range[1]:
                self.loss_guided_sk(losses, time_id, outputs)
        # if self.loss_funcs.w('static_reg') > 0:
        #     # if 'points' in outputs:
        #     #     losses['static_reg'] = self.loss_funcs('static_reg', F.mse_loss(self.points, outputs['points']))
        #     if '_tr' in outputs:
        #         losses['static_reg'] = self.loss_funcs('static_reg', self.loss_static_reg(outputs['_tr']))

        # if self.training and self.sp_guided_steps_range[0] <= self._step <= self.sp_guided_steps_range[1]:
        #     if stage == 'skeleton':
        #         sp_tr = outputs['_sp_tr']
        #         sk_tr = outputs['_sk_tr']
        #     else:
        #         sp_tr = outputs['_sp_tr'].detach() if self.sp_guided_detach else outputs['_sp_tr']
        #         joint = self.sp_points.clone().detach()
        #         a, b, mask = self.joint_pair
        #         joint[mask] = self.joint_pos[a, b]
        #         sk_tr = self.kinematic(joint, sp_tr, inputs['t'], time_id)
        #
        #     if self.loss_funcs.w('guided') > 0:
        #         # guided_loss = (SE3.exp(sp_tr).inv() * SE3.exp(sk_tr)).log().norm(dim=-1).mean()
        #         guided_loss = F.mse_loss(sk_tr, sp_tr)
        #         losses['guided'] = self.loss_funcs('guided', guided_loss)
        #         # losses['gj'] = self.loss_funcs('gj', F.mse_loss(self.sp_points, joint.detach()))
        #     if self.loss_funcs.w('cmp_r', '0') > 0 or self.loss_funcs.w('cmp_p', '0') > 0:
        #         if stage == 'skeleton':
        #             dT_sp = SE3.exp(torch.sum(sp_tr[self.sp_knn] * self.sp_weights[..., None], dim=1))  # [N, ?]
        #             dT_sk = SE3.exp(outputs['_tr'])
        #         else:
        #             dT_sp = SE3.exp(outputs['_tr'])
        #             dT_sp = dT_sp.detach() if self.sp_guided_detach else dT_sp
        #             dT_sk = SE3.exp(torch.sum(sk_tr[self.sp_knn] * self.sp_weights[..., None], dim=1))  # [N, ?]
        #         losses['cmp_r'] = self.loss_funcs('cmp_r', (dT_sp.inv() * dT_sk).log().norm(dim=-1).mean())
        #         if self.loss_funcs.w('cmp_p', '0') > 0:
        #             if stage == 'skeleton':
        #                 point_sp = dT_sp.act(self.points.detach())
        #                 point_sk = outputs['points']
        #             else:
        #                 point_sp = outputs['points'].detach() if self.sp_guided_detach else outputs['points']
        #                 point_sk = dT_sk.act(self.points.detach())
        #             losses['cmp_p'] = self.loss_funcs('cmp_p', F.mse_loss(point_sk, point_sp))

        # if '_sp_tr' in outputs and self.loss_funcs.w('ts') > 0 and time_id is not None:
        #     losses['ts'] = self.loss_funcs('ts', self.time_smooth_loss(
        #         outputs['_sp_tr'], outputs.get('_scales', None), time_id))
        #
        # if '_sp_tr' in outputs and self.loss_funcs.w('sp_ts') > 0:
        #     losses['sp_ts'] = self.loss_funcs('sp_ts', self.loss_sp_time_smooth(outputs['_sp_tr'], inputs['t']))

        # if '_sk_tr' in outputs and '_sp_tr' in outputs:
        #     sk_sp_loss = (SO3.exp(outputs['_sp_tr']).inv() * SO3.exp(outputs['_sk_tr'])).log().norm(dim=-1).mean()
        #     losses['sk_sp'] = self.loss_funcs('sk_sp', sk_sp_loss)

        if '_sk_tr' in outputs and self.loss_funcs.w('sk_l1') > 0:
            losses['sk_l1'] = self.loss_funcs('sk_l1', outputs['_sk_tr'].abs().mean())
        # self.timer.log('other loss')
        # print(self.timer)
        # for k, v in losses.items():
        #     assert not isinstance(v, Tensor) or not v.isnan().any(), f"{k} is nan {v}"
        return {k: v for k, v in losses.items() if isinstance(v, Tensor)}
        # return losses

    @torch.no_grad()
    def superpoint_prune_split(self, optimizer: torch.optim.Optimizer):
        ## prune
        dist = torch.cdist(self.points, self.sp_points)  # [N, M]
        _, knn = torch.topk(dist, k=min(self.num_superpoints, self.sp_knn_num), dim=-1, largest=False)
        weight = torch.gather(self.sp_W, dim=1, index=knn).softmax(dim=-1)  # [N, K]
        W = torch.scatter_reduce(weight.new_zeros(self.num_superpoints), 0, knn.view(-1), weight.view(-1), 'sum')
        mask = torch.ge(W, self.sp_prune_threshold)
        num_pruned = self.num_superpoints - mask.sum().item()
        if num_pruned > 0:
            self.sp_points = self.change_optimizer(optimizer, mask, ['sp_points'], op='prune')['sp_points']
            self.sp_W = self.change_optimizer(optimizer, mask, ['sp_W'], op='prune', dim=1)['sp_W']
            self.sk_W = self.change_optimizer(optimizer, mask, ['sk_W'], op='prune', dim=1)['sk_W']
            self.sp_delta_tr = self.sp_delta_tr[:, mask, :]
            self.joint_cost = self.joint_cost[mask][:, mask]
            self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune')['joint_pos']
            self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune', dim=1)['joint_pos']
            if self.sk_tr_batch_size >= 0:
                self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=1)['joint_rot']
                self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=2)['joint_rot']
            logging.info(f'prune superpoints {self.num_superpoints} to {self.sp_points.shape[0]}')
            self.num_superpoints = self.sp_points.shape[0]
            ## split
            _, knn = torch.topk(dist[:, mask], k=min(self.num_superpoints, self.sp_knn_num), dim=-1, largest=False)
            weight = torch.gather(self.sp_W, dim=1, index=knn).softmax(dim=-1)  # [N, K]
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
            self.sp_W = self.change_optimizer(optimizer, {'sp_W': self.sp_W[:, mask]}, op='concat', dim=1)['sp_W']
            self.sk_W = self.change_optimizer(optimizer, {'sk_W': self.sk_W[:, mask]}, op='concat', dim=1)['sk_W']
            self.sp_delta_tr = torch.cat([self.sp_delta_tr, self.sp_delta_tr[:, mask]], dim=1)
            self.joint_pos = self.change_optimizer(
                optimizer, self.joint_pos[mask], 'joint_pos', op='concat')['joint_pos']
            self.joint_pos = self.change_optimizer(
                optimizer, self.joint_pos[:, mask], 'joint_pos', op='concat', dim=1)['joint_pos']
            if self.sk_tr_batch_size >= 0:
                self.joint_rot = self.change_optimizer(
                    optimizer, self.joint_rot[:, mask, :], 'joint_rot', op='concat', dim=1)['joint_rot']
                self.joint_rot = self.change_optimizer(
                    optimizer, self.joint_rot[:, :, mask], 'joint_rot', op='concat', dim=2)['joint_rot']
            self.joint_cost = torch.cat([
                torch.cat([self.joint_cost, self.joint_cost[:, mask]], dim=1),
                torch.cat([self.joint_cost[mask], self.joint_cost[:, mask][mask, :]], dim=1)
            ], dim=0)
            logging.info(f'split superpoints {self.num_superpoints} to {self.sp_points.shape[0]}')
        self.num_superpoints = self.sp_points.shape[0]
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

    def non_overlap_merge(self, min_diff: Tensor, min_index: Tensor):
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
        for i, t in enumerate(self.train_db_times):
            self.sp_delta_tr[i] = self.sp_deform_net(self.sp_points, t)

        dist = torch.cdist(self.sp_points, self.sp_points)  # [N, M]
        knn = torch.topk(dist, k=min(self.num_superpoints, self.sp_knn_num + 1), dim=-1, largest=False)[1][:, 1:]
        tr_diff = (self.sp_delta_tr[:, :, None] - self.sp_delta_tr[:, knn]).norm(dim=-1)  # [T, M, K]
        tr_diff = tr_diff.mean(dim=0)  # [M, K]
        min_diff, min_index = torch.min(tr_diff, dim=1)
        mask = min_diff.lt(self.sp_merge_threshold)
        num_merge = mask.sum().item()
        if num_merge == 0:
            return
        min_index = knn[torch.arange(self.num_superpoints, device=min_index.device), min_index]
        # mask, merge = self.all_merge(mask, min_index)
        mask, merge = self.non_overlap_merge(min_diff, min_index)

        self.sp_points = self.change_optimizer(optimizer, mask, ['sp_points'], op='prune')['sp_points']
        new_W = torch.scatter_reduce(torch.zeros_like(self.sp_W), 1, merge.expand_as(self.sp_W), self.sp_W, 'sum')
        self.sp_W = self.change_optimizer(optimizer, new_W[:, mask], ['sp_W'], op='replace', dim=1)['sp_W']
        new_W = torch.scatter_reduce(torch.zeros_like(self.sk_W), 1, merge.expand_as(self.sk_W), self.sk_W, 'sum')
        self.sk_W = self.change_optimizer(optimizer, new_W[:, mask], ['sk_W'], op='replace', dim=1)['sk_W']
        self.sp_delta_tr = self.sp_delta_tr[:, mask, :]
        self.joint_cost = self.joint_cost[mask, :][:, mask]
        self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune')['joint_pos']
        self.joint_pos = self.change_optimizer(optimizer, mask, 'joint_pos', op='prune', dim=1)['joint_pos']
        if self.sk_tr_batch_size >= 0:
            self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=1)['joint_rot']
            self.joint_rot = self.change_optimizer(optimizer, mask, 'joint_rot', op='prune', dim=2)['joint_rot']
        logging.info(f'merge superpoints {self.num_superpoints} to {self.sp_points.shape[0]}')
        self.num_superpoints = self.sp_points.shape[0]
        self.update_skeleton_shapes()

    def update_skeleton_shapes(self):
        if self.num_superpoints == self.joint_parents.shape[0]:
            return
        M = self.num_superpoints
        self.skeleton_r = self.skeleton_r.new_zeros(self.num_frames, M, 3)
        if self.sk_feature is not None:
            self.sk_feature = nn.Parameter(self.sk_feature.new_zeros(M, self.sk_feature_dim))
        if not self.sk_use_features:
            self.sk_deform_net.dynamic_net.last = nn.Linear(self.sk_deform_net.dynamic_net.last.in_features, M * 3)
        # if self._step >= self.joint_update_interval[1]:
        self.update_joint()

    def prune_points(self, optimizer: torch.optim.Optimizer, mask):
        super().prune_points(optimizer, mask)
        if self.sp_no_lbs:
            self.p2sp = self.p2sp[~mask]

    def densification_postfix(self, optimizer, mask=None, N=None, **kwargs):
        super().densification_postfix(optimizer, **kwargs, N=N, mask=mask)
        if self.sp_no_lbs:
            if N is None:
                self.p2sp = torch.cat([self.p2sp, self.p2sp[mask]], dim=0)
            else:
                self.p2sp = torch.cat([self.p2sp, self.p2sp[mask].repeat(N)], dim=0)

    def adaptive_control(self, inputs, outputs, optimizer, step: int):
        radii = outputs['radii']
        viewspace_point_tensor = outputs['viewspace_points']
        stage = outputs['stage']
        if not (step >= self.adaptive_control_cfg['densify_interval'][2]):
            if radii.ndim == 2:
                radii = radii.amax(dim=0)
            mask = radii > 0
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[mask] = torch.max(self.max_radii2D[mask], radii[mask])
            self.add_densification_stats(viewspace_point_tensor, mask)

        if stage == 'sp' and utils.check_interval_v2(
            step, *self.adaptive_control_cfg['sp_adjust_interval'], close='()'):
            self.superpoint_prune_split(optimizer)
            logging.info(f"superpoint_prune_split {self.num_superpoints} at step {step}")
        if stage in ['sp'] and utils.check_interval_v2(
            step, *self.adaptive_control_cfg['sp_merge_interval'], close='()'):
            self.superpoint_merge(optimizer)
            # logging.info(f"superpoint_merge {self.num_superpoints} at step {step}")

        if not (step >= max(self.adaptive_control_cfg['densify_interval'][2],
            self.adaptive_control_cfg['prune_interval'][2]) >= 0):
            if utils.check_interval_v2(step, *self.adaptive_control_cfg['densify_interval'], close='()'):
                self.densify(
                    optimizer,
                    max_grad=self.adaptive_control_cfg['densify_grad_threshold'],
                    extent=self.cameras_extent,
                    densify_percent_dense=self.adaptive_control_cfg['densify_percent_dense'],
                )
                logging.info(f'densify gaussians at {step}, there are {len(self.points)} points')

            if utils.check_interval_v2(step, *self.adaptive_control_cfg['prune_interval'], close='()'):
                if step >= self.adaptive_control_cfg['opacity_reset_interval'][1] \
                    and self.adaptive_control_cfg['prune_max_screen_size'] > 0:
                    size_threshold = self.adaptive_control_cfg['prune_max_screen_size']
                else:
                    size_threshold = None
                self.prune(
                    optimizer,
                    min_opacity=self.adaptive_control_cfg['prune_opacity_threshold'],
                    extent=self.cameras_extent,
                    max_screen_size=size_threshold,
                    prune_percent_dense=self.adaptive_control_cfg['prune_percent_dense'],
                )
                logging.info(f'prune gaussians at {step}, there are {len(self.points)} points')
            if (utils.check_interval_v2(step, *self.adaptive_control_cfg['opacity_reset_interval'], close='()') or
                (self.background_type == 'white' and step == self.adaptive_control_cfg['densify_interval'][1])):
                self.reset_opacity(optimizer)
                logging.info(f'reset_opacity at {step}')

    def hook_before_train_step(self):
        if self.sk_is_init or self.get_now_stage() != 'skeleton':
            return
        return
        progress = self._task.progress  # type: ext.Progress # noqa
        progress.pause('train')
        progress.add_task('joint', total=self.joint_init_steps)
        progress.start('joint')
        self.init_skeleton(lambda: progress.step('joint'))
        self._task.save_model('joint_init.pth')
        progress.stop('joint')
        progress.start('train')
        return


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
