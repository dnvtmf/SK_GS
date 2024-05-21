from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from torch import Tensor

from .base_3D import Structure3D
from .camera import CameraExtrinsics, CameraIntrinsics


class PointClouds(Structure3D):
    points: Tensor
    """shape: [N, C] C >= 3, i.e. x, y, z, ..."""
    colors: Optional[Tensor] = None
    """shape: [N, 3/4], r, g, b, (alpha)"""
    normals: Optional[Tensor] = None
    """shape [N, 3]"""

    def __init__(self, points, colors=None, normals=None, info=None):
        super(PointClouds, self).__init__(info)
        if isinstance(points, torch.Tensor):
            self.points = points.detach()
        elif isinstance(points, np.ndarray):
            assert points.ndim == 2 and points.shape[1] >= 3
            self.points = torch.from_numpy(points)
        else:
            raise NotImplementedError
        if colors is not None:
            if isinstance(colors, np.ndarray):
                colors = torch.from_numpy(colors)
            assert isinstance(colors, Tensor) and colors.shape[0] == self.points.shape[0] and colors.shape[1] in [3, 4]
            self.colors = colors
        else:
            self.colors = None
        if normals is not None:
            if isinstance(normals, np.ndarray):
                normals = torch.from_numpy(normals)
            assert isinstance(normals, Tensor) and normals.shape == (self.points.shape[0], 3)
            self.normals = normals
        else:
            self.normals = None

    def to_trimesh(self):
        import trimesh.points
        colors = self.colors
        if colors is not None:
            colors = colors.cpu().numpy()
            if colors.dtype != np.uint8:
                colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
            if colors.shape[-1] != 4:
                colors = np.concatenate([colors, np.full_like(colors[..., :1], 255)], axis=-1)
        tri_pc = trimesh.points.PointCloud(
            self.points.cpu().numpy(),
            colors,
            # self.normals.cpu().numpy() if self.normals is not None else None,
        )
        return tri_pc

    @staticmethod
    def load(filename: Union[str, Path], *args, **kwargs):
        filename = Path(filename)
        if filename.suffix == '.pcd':
            return PointClouds.load_from_pcd(filename, *args, **kwargs)
        else:
            raise NotImplementedError(f'Can not read point clouds. Format {filename.suffix} is not supported now.')

    @staticmethod
    def load_from_pcd(filename: Union[str, Path], *args, **kwargs):
        _info = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
        start_line_no = 0
        for line in lines:
            start_line_no += 1
            items = line.split()
            if items[0][0] == '#':
                continue
            _info[items[0]] = items[1:]
            if items[0] == 'DATA':
                break
        points = torch.tensor([list(map(float, line.split())) for line in lines[start_line_no:]], dtype=torch.float32)
        return PointClouds(points, *args, **kwargs)

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self):
        return f"PointClouds{list(self.points.shape)}"

    def copy(self):
        return PointClouds(self.points, self.colors, self.normals, self.infos)

    def clone(self):
        return PointClouds(self.points.clone(),
            self.colors.clone() if self.colors is not None else None,
            self.normals.clone() if self.normals is not None else None,
            self.infos)

    def pan_zoom_(self, scaling=1., offset=0., *args, **kwargs):
        scaling = self.points.new_tensor(scaling)
        offset = self.points.new_tensor(offset)
        self.points[:, :3] = (self.points[:, :3] + offset) * scaling
        return self

    def flip_(self, x_axis=False, y_axis=False, z_axis=False, center=(0., 0., 0.), *args, **kwargs):
        if x_axis:
            # self.points[:, 0] = -(self.points[:, 0] - flip_center[0]) + flip_center[0]
            self.points[:, 0] = torch.add(-self.points[:, 0], center[0], alpha=2.)
        if y_axis:
            # self.points[:, 1] = -(self.points[:, 1] - flip_center[1]) + flip_center[1]
            self.points[:, 1] = torch.add(-self.points[:, 1], center[1], alpha=2.)
        if z_axis:
            # self.points[:, 2] = -(self.points[:, 2] - flip_center[2]) + flip_center[2]
            self.points[:, 2] = torch.add(-self.points[:, 2], center[2], alpha=2.)
        return self

    def project_to_2d(self, cam_in, cam_ex):
        # type: (CameraIntrinsics, CameraExtrinsics) -> Tensor
        points = torch.cat([self.points, self.points.new_ones(self.points.shape[0], 1)], dim=1)
        P = torch.from_numpy(cam_in.K @ cam_ex.T)
        points = points @ P.t
        points_2d = points[:, :2] / points[:, 2:3]
        return points_2d

    def affine_transform_(self, affine_matrix, *args, **kwargs):
        self.points[:, :3] = self.points[:, :3] @ affine_matrix[:3, :3] + affine_matrix[3, :3]
        return self

    def draw(self, vis):
        import open3d as o3d
        points = self.points.numpy()[:, :3]  # xyz
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.add(pcd)
