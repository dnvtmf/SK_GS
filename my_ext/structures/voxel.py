from typing import List, Union, Any, Optional, Tuple

import torch
from torch import Tensor

from my_ext.utils.utils_3d import points_to_voxel_mean, points_to_voxel_dense
from my_ext.utils.torch_utils import to_tensor
from .base_3D import Structure3D


class SparseVoxel(Structure3D):
    def __init__(self, voxels: Tensor, coordinates: Tensor, resolution, info: dict = None):
        """
            稀疏体素
        Args:
            voxels: shape[N, C] 每个体素的特征
            coordinates: shape [N,3] 每个体素的坐标
            resolution: shape [3] 体素的形状， 如 [256, 256, 256]
            info:
        """
        super(SparseVoxel, self).__init__(info=info)
        self._resolution = to_tensor(resolution, dtype=torch.long)
        # self._axis_range = to_tensor(axis_range, dtype=torch.float).view(2, 3)  # type: Tensor
        # self._voxel_size = (self._axis_range[1] - self._axis_range[0]) / self.resolution

        self.coordinates = to_tensor(coordinates)  # zyx format
        self.voxels = to_tensor(voxels)

    @staticmethod
    def generate_from_points(points: Tensor, resolution, max_voxels=20000, info=None):
        voxels, coordinates = points_to_voxel_mean(points, resolution, (), False, max_voxels)
        return SparseVoxel(voxels, coordinates, resolution, info=info)

    @property
    def resolution(self):
        return self._resolution

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel={tuple(self.voxels.shape)}, " \
               f"resolution={self.resolution.tolist()}, " \
               f"infos={list(self.infos.keys())})"

    def to_dense(self):
        Gx, Gy, Gz = self.resolution.tolist()
        voxels = self.voxels.new_zeros((Gz, Gy, Gx, self.voxels.shape[-1]))
        voxels[self.coordinates.unbind(1)] = self.voxels
        return Voxel(voxels, self.infos)


class Voxel(Structure3D):
    def __init__(self, data, info=None):
        # type: (Any, Union[List[int], Tensor], Optional[dict]) -> None
        """
            (密集)体素

        Args:
            data: shape [D, H, W, C] 体素的坐标
            info: 其他信息
        """
        super(Voxel, self).__init__(info=info)
        self.data = to_tensor(data)

    def __repr__(self):
        return f"{self.__class__.__name__}(resolution={self.resolution}, " \
               f"num_features={self.data.shape[-1]}, " \
               f"infos={list(self.infos.keys())})"

    @property
    def resolution(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape[:3][::-1])

    @staticmethod
    def generate_from_points(points: Tensor, resolution):
        voxels = points_to_voxel_dense(points, resolution)
        return Voxel(voxels)

    def to_sparse(self):
        coords = torch.where(self.data.ne(0).all(dim=-1))
        values = self.data[coords]
        return SparseVoxel(values, torch.stack(coords, -1), self.resolution, self.infos)
