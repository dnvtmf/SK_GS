from typing import Union

from torch import Tensor
import numpy as np
import mcubes


def marching_cubes(voxel: Union[np.ndarray, Tensor], float_isovalue):
    if isinstance(voxel, Tensor):
        voxel = voxel.numpy()
    return mcubes.marching_cubes(voxel, float_isovalue)
