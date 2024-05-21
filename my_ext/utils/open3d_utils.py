from typing import Union

import open3d as o3d
import numpy as np
from torch import Tensor


def to_open3d_type(value: Union[Tensor, np.ndarray]):
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    result = None
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.int32)
        if value.ndim == 1:
            result = o3d.utility.IntVector(value)
        elif value.ndim == 2:
            if value.shape[-1] == 2:
                result = o3d.utility.Vector2iVector(value)
            elif value.shape[-1] == 3:
                result = o3d.utility.Vector3iVector(value)
            elif value.shape[-1] == 4:
                result = o3d.utility.Vector4iVector(value)
    else:
        value = value.astype(np.float64)
        if value.ndim == 1:
            result = o3d.utility.DoubleVector(value)
        elif value.ndim == 2:
            if value.shape[-1] == 3:
                result = o3d.utility.Vector3dVector(value)
            elif value.shape[-1] == 2:
                result = o3d.utility.Vector2dVector(value)
        elif value.ndim == 3:
            if value.shape[-1] == 3:
                result = o3d.utility.Matrix3dVector(value)
    if result is None:
        raise NotImplementedError(value.shape, value.dtype)
    return result
