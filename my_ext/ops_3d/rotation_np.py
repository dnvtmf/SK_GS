from scipy.spatial.transform import Rotation
import numpy as np


def R_to_quaternion(R: np.ndarray) -> np.ndarray:
    x = Rotation.from_matrix(R)
    quat = x.as_quat(canonical=False)
    return quat


def axis_angle_to_R(axis, theta, is_degree=False) -> np.ndarray:
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    return Rotation.from_rotvec(axis * theta, degrees=is_degree).as_matrix()


def test():
    print(axis_angle_to_R([0, 0, 1.], np.pi * 0.3))
