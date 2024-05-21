import torch
from torch import Tensor

from my_ext._C import try_use_C_extension, get_C_function


@try_use_C_extension
def FurthestSampling(points: Tensor, npoint: int):
    """ farthest point sample
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = points.device
    B, N, C = points.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    # farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    farthest = torch.zeros((B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def test():
    from my_ext.utils.test_utils import get_run_speed
    from my_ext._C import get_python_function
    cu_func = get_C_function('FurthestSampling')
    py_func = get_python_function('FurthestSampling')
    B, N, C = 10, 10000, 3
    points = torch.randn(B, N, C).cuda()
    M = 512
    cu_out = cu_func(points, M)
    py_out = py_func(points, M)
    assert (cu_out - py_out).abs().max().item() == 0
    # print(py_out)
    get_run_speed((points, M), None, py_func, cu_func)
