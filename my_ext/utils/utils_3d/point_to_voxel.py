from typing import Union, List, Tuple

import numpy as np
import torch
from torch import Tensor

from my_ext._C import try_use_C_extension


# @numba.njit
def _points_to_voxel_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    grid_size,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        failed = False
        for j in range(ndim):
            if len(coors_range) > 0:
                c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            else:
                c = np.floor(points[i, j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[2], coor[1], coor[0]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[2], coor[1], coor[0]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


# @numba.njit
def _points_to_voxel_kernel(
    points,
    voxel_size,
    coors_range,
    grid_size,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        failed = False
        for j in range(ndim):
            if len(coors_range) > 0:
                c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            else:
                c = np.floor(points[i, j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@try_use_C_extension
def points_to_voxel(points: Tensor, resulotion, axis_range=(), max_points=35, reverse_index=True, max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculates
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than Windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        resulotion (list, tuple) : [3] or array, int. indicate resulotion
        axis_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if isinstance(points, Tensor):
        points = points.detach().cpu().numpy()
    if not isinstance(resulotion, np.ndarray):
        resulotion = np.array(resulotion, dtype=np.int32)
    if len(axis_range) > 0:
        if not isinstance(axis_range, np.ndarray):
            axis_range = np.array(axis_range, dtype=points.dtype)
        voxel_size = (axis_range[3:] - axis_range[:3]) / resulotion
    else:
        voxel_size = None
    voxelmap_shape = resulotion
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points,
            voxel_size,
            axis_range,
            voxelmap_shape,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    else:
        voxel_num = _points_to_voxel_kernel(
            points,
            voxel_size,
            axis_range,
            voxelmap_shape,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )
    # if voxel_num == 0:
    #     print(points.shape, voxel_num)
    #     print('min:', points.min(axis=0), 'max:', points.max(axis=0))
    #     print(f'coors_range={coors_range}')
    #     print(f'voxel_size={voxel_size}')
    #     print(f'voxelmap_shape={voxelmap_shape}')
    #     print(f'max_points={max_points}')
    #     print(f'max_voxels={max_voxels}')
    #     exit(1)
    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


@try_use_C_extension
def points_to_voxel_mean(
    points: Tensor,
    resolution: Union[List[float], Tuple[float]],
    axis_range: Union[List[float], Tuple[float]] = (),
    reverse_index=True,  # True: zyx False xyz
    max_voxels=20000,
) -> Tuple[Tensor, Tensor]:
    resolution = torch.tensor(resolution, dtype=torch.int32)
    if len(axis_range) > 0:
        axis_range = torch.tensor(axis_range, dtype=torch.float)
        voxel_size = (axis_range[3:] - axis_range[:3]) / resolution
    else:
        voxel_size = None

    voxels = torch.zeros((max_voxels, points.shape[1]), dtype=points.dtype)
    coords = torch.zeros((max_voxels, 3), dtype=torch.int64)
    num_points = torch.zeros(max_voxels, dtype=torch.int32)
    voxel_idx = -torch.ones(resolution.tolist(), dtype=torch.int32)
    num_voxel = 0
    for i in range(points.shape[0]):
        if len(axis_range) > 0:
            point_i = torch.floor((points[i, :3] - axis_range[:3]) / voxel_size).long()
        else:
            point_i = torch.floor(points[i, :3]).long()
        if torch.logical_or(point_i.lt(0), point_i.ge(resolution)).any():
            continue
        vid = voxel_idx[point_i[0], point_i[1], point_i[2]]
        if vid == -1:
            if num_voxel >= max_voxels:
                continue
            voxel_idx[point_i[0], point_i[1], point_i[2]] = num_voxel
            vid = num_voxel
            num_voxel += 1
            coords[vid] = point_i
        voxels[vid] += points[i]
        num_points[vid] += 1
    voxels = voxels[:num_voxel] / num_points[:num_voxel, None]
    coords = coords[:num_voxel, (2, 1, 0)] if reverse_index else coords[:num_voxel]
    return voxels, coords


@try_use_C_extension
def points_to_voxel_dense(
    points: Tensor,
    resolution: Union[List[float], Tuple[float]],
    axis_range: Union[List[float], Tuple[float]] = (),
) -> Tensor:
    """生成大小为DxHxW的voxel, axis_range不为空时，不会改变points中的xyz"""
    resolution = torch.tensor(resolution, dtype=torch.int32)
    if len(axis_range) > 0:
        axis_range = torch.tensor(axis_range, dtype=torch.float)
        voxel_size = (axis_range[3:] - axis_range[:3]) / resolution
    else:
        voxel_size = None
    Gx, Gy, Gz = resolution.tolist()

    voxels = torch.zeros((Gz, Gy, Gx, points.shape[1]), dtype=points.dtype)
    num_points = torch.zeros((Gz, Gy, Gx), dtype=torch.int32)
    for i in range(points.shape[0]):
        if len(axis_range) > 0:
            point_i = torch.floor((points[i, :3] - axis_range[:3]) / voxel_size).long()
        else:
            point_i = torch.floor(points[i, :3]).long()
        if torch.logical_or(point_i.lt(0), point_i.ge(resolution)).any():
            continue
        voxels[point_i[2], point_i[1], point_i[0]] += points[i]
        num_points[[point_i[2], point_i[1], point_i[0]]] += 1
    voxels = voxels / num_points.unsqueeze(-1).clamp_min(1.)
    return voxels


def test_1():
    from my_ext._C import get_C_function, get_python_function
    import time
    print()
    np.set_printoptions(precision=4, suppress=False, linewidth=200)
    torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
    c_fun = get_C_function('points_to_voxel')
    py_fun = get_python_function('points_to_voxel')

    pc_range = [-10, -10, -10, 10, 10, 10]
    resulotion = [100, 100, 100]
    points = torch.randn(200000, 3) * (torch.tensor(pc_range[3:]) - torch.tensor(pc_range[:3])) / 6
    print(f'min: {points.amin(dim=0)}, max: {points.amax(dim=0)}, mean: {points.mean(dim=0)}, std: {points.std(dim=0)}')
    points = torch.cat([points, torch.randn(points.shape[0], 1)], dim=1)
    t1 = time.time()
    v1, c1, n1 = c_fun(points, resulotion, pc_range, 35, True, 20000)
    t2 = time.time()
    v2, c2, n2 = py_fun(points, resulotion, pc_range, 35, True, 20000)
    t3 = time.time()
    print(f'C use {t2 - t1:.3f}s, Py use: {t3 - t2:.3f}s')
    print('voxel:', v1.shape, v2.shape, '\n', v1.view(-1)[:10], '\n', v2.reshape(-1)[:10],
        'error:', np.abs(v1.numpy() - v2).max())
    print('coors:', c1.shape, c2.shape, '\n', c1.view(-1)[:10], '\n', c2.reshape(-1)[:10],
        'error:', np.abs(c1.numpy() - c2).max())
    print('numbe:', n1.shape, n2.shape, '\n', n1.view(-1)[:10], '\n', n2.reshape(-1)[:10],
        'error:', np.abs(n1.numpy() - n2).max())
    assert np.abs(c1.numpy() - c2).max() == 0
    assert np.abs(n1.numpy() - n2).max() == 0
    assert np.abs(v1.numpy() - v2).max() == 0

    axis_range = torch.tensor(pc_range)
    points[:, :3] = (points[:, :3] - axis_range[:3]) / ((axis_range[3:] - axis_range[:3]) / torch.tensor(resulotion))
    print('range:', points[:, :3].amin(dim=0), points[:, :3].amax(dim=0))
    t1 = time.time()
    v1, c1, n1 = c_fun(points, resulotion, (), 40, False, 10000)
    t2 = time.time()
    v2, c2, n2 = py_fun(points, resulotion, (), 40, False, 10000)
    t3 = time.time()
    print(f'C use {t2 - t1:.3f}s, Py use: {t3 - t2:.3f}s')
    print('voxel:', v1.shape, v2.shape, '\n', v1.view(-1)[:10], '\n', v2.reshape(-1)[:10],
        'error:', np.abs(v1.numpy() - v2).max())
    print('coors:', c1.shape, c2.shape, '\n', c1.view(-1)[:10], '\n', c2.reshape(-1)[:10],
        'error:', np.abs(c1.numpy() - c2).max())
    print('numbe:', n1.shape, n2.shape, '\n', n1.view(-1)[:10], '\n', n2.reshape(-1)[:10],
        'error:', np.abs(n1.numpy() - n2).max())
    assert np.abs(c1.numpy() - c2).max() == 0
    assert np.abs(n1.numpy() - n2).max() == 0
    assert np.abs(v1.numpy() - v2).max() == 0


def test_2():
    from my_ext._C import get_C_function, get_python_function
    import time
    print()
    np.set_printoptions(precision=4, suppress=False, linewidth=200)
    torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
    c_fun = get_C_function('points_to_voxel_mean')
    py_fun = get_python_function('points_to_voxel_mean')

    pc_range = [-10, -10, -10, 10, 10, 10]
    resulotion = [100, 100, 100]
    points = torch.randn(200000, 3) * (torch.tensor(pc_range[3:]) - torch.tensor(pc_range[:3])) / 6
    print(f'min: {points.amin(dim=0)}, max: {points.amax(dim=0)}, mean: {points.mean(dim=0)}, std: {points.std(dim=0)}')
    points = torch.cat([points, torch.randn(points.shape[0], 1)], dim=1)
    t1 = time.time()
    v1, c1 = c_fun(points, resulotion, pc_range, True, 2000)
    t2 = time.time()
    v2, c2 = py_fun(points, resulotion, pc_range, True, 2000)
    t3 = time.time()
    print(f'C use {t2 - t1:.3f}s, Py use: {t3 - t2:.3f}s')
    print('voxel:', v1.shape, v2.shape, '\n', v1.view(-1)[:10], '\n', v2.view(-1)[:10],
        'error:', torch.abs(v1 - v2).max().item())
    print('coors:', c1.shape, c2.shape, '\n', c1.view(-1)[:10], '\n', c2.view(-1)[:10],
        'error:', torch.abs(c1 - c2).max().item())
    assert torch.abs(c1 - c2).max() == 0
    assert torch.abs(v1 - v2).max() < 1e-6

    axis_range = torch.tensor(pc_range)
    points[:, :3] = (points[:, :3] - axis_range[:3]) / ((axis_range[3:] - axis_range[:3]) / torch.tensor(resulotion))
    t1 = time.time()
    v1, c1 = c_fun(points, resulotion, (), False, 10000)
    t2 = time.time()
    v2, c2 = py_fun(points, resulotion, (), False, 10000)
    t3 = time.time()
    print(f'C use {t2 - t1:.3f}s, Py use: {t3 - t2:.3f}s')
    print('voxel:', v1.shape, v2.shape, '\n', v1.view(-1)[:10], '\n', v2.view(-1)[:10],
        'error:', torch.abs(v1 - v2).max().item())
    print('coors:', c1.shape, c2.shape, '\n', c1.view(-1)[:10], '\n', c2.view(-1)[:10],
        'error:', torch.abs(c1 - c2).max().item())
    assert torch.abs(c1 - c2).max() == 0
    assert torch.abs(v1 - v2).max() < 1e-6


def test_3():
    from my_ext._C import get_C_function, get_python_function
    import time
    print()
    np.set_printoptions(precision=4, suppress=False, linewidth=200)
    torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
    c_fun = get_C_function('points_to_voxel_dense')
    py_fun = get_python_function('points_to_voxel_dense')

    pc_range = [-10, -10, -10, 10, 10, 10]
    resulotion = [100, 200, 50]
    # resulotion = [20, 30, 40]
    points = torch.randn(200000, 3) * (torch.tensor(pc_range[3:]) - torch.tensor(pc_range[:3])) / 6
    print(f'min: {points.amin(dim=0)}, max: {points.amax(dim=0)}, mean: {points.mean(dim=0)}, std: {points.std(dim=0)}')
    points = torch.cat([points, torch.randn(points.shape[0], 1)], dim=1)
    t1 = time.time()
    v1 = c_fun(points, resulotion, pc_range)
    t2 = time.time()
    v2 = py_fun(points, resulotion, pc_range)
    t3 = time.time()
    print(f'C use {t2 - t1:.3f}s, Py use: {t3 - t2:.3f}s')
    print('voxel:', v1.shape, v2.shape, '\n', v1.view(-1)[:10], '\n', v2.view(-1)[:10],
        'error:', torch.abs(v1 - v2).max().item())
    diff_index = torch.where(v1.isclose(v2).logical_not())
    if len(diff_index[0]) > 0:
        print(diff_index)
        print(v1[diff_index])
        print(v2[diff_index])
    assert torch.abs(v1 - v2).max() < 1e-6

    axis_range = torch.tensor(pc_range)
    points[:, :3] = (points[:, :3] - axis_range[:3]) / ((axis_range[3:] - axis_range[:3]) / torch.tensor(resulotion))
    t1 = time.time()
    v1 = c_fun(points, resulotion)  # type: Tensor
    t2 = time.time()
    v2 = py_fun(points, resulotion)  # type: Tensor
    t3 = time.time()
    print(f'C use {t2 - t1:.3f}s, Py use: {t3 - t2:.3f}s')
    print('voxel:', v1.shape, v2.shape, '\n', v1.view(-1)[:10], '\n', v2.view(-1)[:10],
        'error:', torch.abs(v1 - v2).max().item())
    diff_index = torch.where(v1.isclose(v2).logical_not())
    if len(diff_index[0]) > 0:
        print(diff_index)
        print(v1[diff_index])
        print(v2[diff_index])
    assert torch.abs(v1 - v2).max() < 1e-6

    from my_ext.structures import Voxel
    vd = Voxel(v1)
    print(vd)
    vs = vd.to_sparse()
    print(vs)
    vdd = vs.to_dense()
    print(vdd)
    assert (vd.data - vdd.data).abs().max() == 0
