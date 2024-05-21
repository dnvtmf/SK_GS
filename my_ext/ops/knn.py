"""use package faiss, install: conda install faiss-gpu -c conda-forge"""

from typing import Tuple

import torch
from torch import Tensor
import numpy as np
import faiss
import faiss.contrib.torch_utils

from my_ext._C import get_C_function


def swig_ptr_from_FloatTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(x.untyped_storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_IndicesTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_idx_t_ptr(x.untyped_storage().data_ptr() + x.storage_offset() * 8)


faiss.contrib.torch_utils.swig_ptr_from_FloatTensor = swig_ptr_from_FloatTensor
faiss.contrib.torch_utils.swig_ptr_from_IndicesTensor = swig_ptr_from_IndicesTensor


def knn_brute(value: Tensor, query: Tensor, k=10):
    """暴力方式计算KNN, dist=sqrt(x*x+y*y+...,)
    Args:
        value: shape [N, D]
        query: shape [M, D] or None
        k: k nearest neighbor
     Returns:
        (Tensor, Tensor): distances, shape [M, k]; indices, shape [M, k]
    """
    dist = torch.cdist(query, value)
    return torch.topk(dist, k=k, dim=-1, largest=False)


def knn_np(value: np.ndarray, query: np.ndarray, k=1, use_gpu=False) -> (np.ndarray, np.ndarray):
    assert value.ndim == 2 and query.ndim == 2

    # init index
    index = faiss.IndexFlatL2(value.shape[1])
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)

    # Add point cloud to index
    index.add(value)

    # Get NN for each point in data
    D, I = index.search(query, k)
    return D, I


def knn(value: Tensor, query: Tensor = None, k=1, force_cpu=True) -> Tuple[Tensor, Tensor]:
    """对每一个query在value中寻找k最近邻, dist=x*x+y*y+...
    Args:
        value: shape [N, D]
        query: shape [M, D] or None
        k: k nearest neighbor
        force_cpu: force use cpu to compuate knn
    Returns:
        (Tensor, Tensor): distances, shape [M, k]; indices, shape [M, k]
    """

    if query is None:
        query = value
    value, query = value.contiguous().to(torch.float32), query.contiguous().to(torch.float32)
    device = value.device
    if value.is_cuda and not force_cpu:
        # NOTE: knn_gpu may produce incorrect results
        res = faiss.StandardGpuResources()
        # quantizer = faiss.GpuIndexFlatL2(res, value.shape[-1])
        # index = faiss.GpuIndexIVFFlat(res, quantizer, value.shape[-1], 100)
        # index.train(value)
        # index.add(value)
        # return index.search(query, k)
        return faiss.knn_gpu(res, query, value, k)
        # return get_C_function('faiss_knn')(value, query, k)

    index = faiss.IndexFlatL2(query.shape[1])
    index.add(value.cpu())
    D, I = index.search(query.cpu(), k)
    return D.to(device), I.to(device)


def test():
    from my_ext import utils
    timer = utils.TimeWatcher()
    print()
    utils.set_printoptions()
    d = 3
    nb = int(1e4)
    nq = int(100)
    k = 4
    xb = torch.randn(nb, d, dtype=torch.float32)
    xq = torch.randn(nq, d, dtype=torch.float32)
    print("inputs:", utils.show_shape(xb, xq))
    xb_cuda = xb.cuda()
    xq_cuda = xq.cuda()
    for step in range(10):
        timer.start()
        cpu_d, cpu_i = knn(xb, xq, k)
        timer.log('cpu')
        gpu_d, gpu_i = knn(xb_cuda, xq_cuda, k, force_cpu=False)
        timer.log('cuda')
        np_d, np_i = knn_np(xb.numpy(), xq.numpy(), k)
        timer.log('numpy')
        br_d, br_i = knn_brute(xb_cuda, xq_cuda, k)
        timer.log('brute')
        if step == 0:
            timer.reset()
            print(f'inputs: value={xb.shape}, query={xq.shape},k={k}, outputs={cpu_d.shape}')
            print('cpu vs numpy', np.abs(cpu_d.numpy() - np_d).max())
            print('cpu vs cuda', (cpu_d - gpu_d.cpu()).abs().max())
            print('cpu vs brute', (cpu_d - br_d.pow(2).cpu()).abs().max())
    print('time:', timer)


def test_my_knn_udpate():
    from my_ext import utils
    timer = utils.TimeWatcher()
    print()
    utils.set_printoptions()
    N = int(1e5)
    K = 20
    points = torch.randn(N, 3, dtype=torch.float32)
    gt_d, gt_i = knn(points, points, K)
    # NOTE: 当K越大,效果越好
    # 当K较小时, 如K=3时,  a, b, c 可能恰好是彼此的K-近邻, 当points更新后 d比b,c离a更近, 但检测不出来
    # 原因: K-近邻图不联通
    # 可能的改善方法: 随机抽取一些3次近邻
    my_knn_update = get_C_function('my_knn_update')
    index = gt_i.cuda().int()
    dist2 = gt_d.cuda().float()
    print('index, dist2:', index.shape, dist2.shape, index[0])
    noise = torch.randn_like(points) * 0.1
    points = points + noise
    timer.reset()
    timer.start()
    gt_d, gt_i = knn(points, points, K)
    timer.log('cpu')
    points = points.cuda()
    for i in range(20):
        timer.start()
        my_knn_update(index, dist2, points)
        timer.log('my_knn')
        print(f'step {i}, diff: {(dist2.cpu() - gt_d).mean()}')
    print('use time: ', timer)
    dist_cmp = torch.square(points[:, None, :] - points[index]).sum(-1)
    print(dist_cmp.shape, dist2.shape)
    print('check dist2:', (dist2 - dist_cmp).abs().max())
    print('pints[0]', points[0])
    print('gt:', gt_i[0], gt_d[0])
    print('my:', index[0], dist2[0])
    print(index[index[0]])


if __name__ == '__main__':
    test_my_knn_udpate()
