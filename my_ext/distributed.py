import argparse
import os
import pickle

import torch
import torch.distributed

from my_ext.config import get_parser
from my_ext.utils import str2list
from my_ext.utils import add_bool_option

__all__ = ['get_rank', "get_world_size", "is_main_process", "options", "make", "synchronize",
           'broadcast', 'reduce_result', 'reduce_tensor',
           'all_gather', 'gather_tensor', 'gather_tensor_with_same_shape', 'gather_tensor_with_different_shape'
           ]


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def options(parser=None):
    group = get_parser(parser).add_argument_group("分布式训练选项")
    group.add_argument("--distributed", action="store_true", default=False, help=argparse.SUPPRESS)
    # help="Use distributed training?")
    group.add_argument("--local_rank", "--local-rank", metavar="N", type=int, default=0, help="GPU id to use.")
    group.add_argument("--dist-backend", metavar="C", default="nccl", choices=["nccl", "gloo", "mpi"],
        help="The backend type [nccl, gloo, mpi].")
    group.add_argument("--dist-method", metavar="S", default="env://", help="The init_method.")
    group.add_argument("--num-gpu", metavar='I', type=int, default=1, help='The number of used GPUs')
    group.add_argument("--gpu-ids", metavar='Is', type=str2list, default=None, help='The id of GPU')
    group.add_argument("--gpu-id", metavar='I', type=int, default=None, help=argparse.SUPPRESS)
    add_bool_option(group, "--dist-apex", default=False, help="use apex.parallel.DistributedDataParallel?")
    add_bool_option(group, "--sync-bn", default=True, help="Change the BN to SyncBN?")
    return group


def make(cfg: argparse.Namespace):
    if cfg.dist_method.startswith('env://'):
        world_size = os.getenv("WORLD_SIZE")
        cfg.num_gpu = int(world_size) if world_size is not None else 1
    assert isinstance(cfg.num_gpu, int) and cfg.num_gpu > 0, f"num_gpu must be specified!"
    cfg.local_rank = int(os.getenv('LOCAL_RANK', cfg.local_rank))
    if cfg.gpu_ids is None:
        cfg.gpu_ids = list(range(cfg.num_gpu))
    else:
        cfg.num_gpu = len(cfg.gpu_ids)
    if cfg.num_gpu > 1 or cfg.gpu_id is None:
        cfg.gpu_id = cfg.gpu_ids[cfg.local_rank]

    device = torch.device('cuda', cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)
    cfg.distributed = torch.distributed.is_available() and cfg.num_gpu > 1
    if cfg.distributed:
        if cfg.dist_method.startswith('env://'):
            torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_method)
        else:
            torch.distributed.init_process_group(
                backend=cfg.dist_backend,
                init_method=cfg.dist_method,
                world_size=cfg.num_gpu,
                rank=cfg.local_rank,
            )
    return device


# def manual_setting(rank, world_size, backend='nccl', dist_url='tcp://127.0.0.1:32145', gpu_id=None):
#     if gpu_id is None:
#         gpu_id = rank
#     device = torch.device('cuda', gpu_id)
#     torch.cuda.set_device(gpu_id)
#     torch.distributed.init_process_group(backend=backend, init_method=dist_url, world_size=world_size, rank=rank)
#     return device


def synchronize():
    if get_world_size() > 1:
        torch.distributed.barrier()
    # if get_world_size() == 1:
    #     return
    # rank = torch.distributed.get_rank()
    #
    # def _send_and_wait(r):
    #     if rank == r:
    #         tensor = torch.tensor(0, device="cuda")
    #     else:
    #         tensor = torch.tensor(1, device="cuda")
    #     torch.distributed.broadcast(tensor, r)
    #     while tensor.item() == 1:
    #         time.sleep(1)
    #
    # _send_and_wait(0)
    # _send_and_wait(1) # now sync on the main process


def broadcast(x: torch.Tensor, src=0) -> torch.Tensor:
    if get_world_size() > 1:
        torch.distributed.broadcast(x, src=src)
    return x


def reduce_tensor(tensor: torch.Tensor, op='sum') -> torch.Tensor:
    if get_world_size() == 1:
        return tensor
    rt = tensor.clone()
    if op == 'sum':
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    elif op == 'mean':
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM) / get_world_size()
    elif op == 'min':
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.MIN)
    elif op == 'max':
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.MAX)
    elif op == 'prod':
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.PRODUCT)
    else:
        raise NotImplementedError
    return rt


@torch.no_grad()
def reduce_result(mean_value: torch.Tensor, batch_size=1):
    if get_world_size() == 1:
        return mean_value, batch_size
    vt = mean_value.detach() * batch_size
    nt = torch.tensor(batch_size, dtype=mean_value.dtype, device=mean_value.device)
    torch.distributed.all_reduce(vt)
    torch.distributed.all_reduce(nt)
    return vt / nt, nt.item()


def all_gather(data):
    """
    Run all_gather on arbitrary pickle-able data (not necessarily tensors)
    Args:
        data: any pickle-able  object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size.item(),)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather_tensor(data: torch.Tensor, dim=0, dst: int = None, is_same_shape=True) -> torch.Tensor:
    """从多个GPU汇集data，并沿维度<dim>连接，如果dst为None则每个GPU都返回连接后的tensor, 否则仅id为dst的GPU返回"""
    if is_same_shape:
        return gather_tensor_with_same_shape(data, dim, dst)
    else:
        return gather_tensor_with_different_shape(data, dim, dst)[0]


def gather_tensor_with_same_shape(data: torch.Tensor, dim=0, dst: int = None) -> torch.Tensor:
    N = get_world_size()
    if N == 1:
        return data
    is_dst = dst is None or dst == get_rank()
    tensor_list = [torch.zeros_like(data) for _ in range(N)] if is_dst else None
    if dst is None:
        torch.distributed.all_gather(tensor_list, data)
    else:
        torch.distributed.gather(data, tensor_list, dst)
    return torch.cat(tensor_list, dim=dim)


def gather_tensor_with_different_shape(data: torch.Tensor, dim=0, dst: int = None):
    N = get_world_size()
    if N == 1:
        return data
    is_dst = dst is None or dst == get_rank()
    ## get the size of tensor along <dim> per gpu
    size = data.new_tensor(data.shape[dim], dtype=torch.long)
    size_list = [size.clone() for _ in range(N)] if is_dst else None
    if dst is None:
        torch.distributed.all_gather(size_list, size)
    else:
        torch.distributed.gather(tensor=size, gather_list=size_list, dst=dst)
    size_list = [size.item() for size in size_list]
    max_size = max(size_list)
    shape = list(data.shape)
    shape[dim] = max_size
    tensor_list = [data.new_empty(shape) for _ in range(N)] if is_dst else None
    ## pad to same shape
    if data.shape[dim] != max_size:
        shape[dim] = max_size - data.shape[dim]
        tensor = torch.cat([data, data.new_zeros(shape)], dim=dim)
    else:
        tensor = data
    if dst is None:
        torch.distributed.all_gather(tensor_list, tensor)
    else:
        torch.distributed.gather(tensor, tensor_list, dst)
    if not is_dst:
        return None
    return torch.cat([x.narrow(dim, 0, n) for n, x in zip(size_list, tensor_list)], dim=dim), size_list


def is_net_port_used(port, ip='127.0.0.1'):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        s.shutdown(2)
        # print('%s:%d is used' % (ip, port))
        return True
    except:
        # print('%s:%d is unused' % (ip, port))
        return False
