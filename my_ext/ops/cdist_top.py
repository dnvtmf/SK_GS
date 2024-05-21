from typing import Callable, Tuple, Optional

import torch
from torch import Tensor

from my_ext._C import get_C_function, have_C_functions

__all__ = ['cdist_top', 'cdist_top_match_label']


def cdist_top_py(x1, x2, largest=False) -> Tuple[Tensor, Tensor]:
    """
    对x1中的每个向量，在x2中找到与它(欧式)距离最近(远)的向量

    Args:
        x1 (Tensor): shape [..., N, C]
        x2 (Tensor): shape [..., M, C]
        largest (bool):
    Returns:
        return the chamfer distance and the indeices of copprate points. shape [..., N]
    """
    assert x1.ndim == x2.ndim and x1.shape[:-2] == x2.shape[:-2]
    assert x1.shape[-1] == x2.shape[-1]
    shape = x1.shape
    if x1.ndim >= 3:
        x1 = x1.flatten(0, -3)
        x2 = x2.flatten(0, -3)
    else:
        assert x1.ndim == 2
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
    distance = torch.cdist(x1, x2)  # shape:B x N x M
    if largest:
        values, indices = distance.max(dim=-1)  # type: Tuple[Tensor, Tensor]
    else:
        values, indices = distance.min(dim=-1)  # type: Tuple[Tensor, Tensor]
    values = values.view(shape[:-1])
    indices = indices.view(shape[:-1])
    return values, indices


if have_C_functions('cdist_top', 'cdist_top_backward'):
    cdist_top_forward = get_C_function('cdist_top')
    cdist_top_backward = get_C_function('cdist_top_backward')


    class _ChamferDistanceFunction(torch.autograd.Function):
        @staticmethod
        def jvp(ctx, *grad_inputs):
            pass

        @staticmethod
        def forward(ctx, *inputs):
            if len(inputs) == 2:
                x1, x2 = inputs
                largest = False
            else:
                x1, x2, largest = inputs
            x1 = x1.contiguous()
            x2 = x2.contiguous()

            distance, index = cdist_top_forward(x1, x2, largest)
            ctx.save_for_backward(x1, x2, distance, index)
            return distance, index

        @staticmethod
        def backward(ctx, *grad_outputs):
            grad_dist = grad_outputs[0].contiguous()
            points1, points2, distance, index = ctx.saved_tensors
            grad_1, grad_2 = cdist_top_backward(points1, points2, distance, index, grad_dist)
            return grad_1, grad_2, None


    cdist_top = _ChamferDistanceFunction.apply  # type: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
else:
    cdist_top = cdist_top_py


def cdist_top_match_label(x1: Tensor, x2: Tensor, label1: Tensor, label2: Tensor, largest=False):
    max_d = max(x1.abs().max(), x2.abs().max()) * 2.  # >最大距离
    x1 = x1 + label1[..., None] * (max_d + 10)
    x2 = x2 + label2[..., None] * (max_d + 10)
    distance, index = cdist_top(x1, x2, largest)
    mask = distance.gt(max_d)
    distance = torch.masked_fill(distance, mask, -1)
    index = torch.masked_fill(index, mask, -1)
    return distance, index


def test():
    print()
    torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
    for largest in [False, True]:
        for is_cuda in [False, True]:
            print(f'============== cuda={is_cuda}, largest={largest} ================')
            B, N, M, C = 100, 1024, 1111, 3
            P1 = torch.randn(B, N, C)
            P2 = torch.randn(B, M, C)
            if is_cuda:
                P1, P2 = P1.cuda(), P2.cuda()

            x = P1.clone()  # type: Tensor
            y = P2.clone()  # type: Tensor
            x.requires_grad_()
            y.requires_grad_()
            print(x.shape, y.shape)
            d1, i1 = cdist_top_py(x, y, largest)
            d1.mean().backward()
            g1x, g1y = x.grad, y.grad
            # print(d1, i1)
            x = P1.clone()  # type: Tensor
            y = P2.clone()  # type: Tensor
            x.requires_grad_()
            y.requires_grad_()
            d2, i2 = cdist_top(x, y, largest)
            d2.mean().backward()
            g2x, g2y = x.grad, y.grad
            # print(d2, i2)
            print('distance error max:', (d2 - d1).abs().max().item())
            print('point1 grad error max:', (g1x - g2x).abs().max().item())
            print('point2 grad error max:', (g1y - g2y).abs().max().item())
            assert (i1 == i2).all()
            assert (d2 - d1).abs().max().item() < 1e-4
            assert (g1x - g2x).abs().max().item() < 1e-6
            assert (g1y - g2y).abs().max().item() < 1e-6


def test_match_label():
    print()
    torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
    B, N, M, C = 2, 10, 11, 3
    x1 = torch.randn(B, N, C)
    x2 = torch.randn(B, M, C)
    l1 = torch.randint(0, 2, (B, N))
    l2 = torch.randint(0, 2, (B, M))

    d1, i1 = cdist_top_match_label(x1, x2, l1, l2)

    dist = torch.cdist(x1, x2)
    mask = l1[:, :, None].eq(l2[:, None, :])
    max_d = dist.max() + 10
    dist = dist + max_d * (1 - mask.float())
    value, index = dist.min(dim=2)
    mask = value >= max_d
    d2 = torch.where(mask, value.new_tensor(-1), value)
    i2 = torch.where(mask, index.new_tensor(-1), index)
    d2 = torch.masked_fill(value, mask, -1)
    i2 = torch.masked_fill(index, mask, -1)
    print(d1, i1)
    print(d2, i2)
    assert (d1 - d2).abs().max() < 1e6, f"{(d1 - d2).abs().max().item()}"
    assert i1.eq(i2).all()


if __name__ == '__main__':
    test()
