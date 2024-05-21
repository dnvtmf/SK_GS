from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.modules.batchnorm
# try:
#     from extension.quantization import Conv2d
# except ModuleNotFoundError:
from torch.nn import Conv2d


def fuse_conv_bn(conv, bn=None):
    # type: (Union[Conv2d, nn.modules.conv._ConvNd], Optional[nn.modules.batchnorm._NormBase]) -> None
    if bn is None:
        return
    assert isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)
    scale = (bn.running_var.double() + bn.eps).rsqrt()
    if conv.bias is None:
        bias = -bn.running_mean.double() * scale
    else:
        bias = (conv.bias.data.double() - bn.running_mean.double()) * scale
    if bn.affine:
        w = bn.weight.double()
        scale *= w
        bias = bias * w + bn.bias
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)
    nn.init.ones_(bn.running_var)
    nn.init.zeros_(bn.running_mean)

    dtype = conv.weight.dtype
    if hasattr(conv, 'running_scale') and conv.running_scale is not None:
        conv.running_scale = (conv.running_scale * scale.view(-1, 1)).to(dtype=dtype)
    else:
        conv.weight.data = (conv.weight.data * scale.view(-1, 1, 1, 1)).to(dtype=dtype)
    conv.bias = nn.Parameter(bias.to(dtype=dtype))
    return


def test():
    torch.set_default_dtype(torch.float64)
    # N, M, K, S, P = 3, 64, 7, 2, 3
    N, M, K, S, P = 32, 64, 3, 1, 1
    conv = nn.Conv2d(N, M, K, S, P, bias=True)
    bn = nn.BatchNorm2d(M, affine=True)
    bn.eval()
    if bn.affine:
        nn.init.normal_(bn.weight, 1., 0.1)
        nn.init.normal_(bn.bias, 0., 0.1)
    nn.init.normal_(bn.running_var, 1., 0.1)
    nn.init.normal_(bn.running_mean, 0., 0.1)

    x = torch.randn(3, N, 16, 16)
    y = bn(conv(x))
    fuse_conv_bn(conv, bn)
    z = conv(x)
    print('max error', (y - z).abs().max().item())
    assert torch.allclose(y, z), (y - z).abs().max().item()
    z = bn(conv(x))
    assert torch.allclose(y, z), (y - z).abs().max().item()
