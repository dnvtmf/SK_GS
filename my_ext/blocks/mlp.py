from typing import Sequence, Union
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MLP(nn.Module):
    """Multilayer Perceptron"""

    def __init__(self, in_channels, out_channels, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(
                nn.Linear(
                    self.in_channels if l == 0 else self.dim_hidden,
                    self.out_channels if l == num_layers - 1 else self.dim_hidden,
                    bias=bias
                )
            )

        self.net = nn.ModuleList(net)

    def forward(self, x: Tensor):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(in={self.in_channels}, out={self.out_channels}, " \
               f"hidden={self.dim_hidden}, num_layers={self.num_layers})"


class MLP_with_skips(nn.Module):
    def __init__(
        self, in_channels: int, dim_hidden: int, out_channels: Union[int, Sequence[int]] = 0, num_layers: int = 0,
        skips: Sequence[int] = (), bias=True, weight_norm=False, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.skips = tuple(skips)
        self.bias = bias
        self.weight_norm = weight_norm

        net = []
        for i in range(num_layers):
            net.append(nn.Linear(in_channels, self.dim_hidden, bias=bias))
            if weight_norm:
                nn.utils.weight_norm(net[-1])
            in_channels = self.dim_hidden + (self.in_channels if i in self.skips else 0)
        self.net = nn.ModuleList(net)
        if isinstance(out_channels, int):
            self.last = nn.Linear(in_channels, out_channels) if out_channels > 0 else None
        else:
            self.last = nn.ModuleList(nn.Linear(in_channels, oc) for oc in out_channels)
        if weight_norm:
            nn.utils.weight_norm(self.last)

    def forward(self, inputs: Tensor):
        x = inputs
        for i in range(self.num_layers):
            x = self.net[i](x)
            x = F.relu(x, inplace=True)
            if i in self.skips:
                x = torch.cat([x, inputs], dim=-1)

        if isinstance(self.last, nn.ModuleList):
            return [m(x) for m in self.last]
        elif self.last is not None:
            x = self.last(x)
        return x

    def __repr__(self):
        return (f"{self.__class__.__name__}(in={self.in_channels}, out={self.out_channels}, "
                f"hidden={self.dim_hidden}, num_layers={self.num_layers}, skips={self.skips}, bias={self.bias}"
                f"{'weight_norm=True' if self.weight_norm else ''}"
                f")")


try:
    import tinycudann


    class FullyFusedMLP(tinycudann.Network):
        """Lightning fast implementation of small multi-layer perceptrons (MLPs).
        Restricted to hidden layers of size 16, 32, 64, or 128.
        """

        def __init__(
            self, in_channels: int, out_channels: int, dim_hidden: int = 128, num_layers: int = 5, activation='ReLU',
            **kwargs
        ):
            assert dim_hidden in [16, 32, 64, 128]
            assert activation in ['None', 'ReLU', 'LeakyReLU', 'Exponential', 'Sine', 'Sigmoid', 'Softplus', 'Tanh']
            encoding_config = {
                "otype": "FullyFusedMLP",  # Component type.
                "activation": "ReLU",  # Activation of hidden layers.
                "output_activation": "None",  # Activation of the output layer.
                "n_neurons": dim_hidden,  # Neurons in each hidden layer. May only be 16, 32, 64, or 128.
                "n_hidden_layers": num_layers,  # Number of hidden layers.
            }
            super().__init__(in_channels, out_channels, encoding_config, **kwargs)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.dim_hidden = dim_hidden
            self.num_layers = num_layers
            self.activation = activation

        def extra_repr(self):
            return f"in={self.in_channels}, out={self.out_channels}, " \
                   f"hidden={self.dim_hidden}, num_layers={self.num_layers}, activation={self.activation}"

except ImportError:
    class FullyFusedMLP(MLP):
        def __init__(
            self, in_channels: int, out_channels: int, dim_hidden: int = 128, num_layers: int = 5, activation='ReLU',
            **kwargs
        ):
            warnings.warn('please install tinycudann by '
                          '"pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"')
            assert activation == 'ReLU'
            super().__init__(in_channels, out_channels, dim_hidden, num_layers + 1, False)


def test_fully_fused_mlp():
    print()
    m = FullyFusedMLP(16, 16, dim_hidden=16, num_layers=1).cuda()
    # m.params.data[16 * 16:].data.copy_(torch.eye(16, device='cuda').view(-1))
    print(m)
    print([(k, v.shape, v.dtype) for k, v in m.named_parameters()])
    num = (m.in_channels + 15) // 16 * 16 * m.dim_hidden + m.dim_hidden ** 2 * (
        m.num_layers - 1) + m.dim_hidden * ((m.out_channels + 15) // 16 * 16)
    assert num == m.params.numel()
    x = torch.randn(100, m.in_channels).cuda()
    print('input:', x.shape, x.dtype)
    o = m(x)
    print('output:', o.shape, o.dtype)

    m2 = MLP(m.in_channels, m.out_channels, m.dim_hidden, m.num_layers + 1, bias=False).cuda()
    print(m2)
    print([(k, v.shape, v.dtype) for k, v in m2.named_parameters()])
    print(sum(v.numel() for v in m2.parameters()))
    weights = m.params
    i = 0
    for idx, (k, v) in enumerate(m2.named_parameters()):
        A, B = v.shape
        A = (A + 15) // 16 * 16
        B = (B + 15) // 16 * 16
        print(k, v.shape, (A, B))
        # if idx == 0:
        #     v.data.copy_(weights[i:i + v.numel()].view_as(v))
        # else:
        v.data.copy_(weights[i:i + A * B].view(A, B)[:v.shape[0], :v.shape[1]])
        i += A * B
        print(i)
    o2 = m2(x)
    print('error:', (o - o2).abs().max().item())

    # y = o.float()
    # w = torch.inverse(x.T @ x) @ x.T
    # print(w.shape)
    # w = w @ y
    # print((y - x @ w).abs().max().item())
    # print(w.shape)
    # m2.net[0].weight.data.copy_(w.T)
    # o3 = m2(x)
    # print('error:', (o - o3).abs().max().item())
    # print(m.params[:16 * 16].view(16, 16))
