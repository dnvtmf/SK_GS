import numpy as np

import torch
from torch import nn, Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from my_ext._C import get_C_function

from networks.encoders import POSITION_ENCODERS

freq_encode_forward = get_C_function('freq_encode_forward')
freq_encode_backward = get_C_function('freq_encode_backward')


class _freq_encoder(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # force float32 for better precision
    def forward(ctx, inputs, degree, output_dim):
        # inputs: [B, input_dim], float
        # RETURN: [B, F], float

        if not inputs.is_cuda:
            inputs = inputs.cuda()
        inputs = inputs.contiguous()

        B, input_dim = inputs.shape  # batch size, coord dim

        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        freq_encode_forward(inputs, B, input_dim, degree, output_dim, outputs)

        ctx.save_for_backward(inputs, outputs)
        ctx.dims = [B, input_dim, degree, output_dim]

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        grad = grad.contiguous()
        inputs, outputs = ctx.saved_tensors
        B, input_dim, degree, output_dim = ctx.dims

        grad_inputs = torch.zeros_like(inputs)
        freq_encode_backward(grad, outputs, B, input_dim, degree, output_dim, grad_inputs)

        return grad_inputs, None, None


freq_encode = _freq_encoder.apply


@POSITION_ENCODERS.register('freq')
@POSITION_ENCODERS.register('frequency')
class FreqEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = input_dim + input_dim * 2 * degree

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, degree={self.degree}, output_dim={self.output_dim}"

    def forward(self, inputs, **kwargs):
        # inputs: [..., input_dim]
        # return: [..., ]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)

        outputs = freq_encode(inputs, self.degree, self.output_dim)

        outputs = outputs.reshape(prefix_shape + [self.output_dim])

        return outputs


@POSITION_ENCODERS.register('freq_torch')
@POSITION_ENCODERS.register('frequency_torch')
class FreqEncoder_torch(nn.Module):

    def __init__(
        self,
        input_dim,
        degree=4,  # number of frequency
        max_freq_log2=None,
        log_sampling=True,
        include_input=True,
        scale=1.,
        periodic_fns=(torch.sin, torch.cos)
    ):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        max_freq_log2 = degree - 1 if max_freq_log2 is None else max_freq_log2
        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim
        if scale == 'pi':
            scale = torch.pi

        self.output_dim += self.input_dim * degree * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, degree) * scale
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, degree) * scale

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):
        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


@POSITION_ENCODERS.register('freq_var')
@POSITION_ENCODERS.register('frequency_var')
class FreqEncoder_var(nn.Module):
    """随时间变换的FreqEncoder

    Reference: 
    1. BARF: Bundle-Adjusting Neural Radiance Fields (ICCV 2021)
    2. FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization
    """

    def __init__(
        self,
        input_dim,
        L=4,
        max_freq_log2: int = None,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
        scale=torch.pi,
        barf_c2f=(0.2, 0.5),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.L = L
        max_freq_log2 = L - 1 if max_freq_log2 is None else max_freq_log2

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * self.L * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, self.L) * scale  # 2^k pi x
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, self.L) * scale

        self.freq_bands = self.freq_bands.numpy().tolist()
        self._progress = 1.0
        self.barf_c2f = barf_c2f

    def forward(self, input: Tensor, **kwargs):
        out = []
        if self.include_input:
            out.append(input)

        weights = self.get_barf_weight(input, self._progress)
        for i in range(self.L):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq) * weights[i])

        out = torch.cat(out, dim=-1)

        return out

    def get_barf_weight(self, x: Tensor, progress=1.) -> Tensor:
        start, end = self.barf_c2f
        alpha = (progress - start) / (end - start) * self.L  # [0, L]
        k = torch.arange(self.L, dtype=x.dtype, device=x.device)
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(torch.pi).cos_()) / 2
        return weight

    def get_free_nerf_weight(self, x: Tensor, progress=1.) -> Tensor:
        """linearly increasing"""
        k = torch.arange(self.L, dtype=x.dtype, device=x.device)
        return (self.L * progress - k).clamp_(min=0, max=1)

    def change_with_training_progress(self, step=0, num_steps=1, epoch=0, num_epochs=1):
        total_step = epoch * num_steps + step
        self._progress = total_step / (num_epochs * num_steps)
        return

    def extra_repr(self) -> str:
        s = f"L={self.L}, barf_c2f={self.barf_c2f}"
        if not self.include_input:
            s += f", include_input=False"
        return s
