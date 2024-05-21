import numpy as np

import torch
from torch import nn, Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from my_ext._C import get_C_function
from networks.encoders import POSITION_ENCODERS

# from networks.scene_scope import SceneScope

sh_encode_forward = get_C_function('sh_encode_forward')
sh_encode_backward = get_C_function('sh_encode_backward')


class _sh_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # force float32 for better precision
    def forward(ctx, inputs, degree, calc_grad_inputs=False):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        B, input_dim = inputs.shape  # batch size, coord dim
        output_dim = degree ** 2

        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = None

        sh_encode_forward(inputs, outputs, B, input_dim, degree, dy_dx)

        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        inputs, dy_dx = ctx.saved_tensors

        if dy_dx is not None:
            grad = grad.contiguous()
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            return grad_inputs, None, None
        else:
            return None, None, None


sh_encode = _sh_encoder.apply


@POSITION_ENCODERS.register('sphere_harmonics')
@POSITION_ENCODERS.register('SH')
class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
        super().__init__()

        self.input_dim = input_dim  # coord dims, must be 3
        self.degree = degree  # 0 ~ 4
        self.output_dim = degree ** 2

        assert self.input_dim == 3, "SH encoder only support input dim == 3"
        assert 0 < self.degree <= 8, "SH encoder only supports degree in [1, 8]"

    def extra_repr(self):
        return f"input_dim={self.input_dim}, degree={self.degree}"

    def forward(self, direction: Tensor, size: float = 1, scene=None) -> Tensor:
        """
        Args:
            direction: [..., input_dim], normalized direction
            size:
            scene: SceneScope
        Return:
            Tensor: [..., degree^2]
        """
        # direction = direction / size if scene is None else scene.normalize(direction, -1., 1.)  # [-1, 1]
        prefix_shape = list(direction.shape[:-1])
        direction = direction.reshape(-1, self.input_dim)

        outputs = sh_encode(direction, self.degree, direction.requires_grad)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])

        return outputs


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert 4 >= deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5
