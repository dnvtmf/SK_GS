from typing import Tuple

from torch import nn, Tensor


class Light(nn.Module):

    def forward(self, points: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return NotImplemented