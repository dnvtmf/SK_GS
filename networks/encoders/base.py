from torch import nn, Tensor
from my_ext.utils import Registry
from typing import Type


class PositionEncoder(nn.Module):
    """Position Encdoer"""
    input_dim: int
    output_dim: int  # output dim


POSITION_ENCODERS = Registry()  # type: Registry[Type[PositionEncoder]]


@POSITION_ENCODERS.register('None')
class NonePE(PositionEncoder):
    def __init__(self, input_dim=3, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self, points: Tensor) -> Tensor:
        return points
