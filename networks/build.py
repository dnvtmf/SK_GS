from torch import nn

from my_ext.config import get_parser
from my_ext.utils import add_cfg_option, add_registry_option
from networks.base import NERF_NETWORKS, NeRF_Network


def options(parser=None):
    group = get_parser(parser).add_argument_group('Networks Options for Surface Reconstuction')
    add_registry_option(group, NERF_NETWORKS, '-a', '--arch', help=f'The name of architectures')
    add_cfg_option(group, '--arch-cfg', help="The configure for networks")
    return group


def make(cfg, **kwargs) -> NeRF_Network:
    net = NERF_NETWORKS[cfg.arch](**{**cfg.arch_cfg, **kwargs})
    return net
