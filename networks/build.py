from my_ext.config import get_parser
from my_ext.utils import add_cfg_option, add_registry_option
from networks.gaussian_splatting import NETWORKS, GaussianSplatting


def options(parser=None):
    group = get_parser(parser).add_argument_group('Networks Options for Surface Reconstuction')
    add_registry_option(group, NETWORKS, '-a', '--arch', help=f'The name of architectures')
    add_cfg_option(group, '--arch-cfg', help="The configure for networks")
    return group


def make(cfg, **kwargs) -> GaussianSplatting:
    net = NETWORKS[cfg.arch](**{**cfg.arch_cfg, **kwargs})
    return net
