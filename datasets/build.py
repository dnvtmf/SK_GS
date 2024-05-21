from my_ext import datasets, utils
from datasets.base import NERF_DATASETS, NERF_DATASET_STYLE, NERF_Base_Dataset


def options(parser=None):
    group = datasets.options(NERF_DATASETS, NERF_DATASET_STYLE, parser)
    group.add_argument('--scene', default='', help='The name of scene')
    utils.add_bool_option(group, '--weighted-sample', default=False)
    return group


def make(cfg, mode='train', **kwargs) -> NERF_Base_Dataset:
    kwargs.setdefault('scene', cfg.scene)
    kwargs.setdefault('weighted_sample', cfg.weighted_sample and mode == 'train')
    return datasets.make(NERF_DATASETS, NERF_DATASET_STYLE, cfg, mode=mode, **kwargs)
