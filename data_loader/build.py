import my_ext.data_loader.build as ext_build
from my_ext import utils, my_logger

from .ti_batch_sampler import TimeIncrementalBatchSampler
from .canonical_batch_sampler import CanonicalBatchSampler


def options(parser):
    group = ext_build.options(parser)
    group.add_argument('--data-sampler', default='')
    utils.add_cfg_option(parser, '--data-sampler-cfg', default=None,
        help='Time Incremental sampler cfg')
    return group


def make(cfg, dataset, mode='train', batch_sampler='default', collate_fn=None, **kwargs):
    logger = my_logger.get_logger()
    if mode == 'train':
        batch_size = cfg.batch_size[0]
    elif mode == 'eval':
        batch_size = cfg.batch_size[1]
    else:
        batch_size = cfg.batch_size[2]
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']

    batch_sampler = cfg.data_sampler if mode == 'train' and cfg.data_sampler else batch_sampler
    if batch_sampler == 'ti_inc':
        batch_sampler = TimeIncrementalBatchSampler(
            dataset, batch_size=batch_size, length=cfg.epochs, **cfg.data_sampler_cfg)
    elif batch_sampler == 'canonical':
        batch_sampler = CanonicalBatchSampler(
            dataset, batch_size=batch_size, length=cfg.epochs, **cfg.data_sampler_cfg)
    else:
        assert batch_sampler in ['default', 'iterable']
    logger.info(f'use BatchSampler: {batch_sampler} when mode={mode}')
    return ext_build.make(cfg, dataset, mode, batch_sampler, collate_fn, **kwargs)
