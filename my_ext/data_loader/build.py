from typing import Any
import random

from packaging import version
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from my_ext.config import get_parser
from my_ext.utils import add_extend_list_option, add_bool_option
from .batch_collator import *
from .batch_samplers import *


def worker_init(worked_id):
    worker_seed = (torch.initial_seed() + worked_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def options(parser=None):
    group = get_parser(parser).add_argument_group('Options for data loader')
    group.add_argument('--num-workers', default=4, type=int, metavar='I', help="The number of data loader workers")
    add_extend_list_option(group, '-b', '--batch-size', default=[1], num=3, help='The batch size for train/val/test')
    add_bool_option(group, '--pin-memory', default=True, help='Enable pin memory for data loader')
    return group


def make(cfg, dataset, mode='train', batch_sampler: Any = 'default', collate_fn=None, **kwargs):
    kwargs.setdefault('num_workers', cfg.num_workers)
    kwargs.setdefault('pin_memory', cfg.pin_memory)
    if version.parse(torch.__version__) >= version.parse('1.4.0'):
        kwargs.setdefault('worker_init_fn', worker_init)
    # g = torch.Generator()  # 设置样本shuffle随机种子，作为DataLoader的参数
    # g.manual_seed(0)
    # kwargs.setdefault('generator', g)

    assert mode in ['train', 'eval', 'test']
    if mode == 'train':
        batch_size = cfg.batch_size[0]
    elif mode == 'eval':
        batch_size = cfg.batch_size[1]
    else:
        batch_size = cfg.batch_size[2]
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']

    if batch_sampler == 'default':
        if mode == 'train':
            batch_sampler = ShuffleBatchSampler(dataset, batch_size=batch_size)
        else:
            batch_sampler = SequenceBatchSampler(dataset, batch_size=batch_size)
    elif batch_sampler == 'iterable':
        batch_sampler = IterableBatchSampler(
            dataset,
            length=cfg.eval_interval if cfg.eval_interval > 0 else cfg.epochs,
            batch_size=batch_size,
            num_split=1  # kwargs.get('num_split', kwargs['num_workers'])
        )
        if collate_fn is None:
            # collate_fn = getattr(dataset, 'collate', iterable_collate)
            collate_fn = getattr(dataset, 'collate', first_collate)

    if collate_fn is None:
        collate_fn = getattr(dataset, 'collate', default_collate)

    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **kwargs)
    return data_loader