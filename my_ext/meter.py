import math
from collections import defaultdict
from typing import Callable

import torch
from torch.distributed import all_reduce

from my_ext import get_world_size


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.9, reduce=False):
        self.val = math.nan
        self.avg = math.nan
        self.sum = math.nan
        self.ravg = math.nan
        self.count = 0
        self.momentum = momentum
        self.reduce = reduce

    def reset(self):
        self.ravg = math.nan
        self.val = math.nan
        self.avg = math.nan
        self.sum = math.nan
        self.count = 0

    @torch.no_grad()
    def update(self, val, n=1):
        if get_world_size() > 1 and self.reduce:
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val).cuda()
            val = val * n
            all_reduce(val)
            n_tensor = val.new_tensor(n, dtype=torch.int)
            all_reduce(n_tensor)
            n = n_tensor.item()
            val = val / n
            val = val.item()
        else:
            if isinstance(val, torch.Tensor):
                val = val.item()
        self.val = val
        if self.count == 0:
            self.sum = val * n
            self.ravg = self.val
        else:
            self.sum += val * n
            self.ravg = self.momentum * self.ravg + (1.0 - self.momentum) * val
        self.count += n
        self.avg = self.sum / self.count


class DictMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.9, reduce=True, float2str: Callable[[float], str] = '{:6.3f}'.format):
        self.momentum = momentum
        self.data = defaultdict(lambda: defaultdict(float))
        self.count = defaultdict(int)
        self.reudce = reduce
        self._f2s = float2str

    def reset(self):
        self.data = defaultdict(lambda: defaultdict(float))
        self.count = defaultdict(int)

    @torch.no_grad()
    def update(self, val: dict, n=1):
        if get_world_size() > 1 and self.reudce:
            for k, v in val.items():
                v = v * n
                all_reduce(v)
                val[k] = v
            n_tensor = torch.tensor(n, dtype=torch.int).cuda()
            all_reduce(n_tensor)
            n = n_tensor.item()
            for k in val.keys():
                val[k] /= n
        for k, v in val.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.data[k]['val'] = v
            self.data[k]['ravg'] = self.momentum * self.data[k]['ravg'] + (1.0 - self.momentum) * v
            self.data[k]['sum'] += v * n
            self.count[k] += n

    @property
    def value(self):
        return ', '.join(['{}={}'.format(k, self._f2s(self.data[k]['val'])) for k in self.data.keys()])

    @property
    def average(self):
        return ', '.join(['{}={}'.format(k, self._f2s(self.data[k]['sum'] / self.count[k])) for k in self.data.keys()])

    @property
    def running_average(self):
        return ', '.join(['{}={}'.format(k, self._f2s(self.data[k]['ravg'])) for k in self.data.keys()])

    @property
    def sum(self):
        return ', '.join(['{}={}'.format(k, self._f2s(self.data[k]['sum'])) for k in self.data.keys()])

    def get_average(self):
        return {k: self.data[k]['sum'] / self.count[k] for k in self.data.keys()}

    def get_sum(self):
        return {k: self.data[k]['sum'] for k in self.data.keys()}
