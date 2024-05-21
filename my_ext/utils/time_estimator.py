import time
from collections import defaultdict

import torch

from my_ext.utils.str_utils import time2str

__all__ = ['TimeEstimator', 'TimeWatcher']


class TimeEstimator:
    """用于估计训练进度"""

    def __init__(self, total=1, momentum=0.99):
        self.last_time = time.time()
        self.total_steps = total
        self.steps = 0
        self.used = 0
        self.total = 0
        self._momentum = momentum
        self.avg = None

    def reset(self, total=1, momentum=0.99):
        self.last_time = time.time()
        self.total_steps = total
        self.steps = 0
        self.used = 0
        self.total = 0
        self._momentum = momentum
        self.avg = None

    def start(self):
        self.last_time = time.time()

    def step(self, repeat=1):
        self.steps += repeat
        self.used = time.time() - self.last_time
        self.last_time = time.time()
        self.total += self.used
        if self.avg is None:
            self.avg = self.used / repeat
        else:
            self.avg = self._momentum * self.avg + (1. - self._momentum) * (self.used / repeat)

    @property
    def value(self):
        return time2str(self.used)

    @property
    def average(self):
        return self.total / max(1e-5, self.steps)

    @property
    def avg_str(self):
        return time2str(self.total / max(1e-5, self.steps))

    @property
    def sum(self):
        return time2str(self.total)

    @property
    def expect(self):
        return time2str((self.total_steps - self.steps) * self.avg)

    @property
    def progress(self):
        return f'Time: {self.sum}/{self.expect}({time2str(self.avg)})'


class TimeWatcher:
    def __init__(self, cuda_sync=True):
        self._last_time = time.time()
        self._total = defaultdict(float)
        self._num = defaultdict(int)
        self._cuda_sync = cuda_sync

    def reset(self, cuda_sync=None):
        self._last_time = time.time()
        self._total = defaultdict(float)
        self._num = defaultdict(int)
        if cuda_sync is not None:
            self._cuda_sync = cuda_sync

    def start(self):
        if self._cuda_sync:
            torch.cuda.synchronize()
        self._last_time = time.time()

    def log(self, name='', n=1):
        if self._cuda_sync:
            torch.cuda.synchronize()
        now = time.time()
        self._num[name] += n
        self._total[name] += now - self._last_time
        self._last_time = now

    def average(self, name=None):
        if name is None:
            return {k: self._total[k] / self._num[k] for k in self._total.keys()}
        else:
            return self._total[name] / self._num[name]

    def sum(self, name=None):
        if name is None:
            return {k: self._total[k] for k in self._total.keys()}
        else:
            return self._total[name]

    def __repr__(self):
        return ', '.join('{}: {}'.format(k, time2str(self._total[k] / self._num[k])) for k in self._total.keys())
