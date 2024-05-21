import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from my_ext.distributed import get_world_size, get_rank, broadcast

__all__ = ['SequenceBatchSampler', 'ShuffleBatchSampler', 'IterableBatchSampler']


class SequenceBatchSampler(Sampler):
    """
    顺序组织batch, 并保证尽可能迭代次数一致(除非 样本数<batch_size or batch_size=1)
    假设10个数据, batch_size=3, num_gpus=2, 则组织为
    gpu0: [[0, 1, 2], [3, 4]]
    gpu1: [[5, 6, 7], [8, 9]]
    """

    def __init__(self, data_source, batch_size, drop_last=False, pad_last=False, **kwargs):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_replicas = get_world_size()
        self.rank = get_rank()
        self.total = len(data_source)
        self.drop_last = drop_last
        self.pad_last = pad_last
        if drop_last:
            self.length = self.total // (self.batch_size * self.num_replicas)
            self.used_size = self.length * self.batch_size
        elif pad_last:
            self.length = (self.total - 1) // (self.batch_size * self.num_replicas) + 1
            self.used_size = self.length * self.batch_size
        else:
            self.length = ((self.total - 1) // self.batch_size) // self.num_replicas + 1
            assert self.total <= self.batch_size * self.num_replicas * self.length
            start_index = self.rank * self.total // self.num_replicas
            end_index = (self.rank + 1) * self.total // self.num_replicas
            self.used_size = end_index - start_index
        if len(kwargs) > 0:
            print('Unused args:', kwargs)

    def __iter__(self):
        if self.drop_last:
            for i in range(self.length):
                i = i * self.batch_size + self.rank * self.batch_size * self.length
                yield list(range(i, i + self.batch_size))
        elif self.pad_last:
            for i in range(self.length):
                i = i * self.batch_size + self.rank * self.batch_size * self.length
                yield [min(j, self.total - 1) for j in range(i, i + self.batch_size)]
        else:
            start_index = self.rank * self.total // self.num_replicas
            end_index = (self.rank + 1) * self.total // self.num_replicas
            num = end_index - start_index
            num_each_step = [1] * self.length
            assert num >= self.length
            num -= self.length
            for i in range(self.length):
                if num > self.batch_size - 1:
                    num_each_step[i] = self.batch_size
                    num -= self.batch_size - 1
                else:
                    num_each_step[i] += num
                    break
            index = start_index
            for num in num_each_step:
                yield list(range(index, index + num))
                index += num

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"batch_size={self.batch_size}, " \
               f"num_gpus={self.num_replicas}, " \
               f"length={self.length}, " \
               f"drop_last={self.drop_last}, " \
               f"num_samples={self.total}" \
               f")"


class ShuffleBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, seed=None, **kwargs):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_replicas = get_world_size()
        self.rank = get_rank()
        self.total = len(data_source)
        self.length = self.total // self.batch_size // self.num_replicas
        self.used_size = self.length * self.batch_size

        if seed is None and self.num_replicas > 1:
            seed = np.random.randint(65536)
            seed = broadcast(torch.tensor([seed], device='cuda'), 0).item()
        self._random_generator = np.random.RandomState(seed=seed)
        if len(kwargs) > 0:
            print('Unused args:', kwargs)

    def __iter__(self):
        indices = self._random_generator.permutation(len(self.data_source))
        for i in range(self.length):
            i = (i * self.num_replicas + self.rank) * self.batch_size
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"batch_size={self.batch_size}, " \
               f"num_gpus={self.num_replicas}, " \
               f"length={self.length}, " \
               f"num_samples={self.total}" \
               f")"


class IterableBatchSampler(Sampler):
    """无限迭代的取样器，按照num_split, 返回(step, sample_size)"""

    def __init__(self, data_source, batch_size, length=-1, num_split=0, **kwargs) -> None:
        super().__init__()
        self.data_source = data_source
        self.length = length
        self.batch_size = batch_size
        self.num_split = max(num_split, 1)
        self.total = len(data_source)
        self.used_size = self.length * self.batch_size
        if len(kwargs) > 0:
            print('Unused args:', kwargs)
        assert self.batch_size >= self.num_split

    def __iter__(self):
        avg_size = self.batch_size // self.num_split
        last_size = self.batch_size % self.num_split + avg_size

        if self.length > 0:
            for i in range(self.length if self.length > 0 else int(1e9)):
                yield [(i, avg_size if j + 1 < self.num_split else last_size) for j in range(self.num_split)]
            return

    def __len__(self):
        return int(1e9) if self.length <= 0 else self.length

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"batch_size={self.batch_size}, " \
               f"length={self.length}, " \
               f"num_split={self.num_split}" \
               f")"
