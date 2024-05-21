from typing import List, Tuple, Union

import torch
from torch import Tensor


class TensorSequence:
    """多个不定长的Tensor组成的序列"""

    def __init__(self, tensors: Union[List[Tensor], Tuple[Tensor, ...]], dim=0):
        assert isinstance(tensors, (list, tuple)), "tensors is tuple"
        self._dim = dim
        self._num = len(tensors)
        self._prefix_sum = [0]
        self._length = 0
        if self._num > 0:
            self.tensors = torch.cat(tensors, dim=dim)
            index = []
            for i in range(self._num):
                n = tensors[i].size(self._dim)
                self._length += n
                index.extend([i] * n)
                self._prefix_sum.append(self._length)
            self.index = torch.tensor(index, dtype=torch.long, device=self.tensors.device)
        else:
            self.tensors = None
            self.index = None
        self._iter_step = 0

    @property
    def dim(self):
        return self._dim

    @property
    def num(self):
        return self._num

    @property
    def length(self):
        return self._length

    def __len__(self):
        return self._num

    def __getitem__(self, item):
        if isinstance(item, (int, bool)):
            choose = [item if item >= 0 else item + self._num]
        elif isinstance(item, slice):
            choose = list(range(*item.indices(self._num)))
        else:
            choose = list(item)
        tensors = [self.tensors[self._prefix_sum[i]:self._prefix_sum[i + 1]] for i in choose]
        return TensorSequence(tensors, dim=self.dim)

    def at(self, index: int) -> Tensor:
        if index < 0:
            index += self._num
        return self.tensors[self._prefix_sum[index]:self._prefix_sum[index + 1]]

    def append(self, other):
        # type: (Union[Tensor, TensorSequence])->TensorSequence
        if isinstance(other, Tensor):
            n = other.size(self._dim)
            if self.tensors is None:
                self.tensors = other
                self.index = torch.full((n,), 0, device=other.device, dtype=torch.long)
            else:
                self.tensors = torch.cat([self.tensors, other], dim=self._dim)
                self.index = torch.cat([self.index, self.index.new_full((n,), self._num)], dim=0)
            self._num += 1
            self._length += n
            self._prefix_sum.append(self._length)
        elif isinstance(other, TensorSequence):
            if other.num == 0:
                return self
            elif self.num == 0:
                self.tensors = other.tensors
                self.index = other.index
                self._num = other._num
                self._length = other._length
                self._prefix_sum = other._prefix_sum.copy()
                self._dim = other._dim
            else:
                assert self._dim == other._dim
                self._prefix_sum.extend(v + self._length for v in other._prefix_sum[1:])
                self.tensors = torch.cat([self.tensors, other.tensors], dim=self._dim)
                self.index = torch.cat([self.index, other.index + self._num], dim=0)
                self._num += other._num
                self._length += self._length
        else:
            raise RuntimeError()
        return self

    def extend(self, *tensors):
        if len(tensors) == 1 and isinstance(tensors, (tuple, list)):
            tensors = tensors[0]
        new_index = [self.index] if self._num > 0 else []
        new_tensors = [self.tensors] if self._num > 0 else []
        for x in tensors:
            if isinstance(x, Tensor):
                n = x.size(self._dim)
                new_index.append(self.index.new_full((n,), self._num))
                new_tensors.append(x)
                self._num += 1
                self._length += n
                self._prefix_sum.append(self._length)
            elif isinstance(x, TensorSequence) and x._num > 0:
                self._prefix_sum.extend(n + self._length for n in x._prefix_sum)
                new_tensors.append(x.tensors)
                new_index.append(x.index + self._num)
                self._length += x._length
                self._num += x._num
        self.index = torch.cat(new_index, dim=0)
        self.tensors = torch.cat(new_tensors, dim=self._dim)
        return self

    def copy(self):
        result = TensorSequence([])
        result._num = self._num
        result._dim = self._dim
        result._prefix_sum = self._prefix_sum.copy()
        result._length = self._length
        result.tensors = self.tensors
        result.index = self.index
        return result

    def __add__(self, other):
        result = self.copy()
        if isinstance(other, (Tensor, TensorSequence)):
            result.append(other)
        else:
            result.extend(other)
        return result

    def __iter__(self):
        self._iter_step = 0
        return self

    def __next__(self):
        if self._iter_step >= self._num:
            raise StopIteration
        data = self.at(self._iter_step)
        self._iter_step += 1
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(num={self._num}, dim={self._dim}, size={tuple(self.tensors.shape)}, " \
               f"dtype={self.tensors.dtype})"

    def to(self, *args, **kwargs):
        if self._num > 0:
            self.tensors = self.tensors.to(*args, **kwargs)
            self.index = self.index.to(*args, **kwargs)
        return self

    def get_range(self, index: int):
        """The index range [start, end) of i-th tensor"""
        return self._prefix_sum[index], self._prefix_sum[index + 1]


def test():
    print()
    tensors = [torch.randn(n, 3) for n in [3, 0, 2, 4]]
    ts = TensorSequence(tensors)
    print(ts.num, ts.length, ts.dim, ts)
    print(ts[1])
    print(ts[-1])
    print(ts[0:2])
    print(ts[:2])
    print(ts[-2:])
    print(ts[::-1])
    print(ts[:-2])
    print(ts[0, 1, 2])
    print(ts[(0, 1, 2)])
    print(ts.index)

    ts.append(tensors[0])
    ts.extend(tensors)
    ts.extend(*tensors)

    for x in ts:
        print(x)
