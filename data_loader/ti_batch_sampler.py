import logging

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler
from my_ext.distributed import get_world_size, get_rank, broadcast

from datasets.base import NERF_Base_Dataset


class TimeIncrementalBatchSampler(Sampler):
    """逐渐增加采样的范围"""

    def __init__(
        self,
        data_source: NERF_Base_Dataset,  # 数据集
        batch_size,
        stages,  # [[num_steps, start, end, method]]
        length=-1,
        seed=None,
        sample_last_prob=0,
        **kwargs
    ):
        super().__init__(None)
        self.db = data_source
        if hasattr(data_source, "time_ids") and hasattr(data_source, 'num_frames'):
            self.time_ids = data_source.time_ids
            if isinstance(self.time_ids, Tensor):
                self.time_ids = self.time_ids.numpy()
            self.num_frames = data_source.num_frames
        else:
            raise RuntimeError('data_source must have time_ids and num_frames')
        self.length = length
        assert self.length > 0
        self.batch_size = batch_size
        self.total = len(data_source)
        self.rank = get_rank()
        self.num_replicas = get_world_size()
        self.used_size = self.length * self.batch_size
        if seed is None and self.num_replicas > 1:
            seed = np.random.randint(65536)
            seed = broadcast(torch.tensor([seed], device='cuda'), 0).item()
        self._random_generator = np.random.RandomState(seed=seed)
        self.stages = stages
        last_tid_range = None
        for num_steps, tid_start, tid_end, method in self.stages:
            assert num_steps >= 0
            if last_tid_range is None:
                last_tid_range = (tid_start, tid_end)
            else:
                assert last_tid_range[0] >= tid_start and last_tid_range[1] <= tid_end
                last_tid_range = (tid_start, tid_end)
            assert method in ['fix', 'step']
        self.sample_last_prob = sample_last_prob
        if len(kwargs) > 0:
            logging.warning(f'{self.__class__.__name__} Unused args: {list(kwargs.keys())}')

    def __iter__(self):
        total_steps = 0
        last_stage_range = (0, 1)
        indices = None
        for num_steps, tid_start, tid_end, method in self.stages:
            if method == 'fix':
                indices = np.where(np.logical_and(self.time_ids >= tid_start, self.time_ids < tid_end))[0]
            last_range = last_stage_range
            for i in range(num_steps):
                if method == 'step':
                    progress = i / num_steps
                    cur_start = int(last_stage_range[0] * (1 - progress) + tid_start * progress)
                    cur_end = int(last_stage_range[1] * (1 - progress) + tid_end * progress)
                    if indices is None or last_range != (cur_start, cur_end):
                        indices = np.where(np.logical_and(self.time_ids >= cur_start, self.time_ids < cur_end))[0]
                        last_range = (cur_start, cur_end)
                batch_index = self._random_generator.choice(indices, self.batch_size, replace=True)
                if self.sample_last_prob > 0:
                    last_index = self._random_generator.choice(last_range, self.batch_size, replace=True)
                    batch_index = np.where(
                        self._random_generator.rand(self.batch_size) > self.sample_last_prob, batch_index, last_index)
                    print('last ', last_range)
                yield batch_index[self.rank::self.num_replicas].tolist()
                total_steps += 1
                if total_steps > self.length > 0:
                    break
            last_stage_range = (tid_start, tid_end)
        for i in range(total_steps, self.length if self.length > 0 else int(1e9)):
            batch_index = self._random_generator.choice(self.total, self.batch_size, replace=True)
            yield batch_index[self.rank::self.num_replicas].tolist()
        return

    def __len__(self):
        return int(1e9) if self.length <= 0 else self.length

    def __repr__(self):
        s = [
            f"batch_size={self.batch_size}",
            f"length={self.length}",
        ]
        for num_steps, tid_start, tid_end, method in self.stages:
            s.append(f" {method}[{num_steps}]@[{tid_start}, {tid_end})")
        return f"{self.__class__.__name__}({', '.join(s)})"


def test():
    class TestDataset(NERF_Base_Dataset):
        def __init__(self):
            self.num_frames = 100
            self.num_cameras = 20
            self.time_ids = np.arange(self.num_frames).repeat(self.num_cameras)
            print(self.time_ids.shape)
            super().__init__(None, self.time_ids)

    db = TestDataset()
    sampler = TimeIncrementalBatchSampler(db, 1, stages=[
        [100, 0, 1, 'fix'],
        [100, 0, 50, 'step'],
        [200, 0, 100, 'fix']
    ], length=1000, asfsdf=34)
    print(sampler)
    for step, batch_indices in enumerate(sampler):
        pass


if __name__ == '__main__':
    test()
