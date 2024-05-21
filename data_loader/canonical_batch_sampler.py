import numpy as np
import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler
from my_ext.distributed import get_world_size, get_rank, broadcast

from datasets.base import NERF_Base_Dataset


class CanonicalBatchSampler(Sampler):
    """逐渐减少标准态时间的比例， 直至所有时间均匀抽取"""

    def __init__(
        self,
        data_source: NERF_Base_Dataset,  # 数据集
        batch_size,
        canonical_time_id,  # 初始采样的时间数
        step_start,  # 开始增加采样的step
        step_end,  # 开始全部样本的step
        prob_srart=1.0,
        prob_end=0.0,
        length=-1,
        seed=None,
        **kwargs
    ):
        super().__init__(None)
        self.db = data_source
        if hasattr(data_source, "time_ids") and hasattr(data_source, 'num_frames'):
            self.time_ids = data_source.time_ids
            if isinstance(self.time_ids, Tensor):
                self.time_ids = self.time_ids.numpy()
            self.num_frames = data_source.num_frames
            assert 0 <= canonical_time_id < self.num_frames
        else:
            raise RuntimeError('data_source must have time_ids and num_frames')
        self.canonical_time_id = canonical_time_id
        self.can_index = np.where(self.time_ids == canonical_time_id)[0]
        print(self.can_index.shape)
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
        self.step_start = step_start
        self.step_end = step_end
        self.prob_srart = prob_srart
        self.prob_end = prob_end
        if len(kwargs) > 0:
            print('Unused args:', kwargs)

    def __iter__(self):
        for i in range(self.length if self.length > 0 else int(1e9)):
            progress = min(max((i - self.step_start) / (self.step_end - self.step_start), 0), 1)
            prob = self.prob_srart * progress + self.prob_end * (1 - progress)
            if i % 100 == 0:
                print('batch sampler prob={}'.format(prob))
            if self._random_generator.rand() > prob:
                batch_index = self._random_generator.choice(self.can_index, size=self.batch_size, replace=True)
            else:
                batch_index = self._random_generator.choice(self.total, size=self.batch_size, replace=True)
            yield batch_index[self.rank::self.num_replicas].tolist()
        return

    def __len__(self):
        return int(1e9) if self.length <= 0 else self.length

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"batch_size={self.batch_size}, " \
               f"length={self.length}, " \
               f"canonical_time_id={self.canonical_time_id}, num_frames={self.num_frames}, " \
               f"vary steps=[{self.step_start}, {self.step_end}]" \
               f")"
