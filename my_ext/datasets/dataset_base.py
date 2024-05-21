import functools
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import List, Union
import logging

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset as _Dataset

from my_ext.data_transform import Transform


class Dataset(_Dataset, ABC):

    def __init__(
        self,
        root: Path,
        samples=None,
        class_names: List[str] = None,
        transforms: Transform = None,
        cache_in_memory=False,
        **kwargs
    ):
        """ Dataset基础类
        使用new_list, new_dict来避免内存泄露问题(见 https://github.com/pytorch/pytorch/issues/13246)

        Args:
            root:
            samples:
            class_names:
            transforms:
            cache_in_memory: 将加载的数据数据放入内存中，避免重复读取文件
            **kwargs:
        """
        self.samples = samples
        self.transforms = transforms  # 数据增强操作
        self.root = root  # 数据集根目录

        if class_names is not None:
            self._class_names = class_names
            self.cls2id = {name: idx for idx, name in enumerate(class_names)}
            self.id2cls = {idx: name for idx, name in enumerate(class_names)}
            self._num_classes = len(class_names)
        else:
            self._class_names = None
            self.cls2id = None
            self.id2cls = None
            self._num_classes = None

        if 'split_ratio' in kwargs:
            self._split_dataset(kwargs.pop('split_ratio'), kwargs.pop('split_id', 0), kwargs.pop('split_seed', 42))

        self._cache_dir = kwargs.pop('cache_dir', '')  # 缓存目录
        self._cache_dir = Path(self._cache_dir).expanduser() if self._cache_dir else None
        self._cache_disk_force = kwargs.pop('cache_disk_force', False)  # 强制重新缓存
        if self._cache_dir is not None:
            self._cache_dir.mkdir(exist_ok=True)

        self._cache = self.manager.dict() if cache_in_memory else None
        if cache_in_memory:
            self._preload()
        if len(kwargs) > 0:
            logging.info(f"{self.__class__.__name__} got unused parameters: {list(kwargs.keys())}")

    @property
    def manager(self):
        if not hasattr(self, '_manager'):
            self._manager = mp.Manager()
        return self._manager

    def _preload(self):
        pass

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return self._num_classes

    def set_transforms(self, transforms=None):
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def _split_dataset(self, split_ratio=(1,), split_id=0, split_seed=42):
        """根据比例<split_ratio>将数据集随机划分为若干部分，返回第<split_id>部分"""
        if isinstance(split_seed, int) and split_seed > 0:
            rng = np.random.RandomState(split_seed)
            rng.shuffle(self.samples)  # 随机打乱数据集

        assert 0 <= split_id < len(split_ratio)
        num = len(self.samples)
        sum_ratio = sum(split_ratio)
        start_index = int(num * sum(split_ratio[:split_id]) / sum_ratio)
        end_index = int(num * sum(split_ratio[:split_id + 1]) / sum_ratio)
        self.samples = self.samples[start_index:end_index]

    def __repr__(self):
        s = f"{self.__class__.__name__}{'' if self._cache is None else '[RAM]'}:\n"
        if self._cache_dir is not None:
            s += f"  Cache data in: {self._cache_dir}"
            if self._cache_disk_force:
                s += "[force]"
            s += "\n"
        s += f"  Num Samples: {len(self.samples)}\n"
        if self.num_classes is not None:
            s += f"  Num Categories: {self.num_classes}\n"
        s += f"  Root Location: {self.root}\n"
        es = self.extra_repr()
        if isinstance(es, str):
            es = [es]
        for ss in es:
            if ss:
                s += '  ' + ss + '\n'
        s += "  Transforms: {}\n".format(repr(self.transforms).replace('\n', '\n' + ' ' * 4))
        return s

    def extra_repr(self) -> Union[str, List[str]]:
        return ''

    @staticmethod
    def cache_in_memory(func):
        """缓存得到数据，要求数据是int, float, bool, str, np.ndarray, torch.Tensor,及其用tuple,list, dict形成的组合"""
        func_id = func.__hash__()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cls = args[0]  # type: Dataset
            index = args[1]  # type: int
            # print('get', index, func.__name__)
            if cls._cache is None:
                return func(*args, **kwargs)
            key = (func_id, index)
            # print('key:', key)
            if key not in cls._cache:
                cls._cache[key] = func(*args, **kwargs)
            return deepcopy(cls._cache[key])

        return wrapper

    @staticmethod
    def cache_in_disk(func):
        """缓存数据到磁盘
        要求数据可以使用torch.save存取
        """

        @functools.wraps(func)
        def wrapper(cls: Dataset, index: int, *args, **kwargs):
            if cls._cache_dir is None:
                return func(cls, index, *args, **kwargs)
            filename = f"{cls.__class__.__name__}_{func.__name__}_{index}.cache_data"
            if cls._cache_disk_force or not cls._cache_dir.joinpath(filename).exists():
                data = func(cls, index, *args, **kwargs)
                torch.save(data, cls._cache_dir.joinpath(filename))
            else:
                data = torch.load(cls._cache_dir.joinpath(filename), map_location='cpu')
            return data

        return wrapper

    def new_list(self, *args, **kwargs) -> list:
        return self.manager.list(*args, **kwargs)

    def new_n_list(self, n=0, default=None) -> list:
        return self.manager.list([default for _ in range(n)])

    def new_dict(self, *args, **kwargs) -> dict:
        return self.manager.dict(*args, **kwargs)


def test():
    from torch.utils.data import DataLoader

    print()

    class TestDataset(Dataset):

        def __init__(self, n=100):
            samples = self.new_list(range(n))
            super().__init__(Path('.'), samples, cache_in_memory=True)

        @Dataset.cache_in_memory
        def load_random_tensor(self, index):
            return torch.tensor([index, torch.randn(1).item()])

        def __getitem__(self, item):
            print(self.samples[item])
            return self.load_random_tensor(self.samples[item])

    db = TestDataset(4)
    print('==> Dataset:', db)
    print('db[1]:', db[1])
    print('db(1):', db[1])
    loader = DataLoader(db, batch_size=2)

    for epoch in range(3):
        print('=' * 20, f"epoch {epoch}", '=' * 20)
        for i, data in enumerate(loader):
            print(f'batch {i}', data)
