# base on torch.util.data.ConcatDatset
import bisect
from typing import List, Iterable

from .dataset_base import Dataset


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], transforms=None) -> None:
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            assert isinstance(d, Dataset), f"ConcatDataset does not support Such Dataset: {d.__class__.__name__}"
            assert d.class_names == self.datasets[0].class_names
        self.cumulative_sizes = self.cumsum(self.datasets)
        super(ConcatDataset, self).__init__(self.datasets[0].root, None, self.datasets[0].class_names, transforms)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __repr__(self):
        s = self.__class__.__name__ + ':\n  '
        for d in self.datasets:
            s += repr(d).replace('\n', '\n  ')
        return s

    def set_transforms(self, transforms=None):
        for d in self.datasets:
            d.set_transforms(transforms)

    def set_data_filter(self, data_filter):
        for db in self.datasets:
            db.set_data_filter(data_filter)
