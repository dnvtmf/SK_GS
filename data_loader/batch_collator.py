from typing import Optional, Dict, Union, Type, Tuple, Callable
import numpy as np

import torch
from torch.utils.data._utils.collate import collate, default_collate_fn_map, default_collate

__all__ = ['default_collate', 'iterable_collate', 'first_collate']

T_fn_map = Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]

default_collate_fn_map[type(None)] = lambda *args, **kwargs: None


# --------------- iterable collate fn -----------------


def collate_tensor_fn_i(batch, *, collate_fn_map: T_fn_map = None):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        # storage = elem.untyped_storage()._new_shared(numel, device=elem.device) # torch 1.x
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)  # torch 2.0.0
        out = elem.new(storage).resize_(sum([x.shape[0] for x in batch]), *elem.shape[1:])
    return torch.cat(batch, dim=0, out=out)


iterable_collate_fn_map = default_collate_fn_map.copy()
iterable_collate_fn_map[torch.Tensor] = collate_tensor_fn_i


def iterable_collate(batch):
    """cat tensors rather than stack
    
    e.g., batch: [torch.randn(3, 4) for _ in range(2)]
    return tensor shape: (6, 4)
    """
    return collate(batch, collate_fn_map=iterable_collate_fn_map)


# --------------- first collate fn ----------------
def collate_tensor_fn_first(batch, *, collate_fn_map: T_fn_map = None):
    elem = batch[0]  # type: torch.Tensor
    if torch.utils.data.get_worker_info() is not None:
        elem = elem.contiguous().share_memory_()
    return elem


def colllate_fn_first(batch, *, collate_fn_map: T_fn_map = None):
    return batch[0]


def collate_numpy_scalar_fn_first(batch, *, collate_fn_map: T_fn_map = None):
    return torch.as_tensor(batch[0])


first_collate_fn_map = default_collate_fn_map.copy()
first_collate_fn_map[torch.Tensor] = collate_tensor_fn_first
first_collate_fn_map[float] = colllate_fn_first
first_collate_fn_map[int] = colllate_fn_first
first_collate_fn_map[str] = colllate_fn_first
first_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn_first


def first_collate(batch):
    """return first element in batch"""
    return collate(batch, collate_fn_map=first_collate_fn_map)
