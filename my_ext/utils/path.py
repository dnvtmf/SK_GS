import shutil
import os
from pathlib import Path

__all__ = ['dir_create_and_clear', 'dir_create_empty', 'str2path']


def dir_create_and_clear(d: Path, *pattern: str, clear=True):
    d.mkdir(exist_ok=True, parents=True)
    if not clear:
        return d
    if len(pattern) == 0:
        pattern = ['*.*']
    for p in pattern:
        for filename in d.glob(p):
            os.remove(filename)
    return d


def dir_create_empty(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(exist_ok=True, parents=True)
    return d


def str2path(p: str):
    """当不为空串时返回Path，否则返回None"""
    return Path(p).expanduser() if p else None
