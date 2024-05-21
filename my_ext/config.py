import argparse
import os
import warnings
from pathlib import Path
from typing import Union, Optional
import logging

import yaml

from my_ext.utils import change_each, set_config_value, add_path_option

_parser = None
_config = None


def get_parser(parser=None, *args, **kwargs):
    """ 首先使用提供的 parser, 若为 None 则使用默认的全局 parser """
    if parser is not None:
        return parser
    global _parser
    if _parser is None:
        kwargs['formatter_class'] = lambda prog: argparse.HelpFormatter(prog, width=120, max_help_position=30)
        _parser = argparse.ArgumentParser(*args, **kwargs)
    return _parser


def get_config():
    return _config


def get_cfg(key, default=None):
    if not hasattr(_config, key):
        warnings.warn("\033[41mNo such {} item in config, use default {}\033[0m".format(key, default))
    return getattr(_config, key, default)


def set_cfg(key, value):
    setattr(_config, key, value)
    return getattr(_config, key)


def _update_cfg(cfg, new_cfg):
    if isinstance(cfg, dict) and isinstance(new_cfg, dict) and not new_cfg.pop('__replace__', False):
        for k, v in new_cfg.items():
            if k in cfg:
                cfg[k] = _update_cfg(cfg[k], v)
            else:
                cfg[k] = v
        return cfg
    else:
        return new_cfg


def _load_from_yaml(filename):
    if not filename:
        return {}

    logging.warning(f"Load config from yaml file: {filename}")
    with open(filename, "r", encoding='utf-8') as f:
        # yaml_cfg = yaml.load(f)  # if yaml.__version__ < '5.1'
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    if '__base__' in yaml_cfg:
        paths = yaml_cfg.pop('__base__')  # type: list
        cfg = {}
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(filename), path)
            cfg = _update_cfg(cfg, _load_from_yaml(path))
        cfg = _update_cfg(cfg, yaml_cfg)
        return cfg
    else:
        return yaml_cfg


def _load_from_pth(filename, cfg=argparse.Namespace()):
    from my_ext.checkpoint import CheckpointManager

    exclude_key = ['local_rank', 'dist_backend', 'dist_method', 'num_gpu', 'gpu_ids', 'gpu_id']
    CheckpointManager.load_checkpoint(filename)
    _cfg = CheckpointManager.resume('cfg', None)
    if isinstance(_cfg, argparse.Namespace):
        for k, v in _cfg.__dict__.items():
            if k in exclude_key:
                continue
            setattr(cfg, k, v)
    return cfg


def make(args=None, ignore_unknown=False, ignore_warning=False, parser=None):
    global _config
    parser = get_parser(parser)
    cfg = parser.parse_args(args)
    yaml_file = getattr(cfg, "yaml", None)
    resume_file = getattr(cfg, "resume", None)

    # cfg = parser.parse_args([])
    cfg = _load_from_pth(resume_file, cfg)
    yaml_cfg = _load_from_yaml(yaml_file)
    for k, v in yaml_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            if ignore_unknown:
                if not ignore_warning:
                    warnings.warn("Unknown option {}={}".format(k, v))
            else:
                raise AttributeError("Unknown option {}={}".format(k, v))
    _config = parser.parse_args(args, cfg)
    for k, v in vars(_config).items():
        set_config_value(_config, k, v)
    return _config


def options(parser=None):
    group = get_parser(parser).add_argument_group(
        "Config Options. Load order <resume> <yaml> <cmd-line>. For the same option, only the last one work.")
    add_path_option(group, "-c", "--yaml", default="", help="Use YAML file to config.")
    return group


def save(cfg, path: Optional[Union[str, Path]] = None, remove_keys: list = None, sort_keys=True):
    def before_save(x):
        if isinstance(x, Path):
            return x.as_posix()
        elif isinstance(x, tuple):
            return list(x)
        else:
            return x

    cfg = vars(cfg)
    cfg = change_each(cfg, before_save)

    if remove_keys is None:
        remove_keys = []
    remove_keys.extend(['yaml', 'num_gpu', 'distributed', 'gpu_id', 'gpu_ids', 'local_rank'])
    for name in set(remove_keys):
        if name in cfg:
            cfg.pop(name)
    if path is None:
        return yaml.safe_dump(cfg, default_flow_style=None, sort_keys=sort_keys, allow_unicode=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=sort_keys, allow_unicode=True)

    return None
