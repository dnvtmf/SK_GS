from typing import Any, Sequence, TypeVar
import argparse
from pathlib import Path

from my_ext.utils.str_utils import str2bool, str2list, eval_str, str2dict
from my_ext.utils.utils import extend_list, n_tuple
from my_ext.utils.path import str2path
from my_ext.utils.registry import Registry

__all__ = [
    'set_config_value', 'add_bool_option', 'add_registry_option', 'add_path_option', 'add_n_tuple_option',
    'add_extend_list_option', 'add_cfg_option', 'add_choose_option', 'add_dict_option', 'add_default_tuple_option'
]
T = TypeVar('T', argparse.ArgumentParser, argparse._ArgumentGroup)  # noqa
_preprocess_before_use = {
    'path': {},
    'list': {},
    'n_tuple': {},
    'registy': {},
    'choose': {},
    'default_tuple': {},
}


def add_bool_option(parser: T, *name_or_flags: str, default=False, help='', **kwargs: Any):
    parser.add_argument(
        *name_or_flags, nargs='?', const=not default, default=default, type=str2bool, metavar='B', help=help, **kwargs
    )


def add_registry_option(
    parser: T,
    registry: Registry,
    *name_or_flags: str,
    default=None,
    optional=False,  # 可选项
    help='',
    **kwargs
):
    names = list(registry.keys())
    if optional:
        names.append('')
    else:
        assert len(names) > 0, f"Get an Empty Registy"
    names = sorted(names)
    if default is None:
        default = names[0] if len(names) > 0 else None
    else:
        assert default in names, f"Default must in Registry {names}, not {default}"
    action = parser.add_argument(
        *name_or_flags, default=default, choices=names, type=str, metavar='C', help=f'{help} {names}', **kwargs
    )
    _preprocess_before_use['registy'][action.dest] = optional


def add_path_option(
    parser: T,
    *name_or_flags: str,
    default='',
    help='',
    is_dir=False,  # The result of this option must be an existing directory
    is_file=False,  # The result of this option must be an existing file
    exists=False,  # The result of this option must exist
    must=False,  # 是否必需提供路径
    **kwargs
):
    action = parser.add_argument(*name_or_flags, default=default, type=str2path, metavar='P', help=help, **kwargs)
    _preprocess_before_use['path'][action.dest] = (is_dir, is_file, exists, must)


def add_extend_list_option(parser: T, *name_or_flags: str, default: Sequence[Any] = (), help='', num=0, **kwargs):
    """当num>0, 重复列表中最后一个元素，直至数量达到num"""
    if num <= 0 < len(default):
        num = len(default)
    assert num > 0
    action = parser.add_argument(*name_or_flags, default=default, type=str2list, metavar='L', help=help, **kwargs)
    _preprocess_before_use['list'][action.dest] = num


def add_n_tuple_option(parser: T, *name_or_flags: str, default: Sequence[Any] = (), help='', num=0, **kwargs):
    """输入为一个元素则被复制成有num个元素的列表"""
    if num <= 0 < len(default):
        num = len(default)
    else:
        assert len(default) == num
    action = parser.add_argument(*name_or_flags, default=default, type=str2list, metavar='L', help=help, **kwargs)
    _preprocess_before_use['n_tuple'][action.dest] = num


def add_default_tuple_option(parser: T, *name_or_flags: str, default: Sequence[Any] = (), help='', **kwargs):
    """fill the tuple by default value"""
    action = parser.add_argument(*name_or_flags, default=default, type=str2list, metavar='L', help=help, **kwargs)
    _preprocess_before_use['default_tuple'][action.dest] = default


def add_cfg_option(parser: T, *name_or_flags: str, default: Any = None, help='', **kwargs):
    parser.add_argument(
        *name_or_flags, default={} if default is None else default, type=eval_str, metavar='E', help=help, **kwargs
    )


def add_choose_option(parser: T, *name_or_flags: str, default=None, choices=(), help='', **kwargs):
    assert len(choices) > 0
    if default is None:
        default = choices[0]
    action = parser.add_argument(
        *name_or_flags, default=default, choices=choices, metavar='C', type=type(choices[0]), help=help, **kwargs
    )
    _preprocess_before_use['choose'][action.dest] = choices


def add_dict_option(parser: T, *name_or_flags: str, default: dict = None, help='', **kwargs):
    parser.add_argument(
        *name_or_flags, default={} if default is None else default, type=str2dict, metavar='D', help=help, **kwargs
    )


def set_config_value(cfg, key, value):
    if key in _preprocess_before_use['path']:
        is_dir, is_file, exists, must = _preprocess_before_use['path'][key]
        if isinstance(value, str):
            value = str2path(value) if value else None
        assert (not must) or (value is not None), f"option {key} must be an path"
        assert value is None or isinstance(value, Path), f"option {key} must be None or Path not {type(value)}"
        if value is not None:
            assert (not is_dir) or value.is_dir(), f"Directory '{value}' for option {key} does not exist!"
            assert (not is_file) or value.is_file(), f"File '{value}' for option {key} does not exist!"
            assert (not exists) or value.exists(), f"Path '{value}' for option {key} does not exist!"
    elif key in _preprocess_before_use['list']:
        value = list(value) if issubclass(type(value), Sequence) else [value]
        value = extend_list(value, _preprocess_before_use['list'][key])
    elif key in _preprocess_before_use['n_tuple']:
        value = n_tuple(value, _preprocess_before_use['n_tuple'][key])
    elif key in _preprocess_before_use['default_tuple']:
        value = list(value) if issubclass(type(value), Sequence) else [value]
        default = _preprocess_before_use['default_tuple'][key]
        assert len(value) <= len(default)
        value = (*value, *default[len(value):])
    elif key in _preprocess_before_use['registy']:
        if value == '' or value is None:
            optional = _preprocess_before_use['registy']
            assert optional, f"option {key} can not be None"
            value = None
    elif key in _preprocess_before_use['choose']:
        assert value in _preprocess_before_use['choose'][key], f"error config for {key}"
    setattr(cfg, key, value)
    return
