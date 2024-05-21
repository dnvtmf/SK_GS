import warnings
from typing import Callable, Union

from torch import Tensor

_py_functions = {}


def get_python_function(func: str):
    return _py_functions[func]


try:
    from . import _C


    def try_use_C_extension(func: Callable, *depends: str):
        if len(depends) == 0:
            _py_functions[func.__name__] = func
            if hasattr(_C, func.__name__):
                return getattr(_C, func.__name__)
            else:
                warnings.warn(f'No such function in C/CPP/CUDA extension: {func.__name__}')
                return func
        else:
            have_all_depends = True
            for name in depends:
                if not hasattr(_C, name):
                    have_all_depends = False
                    warnings.warn(f'No such function in C/CPP/CUDA extension: {name}')
                    break

            def wrapper(py_func: Callable) -> Callable:
                _py_functions[py_func.__name__] = py_func
                return func if have_all_depends else py_func

            return wrapper


    def get_C_function(func: Union[str, Callable]):
        return getattr(_C, func, None) if isinstance(func, str) else func


    def have_C_functions(*names):
        for name in names:
            if not hasattr(_C, name):
                return False
        return True


    def check_C_runtime(name, relative=False, eps=1e-5, use_assert=False):
        from my_ext.utils import show_shape
        from copy import deepcopy
        import logging

        def error(x, y):
            if use_assert:
                assert type(x) == type(y), f"type is different: {type(x)}, {type(y)}"
            if isinstance(x, (tuple, list)):
                outputs = []
                assert len(x) == len(y), f"size is different: {len(x)} vs {len(y)}"
                for xi, yi in zip(x, y):
                    outputs.append(error(xi, yi))
                return outputs
            elif isinstance(x, dict):
                outputs = {}
                for k in x.keys():
                    assert k in y
                    outputs[k] = error(x[k], y[k])
                return outputs
            elif isinstance(x, Tensor):
                if relative:
                    err = (x - y).abs() / x.abs().max().clamp(eps)
                else:
                    err = (x - y).abs()
                err = err.max().item()
                if use_assert:
                    assert err <= eps, f"Error {err} <= eps={eps}"
                return err
            else:
                if use_assert:
                    assert x == y
                return x == y

        def wrapper(cuda_func):
            def check_wrapper(*args, **kwargs):
                logging.debug(f'intpus: {show_shape(*args, kwargs)}')
                py_out = _py_functions[name](*deepcopy(args), **deepcopy(kwargs))
                cuda_out = cuda_func(*args, **kwargs)
                logging.debug(f'py outputs: {show_shape(py_out)}')
                logging.debug(f'cuda outputs: {show_shape(cuda_out)}')
                logging.debug(f'error: {error(py_out, cuda_out)}')
                # mask = (py_out - cuda_out).abs() > 1e-5
                # print(torch.where(mask), py_out[mask], cuda_out[mask])
                return cuda_out

            return check_wrapper

        return wrapper
except ImportError as e:
    warnings.warn(f'Please Compile C/CPP/CUDA code to use some functions. {e}')


    def try_use_C_extension(func, *depends):
        if len(depends) == 0:
            _py_functions[func.__name__] = func
            return func
        else:
            def warpper(py_func):
                _py_functions[func.__name__] = py_func
                return py_func

            return warpper


    def get_C_function(func: Union[str, Callable]):
        warnings.warn(f'Please Compile CPP/CUDA code get function: {func if isinstance(func, str) else func.__name__}')
        return None


    def have_C_functions(*names):
        return False


    def check_C_runtime(name):
        return lambda x: x
__all__ = ['try_use_C_extension', 'get_C_function', 'get_python_function']
