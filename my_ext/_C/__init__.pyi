from typing import Callable, Optional, Union, overload


@overload
def try_use_C_extension(py_func: Callable) -> Callable: ...


@overload
def try_use_C_extension(c_func: Callable, *depends: str) -> Callable: ...


def get_C_function(func: Union[str, Callable]) -> Optional[Callable]: ...


def get_python_function(func: str) -> Callable: ...


def have_C_functions(*names: str) -> bool: ...


def check_C_runtime(name: str, relative=False, eps=1e-5): ...
