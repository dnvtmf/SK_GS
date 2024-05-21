from typing import TypeVar, Optional, MutableMapping, Callable

T = TypeVar('T')
T2 = TypeVar('T2')


class Registry(MutableMapping[str, T]):
    ignore_case: bool

    def __init__(self, ignore_case=True, *args, **kwargs): ...

    def register(self, module_name: str = None, module: Optional[T2] = None) -> Callable[[T2], T2]: ...

    def _register_generic(self, module_name: str, module: T): ...

    def __getitem__(self, name: str) -> T: ...

    def __setitem__(self, key: str, value: T): ...
