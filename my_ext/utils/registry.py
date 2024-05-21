class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary and provides a register functions.

    Eg. creating a registry:
        some_registry = Registry({"default": default_module})

    There are two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
        some_registry["foo_module"] = foo
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, ignore_case=True, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        self.ignore_case = ignore_case

    def register(self, module_name=None, module=None):
        # used as function call
        if module is not None:
            self._register_generic(module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            if module_name is None:
                self._register_generic(fn.__name__, fn)
            else:
                self._register_generic(module_name, fn)
            return fn

        return register_fn

    def _register_generic(self, module_name: str, module):
        self[module_name.lower() if self.ignore_case else module_name] = module

    def __getitem__(self, name: str):
        try:
            return super().__getitem__(name.lower() if self.ignore_case else name)
        except KeyError as e:
            raise KeyError(f"Support {list(self.keys())} rather than '{name}'" + (
                '(ingore case)' if self.ignore_case else ''))

    def __setitem__(self, key, value):
        return super().__setitem__(key.lower() if self.ignore_case else key, value)

    def __contains__(self, name: str):
        return super().__contains__(name.lower() if self.ignore_case else name)
