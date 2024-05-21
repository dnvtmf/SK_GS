from my_ext.utils.registry import Registry
from my_ext.utils import float2str

METRICS = Registry()


class Metric:
    _names = {}

    def reset(self):
        return NotImplemented

    def update(self, *args, **kwargs):
        return NotImplemented

    def summarize(self):
        return NotImplemented

    def get_result(self, name: str):
        assert name in self._names
        f = self._names[name]
        if f is None:
            f = getattr(self, name)
        return f()

    def __getitem__(self, name: str):
        assert isinstance(name, str)
        return self.get_result(name)

    @property
    def names(self):
        return tuple(self._names.keys())

    def add_metric(self, name: str, func=None):
        self._names[name] = func
        setattr(self, name, func)

    def str(self, name=None, fmt=float2str, **kwargs):
        if name is not None:
            return f"{name}={fmt(self[name], **kwargs)}"
        return ', '.join(f"{name}={fmt(self[name], **kwargs)}" for name in self.names)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
