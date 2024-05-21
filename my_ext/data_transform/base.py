from typing import Union, Optional, TypeVar, Sequence

import numpy as np

from my_ext.structures import Structure2D, Structure3D

_OT = TypeVar('_OT', Structure2D, Structure3D)

__all__ = ['Transform', 'Compose', 'Repeat', 'Identity', 'Transform3D', 'OneOf']


class Transform(object):
    num_input_samples = 1
    num_output_samples = 1

    def __call__(self, *inputs: Union[_OT, Sequence[_OT]]) -> Union[_OT, Sequence[_OT], Sequence[Sequence[_OT]]]:
        """

        :param inputs: 输入若干组实例，每组实例包含Image、BoxList等多个结构
        :return: 输出对应的处理后的实例组
        """
        outputs = []
        # assert len(inputs) % self.num_input_samples == 0
        for i in range(0, len(inputs), self.num_input_samples):
            if self.num_input_samples == 1:
                out = self.transform(*inputs[i])
            else:
                out = self.transform(*inputs[i:i + self.num_input_samples])
            if self.num_output_samples == 1:
                outputs.append(out)
            else:
                outputs.extend(out)
        return outputs

    def __repr__(self):
        format_str = self.__class__.__name__ + "("
        format_str += ", ".join("{}={}".format(k, v) for k, v in self.__dict__.items() if not k.startswith('_'))
        format_str += ")"
        return format_str

    def transform(self, *inputs):
        #  type: (Union[_OT, Sequence[_OT]])-> Sequence[Union[_OT, Sequence[_OT]]]
        """
        若num_input_samples=1，则输入为若干结构，否则为若干组结构，
        若num_output_samples=1，则输出为若干结构，否则为若干组结构
        """
        raise NotImplementedError

    def select(self, inputs, *types):
        outputs = [[] for _ in range(len(types))]
        for i, item in enumerate(inputs):
            for k, type_ in enumerate(types):
                if isinstance(item, type_):
                    outputs[k].append((item, i))
        # for i, type_ in enumerate(types):
        #     assert len(outputs[i]) > 0, f"No type '{type_.__name__}' in inputs"
        return outputs[0] if len(types) == 1 else outputs


class Transform3D(Transform):
    def transform(self, *inputs):
        # type: (Union[Structure3D, Sequence[Structure3D]])-> Sequence[Union[Structure3D, Sequence[Structure3D]]]
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, *transforms: Union[Optional[Transform], Sequence[Optional[Transform]]]):
        if len(transforms) == 1 and isinstance(transforms[0], (tuple, list)):
            self.transforms = [t for t in transforms[0] if t is not None]  # type: Sequence[Transform]
        else:
            self.transforms = [t for t in transforms if t is not None]  # type: Sequence[Transform]
        for t in self.transforms:
            self.num_input_samples *= t.num_input_samples

    def __call__(self, *inputs):
        assert len(inputs) % self.num_input_samples == 0
        has_group = isinstance(inputs[0], (tuple, list))
        if not has_group:
            inputs = [inputs]
        inputs = [[item for item in group if item is not None] for group in inputs]  # filter None
        for t in self.transforms:
            inputs = t(*inputs)
        if not has_group:
            inputs = inputs[0]
        return inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for i, t in enumerate(self.transforms):
            format_string += f"\n  [{i}] " + repr(t).replace('\n', '\n  ')
        format_string += "\n)"
        return format_string

    def transform(self, *inputs):  # should not call this
        return inputs


class Repeat(Transform):
    def __init__(self, num_repeat=1):
        self.num_output_samples = num_repeat
        assert num_repeat >= 1

    def transform(self, *inputs):
        outputs = [inputs]
        for i in range(1, self.num_output_samples):
            outputs.append([item.clone() for item in inputs])
        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}(num_repeat={self.num_output_samples})"


class Identity(Transform):
    def transform(self, *inputs):
        return inputs


class OneOf(Transform):
    """从多个transform中随机选择一个"""

    def __init__(self, *args: Transform, p: Union[float, Sequence[float]] = None):
        self.transforms = list(args)
        if p is None:
            self.p = p
        else:
            p = list(p) if isinstance(p, Sequence) else [p] * len(args)
            assert len(p) == len(args) and sum(p) <= 1 and all(0 <= pi <= 1. for pi in p)
            self.p = p
            p_r = 1 - sum(p)
            if p_r > 0:  # 总和不为1时，添加Identity
                self.transforms.append(Identity())
                self.p.append(p_r)

    def transform(self, *inputs):
        t = np.random.choice(self.transforms, p=self.p)
        return t.transform(*inputs)

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p}"
        for i, t in enumerate(self.transforms):
            s += f'\n  ({i}) ' + repr(t).replace('\n', '\n  ')
        s += '\n)'
        return s
