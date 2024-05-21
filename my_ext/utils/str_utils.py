import math
from typing import Union
from pathlib import Path


def str2num(s: str) -> Union[int, float, bool, str]:
    s = s.strip()
    try:
        value = int(s)
    except ValueError:
        try:
            value = float(s)
        except ValueError:
            if s == 'True' or s == 'true' or s == 'TRUE':
                value = True
            elif s == 'False' or s == 'false' or s == 'FALSE':
                value = False
            elif s == 'None' or s == 'none':
                value = None
            else:
                value = s
    return value


def str2bool(v) -> bool:
    if not isinstance(v, str):
        return bool(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Can not covert "{v}" to string')


def str2dict(s) -> dict:
    if s is None:
        return {}
    if not isinstance(s, str):
        return s
    s = s.split(',')
    d = {}
    for ss in s:
        ss = ss.strip()
        if ss == '':
            continue
        ss = ss.split('=')
        assert len(ss) == 2
        key = ss[0].strip()
        value = str2num(ss[1])
        d[key] = value
    return d


def str2list(s: str) -> list:
    return [str2num(ss) for ss in s.split(',') if ss.strip()]


def str2tuple(s: str) -> tuple:
    return tuple(str2num(ss) for ss in s.split(',') if ss.strip())


def str2vt(s: str) -> Union[int, float, str, bool, None, tuple]:
    """返回一个值或一个tuple"""
    res = [str2num(ss) for ss in s.split(',') if ss.strip()]
    return res if len(res) > 1 else res[0]


def str2path(p: str):
    return Path(p).expanduser()  # .absolute()


def eval_str(s: str):
    """将s安全的转换为结构化数据,
        {} [a=b] (a=b) a:b  转换为dict
        [a, b] (a, b)       转换为list
        's' "s"             为字符串, 保持不变
        d+                  转换为数字
        [0-9]*.[0-9]*       转换为浮点数
        true True TRUE      转换为True
        false False FALSE   转换为False
        none None           转换为None
        其他                 转换为字符串
        ,                   用于分割不同的部分
        空格 回车 Tab         将会被忽略
    """
    n = len(s)
    # 分割字符串
    splits = []
    types = []
    new_start = True
    i = 0
    while i < n:
        if s[i] in [' ', '\t', '\n']:
            pass
        elif s[i] == ',':
            new_start = True
        elif s[i] in '{}[]()=:':
            splits.append(s[i])
            types.append(0)
            new_start = True
        else:
            if s[i] == '\'' or s[i] == '\"':
                j = i + 1
                while j < n:
                    if s[j] == s[i]:
                        break
                    j += 1
                if j == n:
                    raise ValueError("Error Input String!!")
                if new_start:
                    splits.append(s[i + 1:j])
                    types.append(2)
                else:
                    splits[-1] += s[i + 1:j]
                    types[-1] = 2
                i = j
            else:
                if new_start:
                    splits.append(s[i])
                    types.append(1)
                else:
                    splits[-1] += s[i]
            new_start = False
        i += 1

    # 转换列表
    def _convert(values: list):
        if any(w[0] for w in values):
            if len(values) % 2 == 1 or any(w[0] != (k % 2 == 1) for k, w in enumerate(values)):
                raise ValueError("Error Input String!! 字典不正确")
            return {values[k][1]: values[k + 1][1] for k in range(0, len(values), 2)}
        else:
            return [w[1] for w in values]

    # 组成结构体
    stacks = [[]]
    is_dict = False
    for t, v in zip(types, splits):
        if t == 0 and v in '=:':
            if is_dict:
                raise ValueError("Error Input String!! 字典不正确")
            is_dict = True
            continue
        if t == 0:
            if v in '{[(':
                stacks[-1].append((is_dict, v))
                stacks.append([])
            elif v in '}])':
                v = {'}': '{', ']': '[', ')': '('}[v]
                if len(stacks) == 1 or stacks[-2][-1][1] != v:
                    raise ValueError("Error Input String!! 括号不匹配")
                v = stacks.pop(-1)
                stacks[-1][-1] = (stacks[-1][-1][0], _convert(v))
        else:
            stacks[-1].append((is_dict, str2num(v) if t == 1 else v))
        is_dict = False
    if len(stacks) != 1:
        raise ValueError("Error Input String!!")
    result = _convert(stacks[0])
    return result[0] if isinstance(result, list) and len(result) == 1 else result


def float2str(x: float, width: int = 6, threshold=1e-3, precision: int = None) -> str:
    """将浮点数转为固定宽度的字符串

    Args:
        x: 输入的数
        width: 固定宽度 (must >= 6)
        threshold: 小于此阈值的时候用科学计数法表示
        precision: 不为None时 小数点后的数字最多有precision个，如 ' 1.234', '123.45', 否则随机变化，如'1.2345', '123.45'
    """
    assert width >= 6
    x = float(x)
    if (threshold <= abs(x) <= 10 ** width) or math.isinf(x) or math.isnan(x):
        fmt = f'{{0:{width}.{width if precision is None else precision}f}}'
        s = fmt.format(x)[:width]
    else:  # if x < threshold or x >= 10 ** width:  # 科学计数法e.g. 1.2345e-12 or 1.234567e2
        fmt = f'{{0:{width}.{width}e}}'
        s = fmt.format(x)
        left, right = s.split('e')
        right = str(int(right))
        s = f"{left[:width - 1 - len(right)]}e{right}"
    return s


def time2str(second: Union[int, float]) -> str:
    days = int(second / 3600 / 24)
    second = second - days * 3600 * 24
    hours = int(second / 3600)
    second = second - hours * 3600
    minutes = int(second / 60)
    second = second - minutes * 60
    seconds = int(second)
    millisecond = 1000 * (second - seconds)

    if days > 0:
        return '{:2d}D{:02}h'.format(days, hours)
    elif hours > 0:
        return "{:2d}h{:02d}m".format(hours, minutes)
    elif minutes > 0:
        return "{:2d}m{:02d}s".format(minutes, seconds)
    elif seconds > 0:
        return "{:2d}s{:03d}".format(seconds, int(millisecond))
    else:
        return '{:.4f}'.format(millisecond)[:4] + "ms"


def test_eval_str():
    print(eval_str('a'))
    print(eval_str('a=b'))
    print(eval_str('true,True,TRUE,false, False, FALSE, none, None'))
    print(eval_str('1, 1.2, 1e-3'))
    print(eval_str('{a=b, c: 3, d: {e=f}}'))
    print(eval_str('[[1,2,3],4],5'))
    print(eval_str('a n, 123'))
    print(eval_str('123"456"'))
