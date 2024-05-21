import logging
import math
from bisect import bisect_right
from typing import List, Union, Dict, Any

from matplotlib.pyplot import step
from torch.optim.optimizer import Optimizer

from my_ext.config import get_parser
from my_ext.utils import add_cfg_option, float2str, Registry, n_tuple

_lr_methods = Registry(ignore_case=True)


class BaseLR:

    def __init__(self, steps: int, start_lr: float, end_lr: float, *args, **kwarg) -> None:
        pass

    def __call__(self, step: int) -> float:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"


@_lr_methods.register('fix')
class FixLR(BaseLR):

    def __init__(self, steps, start_lr, end_lr):
        self.lr = end_lr

    def __call__(self, step):
        return self.lr


@_lr_methods.register('step')
class StepLR(BaseLR):

    def __init__(self, steps, start_lr, end_lr):
        self.gamma = (end_lr - start_lr) / steps if steps > 0 else 0
        self.lr = start_lr

    def __call__(self, step):
        return self.lr + self.gamma * step


@_lr_methods.register('exp')
class ExpLR(BaseLR):

    def __init__(self, steps, start_lr, end_lr):
        self.lr = start_lr
        self.gamma = (end_lr / start_lr) ** (1 / steps) if steps > 0 else 0

    def __call__(self, step):
        return self.lr * self.gamma ** step


@_lr_methods.register('exp2')
class Exp2LR(BaseLR):
    # Exponential falloff
    # Equal to ExpLR????

    def __init__(self, steps, start_lr, end_lr, base=10.):
        self.lr = start_lr
        self.base = float(base)
        self.gamma = math.log(max(end_lr / start_lr, 1e-12), self.base) / steps

    def __call__(self, step):
        return self.lr * self.base ** (step * self.gamma)

    def __repr__(self):
        return f"{self.__class__.__name__}(base={self.base})"


@_lr_methods.register('ploy')
class PloyLR(BaseLR):

    def __init__(self, steps, start_lr, end_lr, power=0.9):
        self.lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.steps = steps

    def __call__(self, step):
        return (self.lr - self.end_lr) * (1 - step / self.steps) ** self.power + self.end_lr

    def __repr__(self):
        return f"{self.__class__.__name__}(power={self.power})"


@_lr_methods.register('cos')
class CosineLR(BaseLR):

    def __init__(self, steps, start_lr, end_lr):
        self.lr = end_lr
        self.gamma = 0.5 * (start_lr - end_lr)
        self.beta = math.pi / steps if steps > 0 else 0

    def __call__(self, step):
        return self.lr + self.gamma * (1 + math.cos(step * self.beta))


@_lr_methods.register('tri')
class TriangleLR(BaseLR):

    def __init__(self, steps, start_lr, end_lr):
        self.lr = start_lr
        self.half_step = (steps - 1) * 0.5
        self.gamma = (end_lr - start_lr) / self.half_step if steps > 0 else 0

    def __call__(self, step):
        return self.lr + math.fabs(self.half_step - step) * self.gamma


@_lr_methods.register('log_lerp')
class LogLineayLR(BaseLR):
    """Interpolate log-linearly from `start_lr` (t=0) to `end_lr` (t=1)."""

    def __init__(self, steps, start_lr, end_lr):
        assert start_lr > 0 and end_lr > 0
        self.start_lr = math.log(start_lr)
        self.end_lr = math.log(end_lr)
        self.steps = steps

    def __call__(self, step):
        return math.exp(min(max(step / self.steps, 0), 1) * (self.end_lr - self.start_lr) + self.start_lr)


@_lr_methods.register('log_lerp_delay')
class DelayLogLineayLR(BaseLR):
    """Interpolate log-linearly from `start_lr` (t=0) to `end_lr` (t=1)."""

    def __init__(self, steps, start_lr, end_lr):
        assert start_lr > 0 and end_lr > 0
        self.start_lr = math.log(start_lr)
        self.end_lr = math.log(end_lr)
        self.steps = steps
        self.lr_delay_mult = 0.1

    def __call__(self, step):
        delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * math.sin(0.5 * math.pi * step / self.steps)
        return delay_rate * math.exp(min(max(step / self.steps, 0), 1) * (self.end_lr - self.start_lr) + self.start_lr)


class LRScheduler(object):

    def __init__(
        self,
        optimizers: List[Optimizer],
        schedulers: List[Dict[str, Any]],
        steps_per_epoch=1,
        lr=1.0e-3,
    ):
        self._optimizers = optimizers
        self.num_steps = steps_per_epoch

        momentum = []

        if any('momentum' in x for x in schedulers):
            momentum = [n_tuple(x.get('momentum', 1.), 2) for x in schedulers]

        for optimizer in optimizers:
            if not isinstance(optimizer, Optimizer):
                raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                if momentum:
                    if 'momentum' in group:
                        group.setdefault('initial_momentum', group['momentum'])
                    elif 'betas' in group:
                        group.setdefault('initial_momentum', group['betas'][0])
                    else:
                        logging.warning(f'The optimizer {type(optimizer).__name__} do not have momentum or betas')

        num = len(schedulers)
        self._schedulers = schedulers
        self._methods = []
        self._starts = [0]
        self._momentum_change = []
        self.base_lr = lr

        for i in range(num):
            assert schedulers[i]['method'] in _lr_methods, f"No such LR scheduler {schedulers[i]['method']}"
            self._methods.append(
                _lr_methods[schedulers[i]['method']](
                    schedulers[i]['epochs'] * self.num_steps,
                    schedulers[i]['start'],
                    schedulers[i]['end'],
                    *schedulers[i].get('args', []),
                    **schedulers[i].get('kwargs', {}),
                )
            )
            assert schedulers[i]['epochs'] >= 0 and schedulers[i]['repeat'] > 0
            self._starts.append(self._starts[-1] + schedulers[i]['epochs'] * schedulers[i]['repeat'])
            if momentum:
                self._momentum_change.append(StepLR(schedulers[i]['epochs'] * self.num_steps, *momentum[i]))

        self._now_step = 0
        self._now_epoch = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the optimizer.
        """
        return {key: self.__dict__[key] for key in ['num_steps', '_now_step', '_now_epoch']}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, base_lr=None, step: int = None, epoch: int = None, return_momentum=False):
        step = self._now_step if step is None else step
        epoch = self._now_epoch if epoch is None else epoch
        if epoch >= self._starts[-1]:
            idx, epoch, step = -1, self._schedulers[-1]['epochs'], 0
        else:
            idx = bisect_right(self._starts, epoch, hi=len(self._methods)) - 1
            epoch = (epoch - self._starts[idx]) % self._schedulers[idx]['epochs']
        if base_lr is None:
            lr_scale = self.base_lr * self._methods[idx](epoch * self.num_steps + step)
        else:
            lr_scale = base_lr * self._methods[idx](epoch * self.num_steps + step)
        if return_momentum:
            return lr_scale, self._momentum_change[idx](epoch * self.num_steps + step)
        else:
            return lr_scale

    def step(self, step: int = None, epoch: int = 0):
        if step is None:
            self._now_step += 1
            if self._now_step == self.num_steps:
                self._now_step = 0
                self._now_epoch += 1
        else:
            self._now_step = step
            self._now_epoch = epoch
        if self._momentum_change:
            lr_scale, m_scale = self.get_lr(1.0, return_momentum=True)
        else:
            lr_scale = self.get_lr(1.0)
            m_scale = 1
        for optimizer in self._optimizers:
            for param_group in optimizer.param_groups:
                if not param_group.get('fix', False):
                    param_group['lr'] = param_group['initial_lr'] * lr_scale
                if 'initial_momentum' in param_group:
                    if 'momentum' in param_group:
                        param_group['momentum'] = param_group['initial_momentum'] * m_scale
                    elif 'betas' in param_group:
                        param_group['betas'] = (param_group['initial_momentum'] * m_scale, *param_group['betas'][1:])

    def __repr__(self):
        fs = "\n"
        step_len = len(str(self._starts[-1]))
        max_epoch_len = max(len(str(t['epochs'])) for t in self._schedulers)
        max_repeat_len = max(len(str(t['repeat'])) for t in self._schedulers)

        fmt_str = '\tlr: {}-->{} at epoch [{:%dd}, {:%dd}) in {:%dd} x {:%dd} epochs using {}\n' % (
            step_len, step_len, max_epoch_len, max_repeat_len
        )

        for i in range(len(self._methods)):
            if self._schedulers[i]['epochs'] == 0:
                continue
            fs += fmt_str.format(
                float2str(self._schedulers[i]['start']),
                float2str(self._schedulers[i]['end']),
                self._starts[i],
                self._starts[i + 1],
                self._schedulers[i]['epochs'],
                self._schedulers[i]['repeat'],
                self._methods[i],
            )
        fs += f"\tEvery epoch have {self.num_steps} steps"
        return fs

    def draw(self, total_epochs=100, log_scale=True, lr=1, check_func=None):
        import matplotlib.pyplot as plt
        lrs = []
        x = []
        lr_check = []
        for i in range(total_epochs):
            for step in range(self.num_steps):
                x.append(i + step / self.num_steps)
                lrs.append(self.get_lr(lr, step, i))
                if check_func is not None:
                    lr_check.append(check_func(step, self.num_steps, i, total_epochs) * lr)
        ax = plt.gca()
        if log_scale:
            ax.set_yscale('log')
        plt.plot(x, lrs)
        if check_func is not None:
            plt.plot(x, lr_check, ls='--')
        # plt.xticks(list(range(total_epochs)))
        plt.show()


def options(parser=None):
    # train learning rate
    _methods = list(_lr_methods.keys())
    group = get_parser(parser).add_argument_group("Learning rate scheduler Option:")
    group.add_argument("--lr", default=0.1, type=float, metavar="V", help="The base learning rate.")
    group.add_argument("--reference-lr", default=0, type=float, metavar="V", help="The base learning rate.")
    group.add_argument(
        '--reference-batch-size',
        type=int,
        default=256,
        metavar='N',
        help='Auto scale the learning rate by the ratio between reference and real batch size'
    )
    add_cfg_option(group, "--lr-schedulers", default=[['fix', 10000, 1]], help="The learning rate schedulers")
    return group


def make(optimizers: Union[Optimizer, List[Optimizer], Dict[str, Optimizer]], cfg, steps_per_epoch=1):
    if isinstance(optimizers, (list, tuple)):
        optimizers = list(optimizers)
    elif isinstance(optimizers, dict):
        optimizers = list(optimizers.values())
    else:
        optimizers = [optimizers]

    assert isinstance(cfg.lr_schedulers, (list, tuple))
    lr_schedulers = []
    for item in cfg.lr_schedulers:
        if isinstance(item, (list, tuple)):
            # format: [method, epoch, [start, [end, [repeat, ...]]], ]
            assert item[0] in _lr_methods, f"LR method {item[0]} is not supported!"
            n = len(item)
            assert n >= 2
            lr_schedulers.append({
                'method': item[0],
                'epochs': int(item[1]),
                'start': float(item[2]) if n > 2 else 1.,
                'end': float(item[3]) if n > 3 else (float(item[2]) if n > 2 else 1.),
                'repeat': int(item[4]) if n > 4 else 1,
            })
            if n > 5:
                lr_schedulers[-1]['args'] = item[5:]
        elif isinstance(item, dict):
            lr_schedulers.append({
                'method': item.pop('method', 'fix'),
                'epochs': int(item.pop('epochs', 1)),
                'start': float(item.pop('start', 1.)),
                'end': float(item.pop('end', 0.)),
                'repeat': int(item.pop('repeat', 1)),
            })
            lr_schedulers[-1]['kwargs'] = item
        else:
            raise ValueError('Not support format for lr schedulers.')
    scheduler = LRScheduler(optimizers, lr_schedulers, steps_per_epoch, lr=cfg.lr)
    logging.info('==> Learning Rate Scheduler: {}'.format(scheduler))
    return scheduler


def test():
    import torch
    import argparse

    parser_ = argparse.ArgumentParser('Scheduler Test')
    options(parser_)
    # parser_.print_help()
    cfg_ = parser_.parse_args([
        # '--lr-schedulers=[[step,1, 0.1, 1],[cos, 20, 0.01], [cos, 30, 0.1, 0.01, 2], '
        # '[ploy, 10, 1, 0.1, 1, 0.8], [exp2, 10, 1, 0.1]]',
        '--lr-scheduler=[[exp, 100, 1, 0.5]]'
    ])
    print(cfg_)
    optimizer_ = torch.optim.SGD(torch.nn.Linear(10, 10).parameters(), lr=0.1)
    scheduler_ = make(optimizer_, cfg_, 10)

    def check(step, steps, epoch, epochs):
        step = epoch * steps + step
        steps = steps * epochs
        return 2 ** (-step / steps)

    scheduler_.draw(100, log_scale=True, lr=cfg_.lr, check_func=check)
    print('state_dict:', scheduler_.state_dict())
