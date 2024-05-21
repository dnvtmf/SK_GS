from typing import List, Dict, Union, Tuple, Callable

from torch import nn


class Hook:
    _epoch: int
    _step: int
    _num_steps: int
    _num_epochs: int

    @staticmethod
    def set_total(steps, epochs):
        Hook._num_steps = steps
        Hook._num_epochs = epochs

    @staticmethod
    def set_state(step, epoch=None):
        Hook._step = step
        if epoch is not None:
            Hook._epoch = epoch

    def before_train(self, *args, **kwargs):
        pass

    def before_train_epoch(self, *args, **kwargs):
        pass

    def before_train_step(self, *args, **kwargs):
        pass

    def after_train_step(self, *args, **kwargs):
        pass

    def after_train_epoch(self, *args, **kwargs):
        pass

    def before_eval_epoch(self, *args, **kwargs):
        pass

    def before_eval_step(self, *args, **kwargs):
        pass

    def after_eval_step(self, *args, **kwargs):
        pass

    def after_eval_epoch(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @property
    def step(self):
        return self._step

    @property
    def epoch(self):
        return self._epoch

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def num_epochs(self):
        return self._num_epochs


class HookManager(Hook):

    def __init__(self):
        super(HookManager, self).__init__()
        self.hooks = {
            'before_train': [],
            'after_train': [],
            'before_train_epoch': [],
            'before_train_step': [],
            'after_train_step': [],
            'after_train_epoch': [],
            'before_eval_epoch': [],
            'before_eval_step': [],
            'after_eval_step': [],
            'after_eval_epoch': [],
        }  # type: Dict[str, List[Union[Callable, Tuple[Callable, tuple, Dict], Hook]]]

    def before_train(self, *args, **kwargs):
        for hook in self.hooks['before_train']:
            if isinstance(hook, Hook):
                hook.before_train(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def before_train_epoch(self, *args, **kwargs):
        for hook in self.hooks['before_train_epoch']:
            if isinstance(hook, Hook):
                hook.before_train_epoch(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def before_train_step(self, *args, **kwargs):
        for hook in self.hooks['before_train_step']:
            if isinstance(hook, Hook):
                hook.before_train_step(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def after_train_step(self, *args, **kwargs):
        for hook in self.hooks['after_train_step']:
            if isinstance(hook, Hook):
                hook.after_train_step(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def after_train_epoch(self, *args, **kwargs):
        for hook in self.hooks['after_train_epoch']:
            if isinstance(hook, Hook):
                hook.after_train_epoch(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def before_eval_epoch(self, *args, **kwargs):
        for hook in self.hooks['before_eval_epoch']:
            if isinstance(hook, Hook):
                hook.before_eval_epoch(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def before_eval_step(self, *args, **kwargs):
        for hook in self.hooks['before_eval_step']:
            if isinstance(hook, Hook):
                hook.before_eval_step(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def after_eval_step(self, *args, **kwargs):
        for hook in self.hooks['after_eval_step']:
            if isinstance(hook, Hook):
                hook.after_eval_step(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def after_eval_epoch(self, *args, **kwargs):
        for hook in self.hooks['after_eval_epoch']:
            if isinstance(hook, Hook):
                hook.after_eval_epoch(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def after_train(self, *args, **kwargs):
        for hook in self.hooks['after_train']:
            if isinstance(hook, Hook):
                hook.after_train(*args, **kwargs)
            elif isinstance(hook, Callable):
                hook(*args, **kwargs)
            else:
                hook, user_args, user_kwargs = hook
                hook(*args, *user_args, **kwargs, **user_kwargs)

    def add_hook(self, hook: Union[Callable, Hook], hook_type='before_train_epoch', insert=None, *args, **kwargs):
        assert hook_type in self.hooks
        if not isinstance(hook, Hook):
            hook = (hook, args, kwargs)
        if insert is not None:
            self.hooks[hook_type].insert(insert, hook)
        else:
            self.hooks[hook_type].append(hook)

    def add_module_hooks(self, *modules: nn.Module):
        num_add_hook = 0

        def _add_hook(m: nn.Module):
            nonlocal num_add_hook
            for hook_type in self.hooks.keys():
                name = 'hook_' + hook_type
                if hasattr(m, name):
                    self.add_hook(getattr(m, name), hook_type)
                    num_add_hook += 1

        for m in modules:
            m.apply(_add_hook)
        return num_add_hook
