import torch
from torch import Tensor

from .torch_utils import tensor_to


def get_run_speed(inputs, grads, py_func=None, cu_func=None, cpu_func=None, num_test=100, **kwargs):
    if isinstance(inputs, Tensor):
        inputs = (inputs,)
    timer = [torch.cuda.Event(enable_timing=True) for _ in range(num_test * 3 + 3)]
    kwargs['cuda'] = (cu_func, 'cuda')
    kwargs['python'] = (py_func, 'cuda')
    kwargs['cpu'] = (cpu_func, 'cpu')
    for name, func_and_device in kwargs.items():
        if isinstance(func_and_device, (tuple, list)):
            func, device = func_and_device
        else:
            func, device = func_and_device, 'cuda'
        if func is None:
            continue
        inputs, grads = tensor_to(inputs, grads, device=torch.device(device))
        t_forward = 0
        t_backward = 0
        for step in range(num_test + 1):
            timer[step * 3 + 0].record()
            output = func(*inputs)
            timer[step * 3 + 1].record()
            if grads is not None:
                torch.autograd.backward(output, grads)
            timer[step * 3 + 2].record()
        for step in range(1, num_test + 1):
            timer[step * 3 + 0].synchronize()
            timer[step * 3 + 1].synchronize()
            timer[step * 3 + 2].synchronize()
            t_forward += timer[step * 3 + 0].elapsed_time(timer[step * 3 + 1])
            t_backward += timer[step * 3 + 1].elapsed_time(timer[step * 3 + 2])
        t_forward = t_forward / num_test
        t_backward = t_backward / num_test
        if grads is not None:
            print(f'{name:10s} time: forward {t_forward:.3f} ms, backward {t_backward:.3f} ms, '
                  f'total: {t_backward + t_forward:.3f} ms')
        else:
            print(f'{name:10s} time: forward {t_forward:.3f} ms')
