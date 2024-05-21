import functools

import torch

if torch.__version__ >= '1.6.0':
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def __call__(self, func):
            @functools.wraps(func)
            def decorate_autocast(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return decorate_autocast
