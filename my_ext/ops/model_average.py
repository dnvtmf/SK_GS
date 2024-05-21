from copy import deepcopy
import math
from contextlib import contextmanager
import logging

import torch
import torch.nn as nn

from my_ext.utils.torch_utils import get_net
from my_ext.ops.amp_autocast import autocast


class AveragedModel(nn.Module):
    def __init__(
        self, model: nn.Module, enable=True, decay=0.9999, step=1, epoch=-1, start_epoch=-1,
        update_bn=True, device=None, adjust_decay=True
    ):
        super(AveragedModel, self).__init__()
        self.enable = enable
        self.device = device
        self.step_interval = step if epoch <= 0 else -1
        self.epoch_interval = epoch
        self._step_count = 0
        self._epoch_count = 0
        self.decay = -1 if self.epoch_interval > 0 else decay
        self.start_epoch = start_epoch
        self.is_update_bn = update_bn
        self.adjust_decay = adjust_decay

        if enable:
            self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long, device=device))
            self.averaged_model = deepcopy(get_net(model)).to(device=self.device)
            # for p in self.averaged_model.parameters():
            #     p.requires_grad_(False)
        else:
            self.register_buffer('n_averaged', None)
            self.averaged_model = None
        self.eval()
        self._updated = False

    # def _sync_params_and_buffers(self):
    #     if not self._updated:
    #         return
    #     for m in self.averaged_model.parameters():
    #         m.data.copy_(broadcast(m.data.clone(), 0))
    #     for m in self.averaged_model.buffers():
    #         m.data.copy_(broadcast(m.data.clone(), 0))
    #     self._updated = False

    def forward(self, *args, **kwargs):
        # raise NotImplementedError
        # self._sync_params_and_buffers()
        return self.averaged_model(*args, **kwargs)

    @torch.no_grad()
    def _update(self, model: nn.Module, alpha=0.):
        model = get_net(model)
        # for (name_e, w_e), (name_m, w_m) in zip(self.averaged_model.state_dict().items(), model.state_dict().items()):
        #     assert name_e == name_m and w_e.shape == w_m.shape
        #     w_m = w_m.to(w_e.device)
        #     if w_e.dtype.is_floating_point:
        #         w_e.data.mul_(alpha).add_(w_m.detach(), alpha=1. - alpha)
        #     else:
        #         if not name_e.endswith('.num_batches_tracked'):
        #             print(name_e)
        #         w_e.copy_(w_m)
        for w_e, w_m in zip(self.averaged_model.parameters(), model.parameters()):
            w_e.data.mul_(alpha).add_(w_m.detach().to(w_e.device), alpha=1. - alpha)
        self.n_averaged += 1
        self._updated = True

    def update(self, model: nn.Module, epoch: int = None, is_epoch_end=False, reset=False):
        if not self.enable or (epoch is not None and epoch < self.start_epoch):
            return
        if reset:
            self.n_averaged.data.zero_()
            self._epoch_count = 0
            self._step_count = 0
        if is_epoch_end:
            self._epoch_count += 1
            if not (self.epoch_interval > 0 and self._epoch_count % self.epoch_interval == 0):
                return
        else:
            self._step_count += 1
            if not (self.step_interval > 0 and self._step_count % self.step_interval == 0):
                return
        if self.n_averaged == 0:
            alpha = 0
        elif self.decay > 0:
            alpha = self.decay
            if self.adjust_decay:
                alpha *= (1 - math.exp(-(self.n_averaged.item() + 1) / 2000))
        else:
            alpha = 1. - 1. / (self.n_averaged.item() + 1)
        self._update(model, alpha)

    @torch.no_grad()
    def update_bn(self, model: nn.Module, loader, get_input_fn=None, fp16=False):
        if not (self.enable and self.is_update_bn):
            return
        state_dict = self.load_weights(model)
        logging.info('Update BN for averaged model')
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return state_dict

        was_training = model.training
        model.train()
        model.cuda()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0
        logging.info('initialize the parameters of BN')
        if get_input_fn is None:
            get_input_fn = lambda d: (d[0] if isinstance(data, (list, tuple)) else d).cuda()
        for i, data in enumerate(loader, 1):
            with autocast(fp16):
                model(get_input_fn(data))
            logging.info(f"\rupdate bn [{i}/{len(loader)}]")
        logging.info('')

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)
        self.update(model, reset=True)
        return state_dict

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.averaged_model, k, v)

    def load_weights(self, model: nn.Module):
        if not self.enable:
            return None
        st = deepcopy(model.state_dict())
        get_net(model).load_state_dict(self.averaged_model.state_dict())
        return st

    def restore_weights(self, model: nn.Module, state_dict):
        if self.enable and state_dict is not None:
            model.load_state_dict(state_dict)

    @contextmanager
    def temp_load(self, model: nn.Module, enable=True):
        state_dict = self.load_weights(model) if enable else None
        try:
            yield
        finally:
            if enable:
                self.restore_weights(model, state_dict)
        return

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"decay={self.decay}" if self.decay > 0 else "average"
        if self.epoch_interval > 0:
            s += f", every {self.epoch_interval} epochs"
        else:
            s += f", every {self.step_interval} steps"
        s += f", start at epoch {self.start_epoch}"
        s += ")"
        return s
